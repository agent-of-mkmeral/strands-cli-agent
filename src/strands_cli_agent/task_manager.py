"""Task Manager — tracks MCP tasks and re-triggers agent on completion.

This is the core differentiator of strands-cli-agent. When the agent calls a tool
on an MCP server that returns a Task (fire-and-forget), the TaskManager:

1. Records the task (taskId, server, tool name, original context)
2. Starts a background poller for that task
3. When the task completes, injects the result back into the agent conversation
4. The agent processes the result automatically — push, not pull

This enables the pattern where:
  - User: "Research all open MCP issues"
  - Agent calls send_message("researcher", ...) on containerized-strands-agents
  - Server returns Task {taskId: "task-123", status: "working"}
  - Agent responds: "I've dispatched that. I'll notify you when it's done."
  - ... time passes ...
  - TaskManager detects task completed → re-triggers agent with the result
  - Agent: "The researcher found 12 issues. Here's the summary: ..."
"""

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class TrackedTask:
    """A task being tracked by the manager."""

    task_id: str
    server_name: str  # MCP server prefix (e.g., "containerized-strands-agents")
    tool_name: str  # The tool that was called (e.g., "send_message")
    agent_id: str | None = None  # Agent within the server (e.g., "researcher")
    arguments: dict[str, Any] | None = None
    status: str = "working"  # working, completed, failed, cancelled
    status_message: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    result: dict[str, Any] | None = None
    poll_interval_ms: int = 5000
    # Context for re-triggering agent
    original_user_message: str | None = None


class TaskManager:
    """Manages MCP task lifecycle with push-based completion handling.

    When a tool call returns an MCP Task, the TaskManager tracks it and
    polls for completion via the MCP tasks/get protocol endpoint (using
    the MCPClient's internal session). When the task finishes, it calls
    the completion callback to re-trigger the agent.

    The key design decision: task completion feeds back into the agent
    automatically. The user doesn't need to ask "is it done?" — the agent
    tells them.
    """

    def __init__(
        self,
        state_file: Path | None = None,
        on_task_completed: Callable[[TrackedTask], None] | None = None,
        on_task_failed: Callable[[TrackedTask], None] | None = None,
        on_task_status_changed: Callable[[TrackedTask], None] | None = None,
    ):
        self._tasks: dict[str, TrackedTask] = {}
        self._state_file = state_file or Path.home() / ".strands-cli" / "tasks.json"
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        # Callbacks
        self._on_completed = on_task_completed
        self._on_failed = on_task_failed
        self._on_status_changed = on_task_status_changed

        # Background polling
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # MCPClient references for protocol-level task polling
        # Maps server_name (prefix) → MCPClient instance
        self._mcp_clients: dict[str, Any] = {}

        # Current user message for context
        self._current_user_message: str | None = None

        # Load existing tasks
        self._load()

    def _load(self) -> None:
        """Load tracked tasks from disk."""
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
            for task_id, entry in data.items():
                self._tasks[task_id] = TrackedTask(**entry)
            logger.info(f"Loaded {len(self._tasks)} tracked tasks from {self._state_file}")
        except Exception as e:
            logger.warning(f"Failed to load tasks from {self._state_file}: {e}")

    def _save(self) -> None:
        """Persist tracked tasks to disk."""
        try:
            data = {}
            for task_id, task in self._tasks.items():
                data[task_id] = {
                    "task_id": task.task_id,
                    "server_name": task.server_name,
                    "tool_name": task.tool_name,
                    "agent_id": task.agent_id,
                    "arguments": task.arguments,
                    "status": task.status,
                    "status_message": task.status_message,
                    "created_at": task.created_at,
                    "completed_at": task.completed_at,
                    "result": task.result,
                    "poll_interval_ms": task.poll_interval_ms,
                    "original_user_message": task.original_user_message,
                }
            self._state_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.warning(f"Failed to save tasks: {e}")

    def register_mcp_client(self, server_name: str, mcp_client: Any) -> None:
        """Register an MCPClient for MCP protocol-level task polling.

        The MCPClient's internal session is used to call tasks/get on the
        server to check task status. This is more reliable than custom
        polling because it uses the standard MCP Tasks protocol.

        Args:
            server_name: The MCP server prefix (e.g., "containerized-strands-agents")
            mcp_client: The Strands MCPClient instance
        """
        self._mcp_clients[server_name] = mcp_client
        logger.info(f"Registered MCPClient for task polling: {server_name}")

    def track_task(
        self,
        task_id: str,
        server_name: str,
        tool_name: str,
        agent_id: str | None = None,
        arguments: dict[str, Any] | None = None,
        poll_interval_ms: int = 5000,
        original_user_message: str | None = None,
    ) -> TrackedTask:
        """Start tracking a new task.

        Args:
            task_id: The MCP task ID returned by the server
            server_name: Which MCP server prefix created this task
            tool_name: The tool that was called
            agent_id: The agent within the server (e.g., "researcher")
            arguments: Tool arguments (for context)
            poll_interval_ms: How often to poll (from server hint)
            original_user_message: The user's original message (for re-triggering context)

        Returns:
            The TrackedTask object
        """
        with self._lock:
            # Don't re-track existing tasks
            if task_id in self._tasks:
                logger.debug(f"Task {task_id} already tracked, skipping")
                return self._tasks[task_id]

            task = TrackedTask(
                task_id=task_id,
                server_name=server_name,
                tool_name=tool_name,
                agent_id=agent_id,
                arguments=arguments,
                status="working",
                poll_interval_ms=poll_interval_ms,
                original_user_message=original_user_message or self._current_user_message,
            )
            self._tasks[task_id] = task
            self._save()
            logger.info(f"Tracking task {task_id} from {server_name}/{tool_name} (agent: {agent_id})")

            # Ensure polling is running
            self._ensure_polling()

            return task

    def get_task(self, task_id: str) -> TrackedTask | None:
        """Get a tracked task by ID."""
        return self._tasks.get(task_id)

    def get_active_tasks(self) -> list[TrackedTask]:
        """Get all tasks that are still working."""
        return [t for t in self._tasks.values() if t.status == "working"]

    def get_all_tasks(self) -> list[TrackedTask]:
        """Get all tracked tasks."""
        return list(self._tasks.values())

    def _ensure_polling(self) -> None:
        """Start the background polling thread if not already running."""
        if self._poll_thread and self._poll_thread.is_alive():
            return

        self._stop_event.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True, name="task-poller")
        self._poll_thread.start()
        logger.info("Started task polling thread")

    def _poll_loop(self) -> None:
        """Background thread that polls active tasks."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_poll_loop())
        finally:
            loop.close()

    async def _async_poll_loop(self) -> None:
        """Async polling loop — checks all active tasks."""
        while not self._stop_event.is_set():
            active_tasks = self.get_active_tasks()
            if not active_tasks:
                # No active tasks — sleep briefly and check again
                await asyncio.sleep(2)
                continue

            for task in active_tasks:
                if self._stop_event.is_set():
                    break
                try:
                    await self._poll_task(task)
                except Exception as e:
                    logger.warning(f"Error polling task {task.task_id}: {e}")

            # Sleep for the minimum poll interval among active tasks
            min_interval = min(t.poll_interval_ms for t in active_tasks) / 1000.0
            min_interval = max(min_interval, 2.0)  # At least 2 seconds
            await asyncio.sleep(min_interval)

    async def _poll_task(self, task: TrackedTask) -> None:
        """Poll a single task's status via MCP protocol tasks/get.

        Uses the MCPClient's internal session to call the standard MCP
        tasks/get endpoint. Falls back gracefully if the session isn't
        available.
        """
        mcp_client = self._mcp_clients.get(task.server_name)
        if not mcp_client:
            logger.debug(f"No MCPClient for server '{task.server_name}', cannot poll task {task.task_id}")
            return

        try:
            # Access MCPClient's internal session and event loop
            session = getattr(mcp_client, '_background_thread_session', None)
            event_loop = getattr(mcp_client, '_background_thread_event_loop', None)

            if not session or not event_loop:
                logger.debug(f"MCPClient session not available for {task.server_name}")
                return

            # Submit tasks/get request to MCPClient's event loop
            future = asyncio.run_coroutine_threadsafe(
                session.experimental.get_task(task.task_id),
                event_loop,
            )
            result = future.result(timeout=15)

            old_status = task.status
            task.status = result.status
            task.status_message = getattr(result, 'statusMessage', None)

            if hasattr(result, 'pollInterval') and result.pollInterval:
                task.poll_interval_ms = result.pollInterval

            if old_status != task.status:
                logger.info(f"Task {task.task_id} status: {old_status} → {task.status}")
                if self._on_status_changed:
                    self._on_status_changed(task)

            # Terminal states
            if task.status in ("completed", "failed", "cancelled"):
                task.completed_at = datetime.now(timezone.utc).isoformat()

                # Get the task result/payload for completed tasks
                if task.status == "completed":
                    try:
                        from mcp.types import GetTaskPayloadResult

                        payload_future = asyncio.run_coroutine_threadsafe(
                            session.experimental.get_task_result(task.task_id, GetTaskPayloadResult),
                            event_loop,
                        )
                        payload = payload_future.result(timeout=15)

                        if hasattr(payload, "content") and payload.content:
                            task.result = {
                                "content": [
                                    c.model_dump() if hasattr(c, "model_dump") else c
                                    for c in payload.content
                                ]
                            }
                        elif hasattr(payload, "model_dump"):
                            task.result = payload.model_dump()
                    except Exception as e:
                        logger.warning(f"Failed to get result for task {task.task_id}: {e}")
                        task.result = {"error": str(e)}

                with self._lock:
                    self._save()

                # Fire callbacks
                if task.status == "completed" and self._on_completed:
                    try:
                        self._on_completed(task)
                    except Exception as e:
                        logger.error(f"Completion callback error for {task.task_id}: {e}")
                elif task.status in ("failed", "cancelled") and self._on_failed:
                    try:
                        self._on_failed(task)
                    except Exception as e:
                        logger.error(f"Failure callback error for {task.task_id}: {e}")

        except Exception as e:
            logger.debug(f"Poll failed for task {task.task_id}: {e}")

    def handle_task_notification(self, server_name: str, task_id: str, status: str, status_message: str | None = None) -> None:
        """Handle an incoming MCP task status notification (push from server).

        This is the push path — the server sends notifications/tasks/status
        and we update our tracked task immediately instead of waiting for a poll.

        Args:
            server_name: Which server sent the notification
            task_id: The task ID
            status: New status (working, completed, failed, cancelled)
            status_message: Optional status message
        """
        task = self._tasks.get(task_id)
        if not task:
            logger.debug(f"Received notification for unknown task {task_id}")
            return

        old_status = task.status
        task.status = status
        task.status_message = status_message

        if old_status != status:
            logger.info(f"[notification] Task {task_id} status: {old_status} → {status}")

            if self._on_status_changed:
                self._on_status_changed(task)

            if status in ("completed", "failed", "cancelled"):
                task.completed_at = datetime.now(timezone.utc).isoformat()

                with self._lock:
                    self._save()

                if status == "completed" and self._on_completed:
                    # Need to fetch result async — schedule it
                    threading.Thread(
                        target=self._fetch_result_and_callback,
                        args=(task,),
                        daemon=True,
                    ).start()
                elif self._on_failed:
                    try:
                        self._on_failed(task)
                    except Exception as e:
                        logger.error(f"Failure callback error for {task_id}: {e}")

    def _fetch_result_and_callback(self, task: TrackedTask) -> None:
        """Fetch task result via MCPClient and fire completion callback."""
        mcp_client = self._mcp_clients.get(task.server_name)
        if not mcp_client:
            # No client, just fire callback without result
            if self._on_completed:
                try:
                    self._on_completed(task)
                except Exception as e:
                    logger.error(f"Completion callback error for {task.task_id}: {e}")
            return

        try:
            session = getattr(mcp_client, '_background_thread_session', None)
            event_loop = getattr(mcp_client, '_background_thread_event_loop', None)

            if session and event_loop:
                from mcp.types import GetTaskPayloadResult

                future = asyncio.run_coroutine_threadsafe(
                    session.experimental.get_task_result(task.task_id, GetTaskPayloadResult),
                    event_loop,
                )
                payload = future.result(timeout=15)

                if hasattr(payload, "content") and payload.content:
                    task.result = {
                        "content": [
                            c.model_dump() if hasattr(c, "model_dump") else c
                            for c in payload.content
                        ]
                    }
        except Exception as e:
            logger.warning(f"Failed to fetch result for task {task.task_id}: {e}")
            task.result = {"error": str(e)}

        with self._lock:
            self._save()

        if self._on_completed:
            try:
                self._on_completed(task)
            except Exception as e:
                logger.error(f"Completion callback error for {task.task_id}: {e}")

    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop_event.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=5)
            logger.info("Task polling thread stopped")

    def clear_completed(self) -> int:
        """Remove completed/failed/cancelled tasks from tracking.

        Returns:
            Number of tasks removed
        """
        with self._lock:
            to_remove = [tid for tid, t in self._tasks.items() if t.status in ("completed", "failed", "cancelled")]
            for tid in to_remove:
                del self._tasks[tid]
            self._save()
            return len(to_remove)
