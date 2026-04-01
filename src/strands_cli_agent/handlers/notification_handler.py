"""Notification handler — bridges MCP server notifications to the CLI and TaskManager.

When connected to MCP servers, this handler processes:
1. notifications/tasks/status → Updates TaskManager, triggers agent re-invocation
2. notifications/message (logging) → Displays real-time server logs in the CLI
3. notifications/progress → Shows progress bars for long-running operations
"""

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from strands_cli_agent.task_manager import TaskManager

logger = logging.getLogger(__name__)
console = Console()


class NotificationHandler:
    """Handles MCP server notifications and routes them appropriately.

    This is the push-based notification receiver. When an MCP server sends
    a notification, this handler:
    - For task status: Updates the TaskManager which may re-trigger the agent
    - For logging: Displays in the CLI with appropriate formatting
    - For progress: Shows progress indicators

    Usage:
        handler = NotificationHandler(task_manager=task_manager)
        # Pass handler.on_notification as the message_handler to MCPClient
    """

    def __init__(
        self,
        task_manager: "TaskManager | None" = None,
        show_logs: bool = True,
        show_progress: bool = True,
    ):
        self._task_manager = task_manager
        self._show_logs = show_logs
        self._show_progress = show_progress
        # Track which server each notification comes from
        self._current_server: str = "unknown"

    def set_server_name(self, name: str) -> None:
        """Set the current server name for attribution."""
        self._current_server = name

    async def on_logging(self, params: Any) -> None:
        """Handle notifications/message (logging from MCP server).

        These are real-time log messages from the server — e.g.,
        "Agent researcher: Cloning repository..." or "Agent coder: Tests passed ✓"
        """
        if not self._show_logs:
            return

        level = getattr(params, "level", "info")
        data = getattr(params, "data", "")

        # Format based on log level
        color_map = {
            "debug": "dim",
            "info": "cyan",
            "notice": "blue",
            "warning": "yellow",
            "error": "red",
            "critical": "bold red",
            "alert": "bold red",
            "emergency": "bold red",
        }
        color = color_map.get(level, "white")

        # Display the log message
        server_tag = f"[dim]\\[{self._current_server}][/dim] "
        console.print(f"  {server_tag}[{color}]{data}[/{color}]")

    def on_task_status(self, server_name: str, params: Any) -> None:
        """Handle notifications/tasks/status from MCP server.

        This is the PUSH path for task updates. Instead of polling,
        the server proactively tells us when a task's status changes.

        Args:
            server_name: Which MCP server sent this notification
            params: TaskStatusNotificationParams with taskId, status, statusMessage, etc.
        """
        task_id = getattr(params, "taskId", None)
        status = getattr(params, "status", None)
        status_message = getattr(params, "statusMessage", None)

        if not task_id or not status:
            logger.warning(f"Malformed task notification from {server_name}: {params}")
            return

        logger.info(f"📨 Task notification from {server_name}: {task_id} → {status}")

        # Display in CLI
        status_emoji = {
            "working": "⏳",
            "completed": "✅",
            "failed": "❌",
            "cancelled": "🚫",
            "input_required": "❓",
        }
        emoji = status_emoji.get(status, "📋")
        console.print(
            f"\n  {emoji} [bold]Task Update[/bold] "
            f"[dim]({server_name})[/dim]: "
            f"[cyan]{task_id[:12]}...[/cyan] → "
            f"[{'green' if status == 'completed' else 'yellow' if status == 'working' else 'red'}]{status}[/]"
        )
        if status_message:
            console.print(f"     [dim]{status_message}[/dim]")

        # Route to TaskManager for tracking + agent re-trigger
        if self._task_manager:
            self._task_manager.handle_task_notification(
                server_name=server_name,
                task_id=task_id,
                status=status,
                status_message=status_message,
            )

    async def on_message(self, message: Any) -> None:
        """Generic message handler for MCP ClientSession.

        This handles ALL incoming messages from the server, including
        notifications. We route task status notifications to the TaskManager.
        """
        # Check if it's a server notification
        from mcp import types as mcp_types

        if isinstance(message, mcp_types.ServerNotification):
            notification = message
        elif hasattr(message, "root") and isinstance(getattr(message, "root", None), mcp_types.ServerNotification):
            notification = message.root
        else:
            # Not a notification — might be a request, let default handler deal with it
            return

        # Route by notification type
        root = notification.root if hasattr(notification, "root") else notification
        if isinstance(root, mcp_types.TaskStatusNotification):
            self.on_task_status(self._current_server, root.params)
        elif isinstance(root, mcp_types.LoggingMessageNotification):
            await self.on_logging(root.params)
        elif isinstance(root, mcp_types.ProgressNotification):
            if self._show_progress:
                params = root.params
                progress = getattr(params, "progress", 0)
                total = getattr(params, "total", None)
                if total:
                    pct = int(progress / total * 100)
                    console.print(f"  [dim]\\[{self._current_server}][/dim] Progress: {pct}% ({progress}/{total})")
