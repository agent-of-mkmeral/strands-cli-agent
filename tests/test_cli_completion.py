"""Tests for CLI completion watcher and agent re-invocation.

Tests the key behavior: when a task completes, the agent is automatically
re-invoked with the result, even if the user is idle at the prompt.
"""

import queue
import threading
import time
from unittest.mock import MagicMock, patch, call

import pytest

from strands_cli_agent.task_manager import TrackedTask


@pytest.fixture
def sample_task():
    """A completed task with result."""
    return TrackedTask(
        task_id="task-test-123",
        server_name="containerized-strands-agents",
        tool_name="send_message",
        agent_id="researcher",
        status="completed",
        status_message="Done",
        result={
            "content": [
                {"type": "text", "text": "Found 12 open MCP issues. Top ones: ..."}
            ]
        },
        original_user_message="Research all open MCP issues",
    )


@pytest.fixture
def failed_task():
    """A failed task."""
    return TrackedTask(
        task_id="task-fail-456",
        server_name="containerized-strands-agents",
        tool_name="send_message",
        agent_id="coder",
        status="failed",
        status_message="Container OOM killed",
        original_user_message="Write a complex algorithm",
    )


class TestFormatTaskResult:
    def test_completed_with_text_result(self, sample_task):
        from strands_cli_agent.cli import _format_task_result

        result = _format_task_result(sample_task)
        assert "[Task Completed]" in result
        assert "researcher" in result
        assert "send_message" in result
        assert "completed" in result
        assert "Found 12 open MCP issues" in result
        assert "Research all open MCP issues" in result

    def test_completed_with_dict_result(self):
        from strands_cli_agent.cli import _format_task_result

        task = TrackedTask(
            task_id="task-dict",
            server_name="s",
            tool_name="t",
            status="completed",
            result={"summary": "all good"},
        )
        result = _format_task_result(task)
        assert "[Task Completed]" in result
        assert "all good" in result

    def test_failed_task(self, failed_task):
        from strands_cli_agent.cli import _format_task_result

        result = _format_task_result(failed_task)
        assert "[Task Completed]" in result  # It still says Task Completed in header
        assert "failed" in result
        assert "Container OOM killed" in result

    def test_no_result(self):
        from strands_cli_agent.cli import _format_task_result

        task = TrackedTask(
            task_id="task-noresult",
            server_name="s",
            tool_name="t",
            status="completed",
            result=None,
        )
        result = _format_task_result(task)
        assert "[Task Completed]" in result

    def test_no_user_message(self):
        from strands_cli_agent.cli import _format_task_result

        task = TrackedTask(
            task_id="task-nomsg",
            server_name="s",
            tool_name="t",
            status="completed",
            original_user_message=None,
        )
        result = _format_task_result(task)
        assert "Original request" not in result

    def test_uses_agent_id_as_label(self, sample_task):
        from strands_cli_agent.cli import _format_task_result

        result = _format_task_result(sample_task)
        assert "researcher" in result

    def test_falls_back_to_server_name(self):
        from strands_cli_agent.cli import _format_task_result

        task = TrackedTask(
            task_id="task-noagent",
            server_name="my-server",
            tool_name="t",
            agent_id=None,
            status="completed",
        )
        result = _format_task_result(task)
        assert "my-server" in result


class TestCompletionWatcher:
    """Tests for the completion watcher thread behavior."""

    def test_watcher_processes_completion(self, sample_task):
        """Watcher should call agent() when a task completes."""
        from strands_cli_agent.cli import (
            _completion_queue,
            _completion_watcher_fn,
            _stop_watcher,
            _agent_idle,
            _agent_lock,
        )

        mock_agent = MagicMock()
        _stop_watcher.clear()
        _agent_idle.set()

        # Put a completion in the queue
        _completion_queue.put(sample_task)

        # Run watcher in a thread
        watcher = threading.Thread(
            target=_completion_watcher_fn,
            args=(mock_agent,),
            daemon=True,
        )
        watcher.start()

        # Wait for processing
        time.sleep(2.0)

        # Stop watcher
        _stop_watcher.set()
        watcher.join(timeout=3)

        # Agent should have been called with the task result
        assert mock_agent.call_count >= 1
        call_args = mock_agent.call_args[0][0]
        assert "[Task Completed]" in call_args
        assert "researcher" in call_args
        assert "Found 12 open MCP issues" in call_args

    def test_watcher_waits_when_agent_busy(self, sample_task):
        """Watcher should wait for agent to be idle before re-invoking."""
        from strands_cli_agent.cli import (
            _completion_queue,
            _completion_watcher_fn,
            _stop_watcher,
            _agent_idle,
            _agent_lock,
        )

        mock_agent = MagicMock()
        _stop_watcher.clear()

        # Mark agent as busy
        _agent_idle.clear()

        # Put a completion in the queue
        _completion_queue.put(sample_task)

        # Run watcher
        watcher = threading.Thread(
            target=_completion_watcher_fn,
            args=(mock_agent,),
            daemon=True,
        )
        watcher.start()

        # Give watcher time to start — it should be waiting
        time.sleep(1.0)
        assert mock_agent.call_count == 0, "Agent should NOT be called while busy"

        # Now mark agent as idle
        _agent_idle.set()

        # Wait for processing
        time.sleep(2.0)

        _stop_watcher.set()
        watcher.join(timeout=3)

        assert mock_agent.call_count >= 1, "Agent should be called after becoming idle"

    def test_watcher_processes_multiple_completions(self):
        """Multiple completions are processed sequentially."""
        from strands_cli_agent.cli import (
            _completion_queue,
            _completion_watcher_fn,
            _stop_watcher,
            _agent_idle,
        )

        mock_agent = MagicMock()
        _stop_watcher.clear()
        _agent_idle.set()

        # Queue multiple completions
        for i in range(3):
            _completion_queue.put(TrackedTask(
                task_id=f"task-multi-{i}",
                server_name="s",
                tool_name="t",
                agent_id=f"agent-{i}",
                status="completed",
                result={"content": [{"type": "text", "text": f"Result {i}"}]},
            ))

        watcher = threading.Thread(
            target=_completion_watcher_fn,
            args=(mock_agent,),
            daemon=True,
        )
        watcher.start()
        time.sleep(3.0)

        _stop_watcher.set()
        watcher.join(timeout=3)

        assert mock_agent.call_count == 3

    def test_watcher_handles_agent_error(self, sample_task):
        """Watcher should not crash if agent() raises an exception."""
        from strands_cli_agent.cli import (
            _completion_queue,
            _completion_watcher_fn,
            _stop_watcher,
            _agent_idle,
        )

        mock_agent = MagicMock(side_effect=Exception("Model error"))
        _stop_watcher.clear()
        _agent_idle.set()

        _completion_queue.put(sample_task)

        # Add a second task to prove the watcher continues after the error
        _completion_queue.put(TrackedTask(
            task_id="task-after-error",
            server_name="s",
            tool_name="t",
            status="completed",
        ))

        watcher = threading.Thread(
            target=_completion_watcher_fn,
            args=(mock_agent,),
            daemon=True,
        )
        watcher.start()
        time.sleep(3.0)

        _stop_watcher.set()
        watcher.join(timeout=3)

        # Both tasks should have been attempted
        assert mock_agent.call_count == 2

    def test_watcher_stops_cleanly(self):
        """Watcher should stop when _stop_watcher is set."""
        from strands_cli_agent.cli import (
            _completion_watcher_fn,
            _stop_watcher,
            _agent_idle,
        )

        mock_agent = MagicMock()
        _stop_watcher.clear()
        _agent_idle.set()

        watcher = threading.Thread(
            target=_completion_watcher_fn,
            args=(mock_agent,),
            daemon=True,
        )
        watcher.start()

        time.sleep(0.5)
        _stop_watcher.set()
        watcher.join(timeout=3)

        assert not watcher.is_alive()


class TestAgentLockSerialization:
    """Tests that _agent_lock properly serializes agent calls."""

    def test_lock_prevents_concurrent_calls(self, sample_task):
        """Agent lock should prevent user input and completion from running concurrently."""
        from strands_cli_agent.cli import _agent_lock, _agent_idle

        call_log = []
        call_lock = threading.Lock()

        def mock_agent_slow(msg):
            with call_lock:
                call_log.append(("start", msg[:20], time.time()))
            time.sleep(0.5)
            with call_lock:
                call_log.append(("end", msg[:20], time.time()))

        _agent_idle.set()

        # Simulate user message (holds lock for 0.5s)
        def user_call():
            with _agent_lock:
                _agent_idle.clear()
                try:
                    mock_agent_slow("user message")
                finally:
                    _agent_idle.set()

        # Simulate completion (also needs lock)
        def completion_call():
            time.sleep(0.1)  # Start slightly after user
            with _agent_lock:
                _agent_idle.clear()
                try:
                    mock_agent_slow("completion result")
                finally:
                    _agent_idle.set()

        t1 = threading.Thread(target=user_call)
        t2 = threading.Thread(target=completion_call)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        # Verify: first call should end before second call starts
        assert len(call_log) == 4
        # First end should be before second start
        first_end_time = call_log[1][2]
        second_start_time = call_log[2][2]
        assert first_end_time <= second_start_time, "Calls should be serialized"


class TestOnTaskCallbacks:
    def test_on_task_completed_queues(self, sample_task):
        """_on_task_completed should put task in the completion queue."""
        from strands_cli_agent.cli import _on_task_completed, _completion_queue

        # Drain queue first
        while not _completion_queue.empty():
            _completion_queue.get_nowait()

        with patch.object(
            __import__("strands_cli_agent.handlers.callback_handler", fromlist=["callback_handler_instance"]).callback_handler_instance,
            "on_task_completed",
        ):
            _on_task_completed(sample_task)

        assert not _completion_queue.empty()
        queued = _completion_queue.get_nowait()
        assert queued.task_id == "task-test-123"

    def test_on_task_failed_queues(self, failed_task):
        """_on_task_failed should put task in the completion queue."""
        from strands_cli_agent.cli import _on_task_failed, _completion_queue

        # Drain queue first
        while not _completion_queue.empty():
            _completion_queue.get_nowait()

        with patch.object(
            __import__("strands_cli_agent.handlers.callback_handler", fromlist=["callback_handler_instance"]).callback_handler_instance,
            "on_task_failed",
        ):
            _on_task_failed(failed_task)

        assert not _completion_queue.empty()
        queued = _completion_queue.get_nowait()
        assert queued.task_id == "task-fail-456"
