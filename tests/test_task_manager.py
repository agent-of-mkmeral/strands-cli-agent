"""Tests for TaskManager with MCPClient-based polling."""

import json
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from strands_cli_agent.task_manager import TaskManager, TrackedTask


@pytest.fixture
def tmp_state_file(tmp_path):
    return tmp_path / "tasks.json"


@pytest.fixture
def task_manager(tmp_state_file):
    tm = TaskManager(state_file=tmp_state_file)
    yield tm
    tm.stop()


class TestTrackedTask:
    def test_defaults(self):
        task = TrackedTask(
            task_id="task-123",
            server_name="my-server",
            tool_name="send_message",
        )
        assert task.task_id == "task-123"
        assert task.server_name == "my-server"
        assert task.tool_name == "send_message"
        assert task.agent_id is None
        assert task.status == "working"
        assert task.result is None
        assert task.poll_interval_ms == 5000

    def test_with_agent_id(self):
        task = TrackedTask(
            task_id="task-456",
            server_name="agent-host",
            tool_name="send_message",
            agent_id="researcher",
        )
        assert task.agent_id == "researcher"


class TestTaskManagerTracking:
    def test_track_new_task(self, task_manager):
        task = task_manager.track_task(
            task_id="task-abc",
            server_name="test-server",
            tool_name="test_tool",
            agent_id="agent-1",
        )
        assert task.task_id == "task-abc"
        assert task.server_name == "test-server"
        assert task.tool_name == "test_tool"
        assert task.agent_id == "agent-1"
        assert task.status == "working"

    def test_track_duplicate_returns_existing(self, task_manager):
        task1 = task_manager.track_task(
            task_id="task-dup",
            server_name="s",
            tool_name="t",
        )
        task2 = task_manager.track_task(
            task_id="task-dup",
            server_name="s2",
            tool_name="t2",
        )
        # Should return existing, not create new
        assert task1 is task2
        assert task2.server_name == "s"

    def test_get_task(self, task_manager):
        task_manager.track_task(task_id="task-get", server_name="s", tool_name="t")
        assert task_manager.get_task("task-get") is not None
        assert task_manager.get_task("nonexistent") is None

    def test_get_active_tasks(self, task_manager):
        task_manager.track_task(task_id="task-1", server_name="s", tool_name="t")
        task_manager.track_task(task_id="task-2", server_name="s", tool_name="t")
        active = task_manager.get_active_tasks()
        assert len(active) == 2
        assert all(t.status == "working" for t in active)

    def test_get_all_tasks(self, task_manager):
        task_manager.track_task(task_id="task-a", server_name="s", tool_name="t")
        assert len(task_manager.get_all_tasks()) == 1

    def test_track_with_user_message(self, task_manager):
        task = task_manager.track_task(
            task_id="task-msg",
            server_name="s",
            tool_name="t",
            original_user_message="Find all issues",
        )
        assert task.original_user_message == "Find all issues"

    def test_track_inherits_current_user_message(self, task_manager):
        task_manager._current_user_message = "Research MCP"
        task = task_manager.track_task(
            task_id="task-inherit",
            server_name="s",
            tool_name="t",
        )
        assert task.original_user_message == "Research MCP"


class TestTaskManagerPersistence:
    def test_save_and_load(self, tmp_state_file):
        tm1 = TaskManager(state_file=tmp_state_file)
        tm1.track_task(
            task_id="task-persist",
            server_name="test-server",
            tool_name="send_message",
            agent_id="researcher",
        )
        tm1.stop()

        # Load from same file
        tm2 = TaskManager(state_file=tmp_state_file)
        task = tm2.get_task("task-persist")
        assert task is not None
        assert task.server_name == "test-server"
        assert task.agent_id == "researcher"
        tm2.stop()

    def test_clear_completed(self, task_manager):
        task_manager.track_task(task_id="task-done", server_name="s", tool_name="t")
        task_manager._tasks["task-done"].status = "completed"

        task_manager.track_task(task_id="task-active", server_name="s", tool_name="t")

        cleared = task_manager.clear_completed()
        assert cleared == 1
        assert task_manager.get_task("task-done") is None
        assert task_manager.get_task("task-active") is not None

    def test_corrupted_file(self, tmp_state_file):
        tmp_state_file.write_text("not json!!!")
        tm = TaskManager(state_file=tmp_state_file)
        assert len(tm.get_all_tasks()) == 0
        tm.stop()


class TestMCPClientRegistration:
    def test_register_mcp_client(self, task_manager):
        mock_client = MagicMock()
        task_manager.register_mcp_client("my-server", mock_client)
        assert "my-server" in task_manager._mcp_clients

    def test_multiple_clients(self, task_manager):
        task_manager.register_mcp_client("server-a", MagicMock())
        task_manager.register_mcp_client("server-b", MagicMock())
        assert len(task_manager._mcp_clients) == 2


class TestNotificationHandling:
    def test_handle_task_notification_updates_status(self, task_manager):
        task_manager.track_task(task_id="task-notif", server_name="s", tool_name="t")
        task_manager.handle_task_notification(
            server_name="s",
            task_id="task-notif",
            status="completed",
            status_message="Done!",
        )
        task = task_manager.get_task("task-notif")
        assert task.status == "completed"
        assert task.completed_at is not None

    def test_handle_unknown_task(self, task_manager):
        # Should not raise
        task_manager.handle_task_notification(
            server_name="s",
            task_id="unknown-task",
            status="completed",
        )

    def test_completion_callback_fired(self, tmp_state_file):
        completed_tasks = []

        def on_completed(task):
            completed_tasks.append(task)

        tm = TaskManager(
            state_file=tmp_state_file,
            on_task_completed=on_completed,
        )
        mock_client = MagicMock()
        mock_client._background_thread_session = None  # No session for fetch
        tm.register_mcp_client("s", mock_client)
        tm.track_task(task_id="task-cb", server_name="s", tool_name="t")

        tm.handle_task_notification("s", "task-cb", "completed", "Done")

        # Give the fetch thread a moment
        time.sleep(0.5)
        assert len(completed_tasks) == 1
        assert completed_tasks[0].task_id == "task-cb"
        tm.stop()

    def test_failure_callback_fired(self, tmp_state_file):
        failed_tasks = []

        def on_failed(task):
            failed_tasks.append(task)

        tm = TaskManager(
            state_file=tmp_state_file,
            on_task_failed=on_failed,
        )
        tm.track_task(task_id="task-fail", server_name="s", tool_name="t")
        tm.handle_task_notification("s", "task-fail", "failed", "Agent crashed")

        assert len(failed_tasks) == 1
        assert failed_tasks[0].status_message == "Agent crashed"
        tm.stop()


class TestPolling:
    def test_polling_thread_starts(self, task_manager):
        task_manager.track_task(task_id="task-poll", server_name="s", tool_name="t")
        # Polling thread should be started
        assert task_manager._poll_thread is not None
        assert task_manager._poll_thread.is_alive()

    def test_stop_polling(self, task_manager):
        task_manager.track_task(task_id="task-stop", server_name="s", tool_name="t")
        task_manager.stop()
        assert not task_manager._poll_thread.is_alive()
