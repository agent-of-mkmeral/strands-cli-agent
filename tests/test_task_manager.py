"""Tests for TaskManager — the core push-based task completion engine."""

import json
import threading
import time
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


class TestTaskTracking:
    def test_track_new_task(self, task_manager):
        task = task_manager.track_task(
            task_id="task-001",
            server_name="agent-host",
            tool_name="send_message",
            arguments={"agent_id": "researcher", "message": "find issues"},
        )
        assert task.task_id == "task-001"
        assert task.status == "working"
        assert task.server_name == "agent-host"
        assert task.tool_name == "send_message"

    def test_get_task(self, task_manager):
        task_manager.track_task("task-001", "server", "tool")
        task = task_manager.get_task("task-001")
        assert task is not None
        assert task.task_id == "task-001"

    def test_get_nonexistent_task(self, task_manager):
        assert task_manager.get_task("nonexistent") is None

    def test_get_active_tasks(self, task_manager):
        task_manager.track_task("task-001", "server", "tool")
        task_manager.track_task("task-002", "server", "tool")
        active = task_manager.get_active_tasks()
        assert len(active) == 2

    def test_get_all_tasks(self, task_manager):
        task_manager.track_task("task-001", "server", "tool")
        all_tasks = task_manager.get_all_tasks()
        assert len(all_tasks) == 1


class TestPersistence:
    def test_save_and_load(self, tmp_state_file):
        tm1 = TaskManager(state_file=tmp_state_file)
        tm1.track_task("task-001", "server", "tool", arguments={"key": "value"})
        tm1.stop()

        # Load from same file
        tm2 = TaskManager(state_file=tmp_state_file)
        task = tm2.get_task("task-001")
        assert task is not None
        assert task.task_id == "task-001"
        assert task.arguments == {"key": "value"}
        tm2.stop()

    def test_persistence_on_track(self, tmp_state_file, task_manager):
        task_manager.track_task("task-001", "server", "tool")
        assert tmp_state_file.exists()
        data = json.loads(tmp_state_file.read_text())
        assert "task-001" in data

    def test_empty_state_file(self, tmp_state_file):
        tm = TaskManager(state_file=tmp_state_file)
        assert len(tm.get_all_tasks()) == 0
        tm.stop()

    def test_corrupted_state_file(self, tmp_state_file):
        tmp_state_file.write_text("not valid json")
        tm = TaskManager(state_file=tmp_state_file)
        assert len(tm.get_all_tasks()) == 0
        tm.stop()


class TestNotificationHandling:
    def test_handle_completion_notification(self, task_manager):
        completed_tasks = []
        task_manager._on_completed = lambda t: completed_tasks.append(t)

        task_manager.track_task("task-001", "server", "tool")
        task_manager.handle_task_notification("server", "task-001", "completed", "Done!")

        task = task_manager.get_task("task-001")
        assert task.status == "completed"
        assert task.completed_at is not None

    def test_handle_failure_notification(self, task_manager):
        failed_tasks = []
        task_manager._on_failed = lambda t: failed_tasks.append(t)

        task_manager.track_task("task-001", "server", "tool")
        task_manager.handle_task_notification("server", "task-001", "failed", "Error occurred")

        task = task_manager.get_task("task-001")
        assert task.status == "failed"
        assert task.status_message == "Error occurred"
        assert len(failed_tasks) == 1

    def test_handle_unknown_task_notification(self, task_manager):
        # Should not raise
        task_manager.handle_task_notification("server", "unknown-task", "completed")

    def test_status_change_callback(self, task_manager):
        changes = []
        task_manager._on_status_changed = lambda t: changes.append(t.status)

        task_manager.track_task("task-001", "server", "tool")
        task_manager.handle_task_notification("server", "task-001", "completed")

        assert "completed" in changes


class TestClearCompleted:
    def test_clear_completed(self, task_manager):
        task_manager.track_task("task-001", "server", "tool")
        task_manager.track_task("task-002", "server", "tool")
        task_manager.handle_task_notification("server", "task-001", "completed")

        removed = task_manager.clear_completed()
        assert removed == 1
        assert len(task_manager.get_all_tasks()) == 1
        assert task_manager.get_task("task-002") is not None

    def test_clear_no_completed(self, task_manager):
        task_manager.track_task("task-001", "server", "tool")
        removed = task_manager.clear_completed()
        assert removed == 0


class TestOriginalContext:
    def test_stores_original_message(self, task_manager):
        task = task_manager.track_task(
            "task-001", "server", "tool",
            original_user_message="Research MCP issues",
        )
        assert task.original_user_message == "Research MCP issues"

    def test_original_message_persists(self, tmp_state_file):
        tm1 = TaskManager(state_file=tmp_state_file)
        tm1.track_task("task-001", "server", "tool", original_user_message="hello")
        tm1.stop()

        tm2 = TaskManager(state_file=tmp_state_file)
        task = tm2.get_task("task-001")
        assert task.original_user_message == "hello"
        tm2.stop()
