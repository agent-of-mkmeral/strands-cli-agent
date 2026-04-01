"""Tests for CallbackHandler task detection in tool results."""

import json
from unittest.mock import MagicMock

import pytest

from strands_cli_agent.handlers.callback_handler import CallbackHandler


@pytest.fixture
def handler():
    return CallbackHandler()


@pytest.fixture
def task_manager():
    tm = MagicMock()
    tm.get_task.return_value = None  # No task tracked yet
    tm.track_task.return_value = MagicMock()
    return tm


@pytest.fixture
def wired_handler(handler, task_manager):
    handler.set_task_manager(
        task_manager,
        mcp_prefixes={"containerized-strands-agents", "my-server"},
    )
    return handler


class TestServerNameExtraction:
    def test_known_prefix(self, wired_handler):
        assert wired_handler._get_server_name("containerized-strands-agents_send_message") == "containerized-strands-agents"

    def test_known_prefix_different_tool(self, wired_handler):
        assert wired_handler._get_server_name("my-server_list_agents") == "my-server"

    def test_unknown_prefix(self, wired_handler):
        result = wired_handler._get_server_name("unknown-tool")
        assert result == "unknown-tool"


class TestTaskDetection:
    def test_detect_task_in_result(self, wired_handler, task_manager):
        """Task is detected and tracked when tool result contains taskId."""
        wired_handler._pending_tool_names["tool-use-1"] = "containerized-strands-agents_send_message"
        wired_handler.set_current_user_message("Research MCP issues")

        tool_result = {
            "content": [
                {
                    "text": json.dumps({
                        "status": "dispatched",
                        "agent_id": "researcher",
                        "taskId": "task-abc123",
                        "taskStatus": "working",
                        "message": "Task created.",
                    })
                }
            ]
        }

        wired_handler._detect_task_in_result(tool_result, "tool-use-1")

        task_manager.track_task.assert_called_once_with(
            task_id="task-abc123",
            server_name="containerized-strands-agents",
            tool_name="send_message",
            agent_id="researcher",
            poll_interval_ms=5000,
            original_user_message="Research MCP issues",
        )

    def test_no_task_without_task_id(self, wired_handler, task_manager):
        """Tool results without taskId are ignored."""
        wired_handler._pending_tool_names["tool-use-2"] = "my-server_list_agents"

        tool_result = {
            "content": [
                {"text": json.dumps({"status": "success", "agents": []})}
            ]
        }

        wired_handler._detect_task_in_result(tool_result, "tool-use-2")
        task_manager.track_task.assert_not_called()

    def test_no_task_for_error_status(self, wired_handler, task_manager):
        """Tool results with status=error are not tracked."""
        wired_handler._pending_tool_names["tool-use-3"] = "containerized-strands-agents_send_message"

        tool_result = {
            "content": [
                {"text": json.dumps({"status": "error", "taskId": "task-err", "error": "Failed"})}
            ]
        }

        wired_handler._detect_task_in_result(tool_result, "tool-use-3")
        task_manager.track_task.assert_not_called()

    def test_no_duplicate_tracking(self, wired_handler, task_manager):
        """Already-tracked tasks are not re-tracked."""
        task_manager.get_task.return_value = MagicMock()  # Task exists
        wired_handler._pending_tool_names["tool-use-4"] = "containerized-strands-agents_send_message"

        tool_result = {
            "content": [
                {"text": json.dumps({"status": "dispatched", "taskId": "task-existing", "agent_id": "coder"})}
            ]
        }

        wired_handler._detect_task_in_result(tool_result, "tool-use-4")
        task_manager.track_task.assert_not_called()

    def test_non_json_content_ignored(self, wired_handler, task_manager):
        """Non-JSON text content is safely ignored."""
        wired_handler._pending_tool_names["tool-use-5"] = "containerized-strands-agents_send_message"

        tool_result = {
            "content": [
                {"text": "This is just plain text, not JSON"},
                {"text": "123"},
            ]
        }

        wired_handler._detect_task_in_result(tool_result, "tool-use-5")
        task_manager.track_task.assert_not_called()

    def test_no_task_manager_no_crash(self, handler):
        """Without TaskManager wired, detection is a no-op."""
        tool_result = {
            "content": [
                {"text": json.dumps({"taskId": "task-orphan", "status": "dispatched"})}
            ]
        }
        # Should not raise
        handler._detect_task_in_result(tool_result, "tool-use-6")


class TestCallbackIntegration:
    def test_tool_name_tracked_from_assistant_message(self, wired_handler):
        """Tool names are captured from assistant toolUse messages."""
        message = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tu-123",
                        "name": "containerized-strands-agents_send_message",
                        "input": {"agent_id": "researcher", "message": "hello"},
                    }
                }
            ],
        }
        wired_handler.callback_handler(message=message)
        assert "tu-123" in wired_handler._pending_tool_names
        assert wired_handler._pending_tool_names["tu-123"] == "containerized-strands-agents_send_message"

    def test_task_detected_in_user_tool_result(self, wired_handler, task_manager):
        """Full flow: assistant calls tool → user gets result → task detected."""
        # Step 1: Assistant sends tool call
        assistant_msg = {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "tu-flow",
                        "name": "containerized-strands-agents_send_message",
                        "input": {"agent_id": "researcher", "message": "find issues"},
                    }
                }
            ],
        }
        wired_handler.callback_handler(message=assistant_msg)

        # Step 2: User gets tool result with taskId
        user_msg = {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tu-flow",
                        "status": "success",
                        "content": [
                            {
                                "text": json.dumps({
                                    "status": "dispatched",
                                    "agent_id": "researcher",
                                    "taskId": "task-flow-123",
                                    "taskStatus": "working",
                                })
                            }
                        ],
                    }
                }
            ],
        }
        wired_handler.callback_handler(message=user_msg)

        # Should have been tracked
        task_manager.track_task.assert_called_once()
        call_kwargs = task_manager.track_task.call_args[1]
        assert call_kwargs["task_id"] == "task-flow-123"
        assert call_kwargs["server_name"] == "containerized-strands-agents"
        assert call_kwargs["tool_name"] == "send_message"
        assert call_kwargs["agent_id"] == "researcher"

    def test_poll_interval_from_response(self, wired_handler, task_manager):
        """Custom pollInterval from server response is used."""
        wired_handler._pending_tool_names["tu-pi"] = "containerized-strands-agents_send_message"

        tool_result = {
            "content": [
                {"text": json.dumps({
                    "status": "dispatched",
                    "taskId": "task-pi",
                    "agent_id": "coder",
                    "pollInterval": 10000,
                })}
            ]
        }

        wired_handler._detect_task_in_result(tool_result, "tu-pi")
        call_kwargs = task_manager.track_task.call_args[1]
        assert call_kwargs["poll_interval_ms"] == 10000
