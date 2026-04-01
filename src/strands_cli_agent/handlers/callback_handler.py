"""CLI callback handler with task-awareness.

Based on agent-builder's callback handler but enhanced with:
- Task status indicators (shows when tasks are dispatched/completed)
- Notification display (real-time MCP server logs)
- Desktop notifications for task completion (optional)
"""

import os
import time
from typing import Any

from colorama import Fore, Style, init
from halo import Halo

init(autoreset=True)

SPINNERS = {
    "dots": {
        "interval": 80,
        "frames": ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"],
    }
}

TOOL_COLORS = {
    "running": Fore.GREEN,
    "success": Fore.GREEN,
    "error": Fore.RED,
    "info": Fore.CYAN,
}


class ToolSpinner:
    def __init__(self, text: str = "", color: str = TOOL_COLORS["running"]):
        self.spinner = Halo(
            text=text, spinner=SPINNERS["dots"], color="green",
            text_color="green", interval=80,
        )
        self.color = color
        self.current_text = text

    def start(self, text: str | None = None):
        if text:
            self.current_text = text
        print()
        self.spinner.start(f"{self.color}{self.current_text}{Style.RESET_ALL}")

    def update(self, text: str):
        self.current_text = text
        self.spinner.text = f"{self.color}{text}{Style.RESET_ALL}"

    def succeed(self, text: str | None = None):
        if text:
            self.current_text = text
        self.spinner.succeed(f"{TOOL_COLORS['success']}{self.current_text}{Style.RESET_ALL}")

    def fail(self, text: str | None = None):
        if text:
            self.current_text = text
        self.spinner.fail(f"{TOOL_COLORS['error']}{self.current_text}{Style.RESET_ALL}")

    def info(self, text: str | None = None):
        if text:
            self.current_text = text
        self.spinner.info(f"{TOOL_COLORS['info']}{self.current_text}{Style.RESET_ALL}")

    def stop(self):
        self.spinner.stop()


class CallbackHandler:
    def __init__(self):
        self.thinking_spinner = None
        self.current_spinner: ToolSpinner | None = None
        self.current_tool = None
        self.tool_histories: dict[str, dict] = {}
        # Task tracking for display
        self._active_tasks: dict[str, str] = {}  # task_id -> tool_name

    def callback_handler(self, **kwargs: Any) -> None:
        reasoningText = kwargs.get("reasoningText", False)
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        force_stop = kwargs.get("force_stop", False)
        message = kwargs.get("message", {})
        current_tool_use = kwargs.get("current_tool_use", {})
        init_event_loop = kwargs.get("init_event_loop", False)
        start_event_loop = kwargs.get("start_event_loop", False)
        event_loop_throttled_delay = kwargs.get("event_loop_throttled_delay", None)
        console = kwargs.get("console", None)

        try:
            if self.thinking_spinner and (data or current_tool_use):
                self.thinking_spinner.stop()

            if init_event_loop:
                from rich.status import Status
                self.thinking_spinner = Status(
                    "[blue] thinking...[/blue]", spinner="dots", console=console,
                )
                self.thinking_spinner.start()

            if reasoningText:
                print(reasoningText, end="")

            if start_event_loop:
                self.thinking_spinner.update("[blue] thinking...[/blue]")
        except BaseException:
            pass

        if event_loop_throttled_delay and console:
            if self.current_spinner:
                self.current_spinner.stop()
            console.print(
                f"[red]Throttled! Waiting [bold]{event_loop_throttled_delay}s[/bold]...[/red]"
            )

        if force_stop:
            if self.thinking_spinner:
                self.thinking_spinner.stop()
            if self.current_spinner:
                self.current_spinner.stop()

        if data:
            if complete:
                print(f"{Fore.WHITE}{data}{Style.RESET_ALL}")
            else:
                print(f"{Fore.WHITE}{data}{Style.RESET_ALL}", end="")

        if current_tool_use and current_tool_use.get("input"):
            tool_id = current_tool_use.get("toolUseId")
            tool_name = current_tool_use.get("name")
            tool_input = current_tool_use.get("input", "")

            if tool_id != self.current_tool:
                if self.current_spinner:
                    self.current_spinner.stop()
                self.current_tool = tool_id
                self.current_spinner = ToolSpinner(f"🛠️  {tool_name}: Preparing...", TOOL_COLORS["running"])
                self.current_spinner.start()
                self.tool_histories[tool_id] = {
                    "name": tool_name, "start_time": time.time(), "input_size": 0,
                }

            if tool_id in self.tool_histories:
                current_size = len(tool_input)
                if current_size > self.tool_histories[tool_id]["input_size"]:
                    self.tool_histories[tool_id]["input_size"] = current_size
                    if self.current_spinner:
                        self.current_spinner.update(f"🛠️  {tool_name}: {current_size} chars")

        if isinstance(message, dict):
            if message.get("role") == "assistant":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_use = content.get("toolUse")
                        if tool_use:
                            tool_name = tool_use.get("name")
                            if self.current_spinner:
                                self.current_spinner.info(f"🔧 {tool_name}")

            elif message.get("role") == "user":
                for content in message.get("content", []):
                    if isinstance(content, dict):
                        tool_result = content.get("toolResult")
                        if tool_result:
                            tool_id = tool_result.get("toolUseId")
                            status = tool_result.get("status")
                            if tool_id in self.tool_histories:
                                tool_info = self.tool_histories[tool_id]
                                duration = round(time.time() - tool_info["start_time"], 2)

                                msg = f"{tool_info['name']} completed in {duration}s"
                                if status != "success":
                                    msg = f"{tool_info['name']} failed after {duration}s"

                                if self.current_spinner:
                                    if status == "success":
                                        self.current_spinner.succeed(msg)
                                    else:
                                        self.current_spinner.fail(msg)

                                del self.tool_histories[tool_id]
                                self.current_spinner = None
                                self.current_tool = None

    def on_task_dispatched(self, task_id: str, tool_name: str, agent_id: str) -> None:
        """Called when a task is dispatched to an MCP server."""
        self._active_tasks[task_id] = tool_name
        print(f"\n{Fore.CYAN}📋 Task dispatched → {agent_id} (task: {task_id[:12]}...){Style.RESET_ALL}")

    def on_task_completed(self, task_id: str, agent_id: str) -> None:
        """Called when a tracked task completes."""
        tool_name = self._active_tasks.pop(task_id, "unknown")
        print(f"\n{Fore.GREEN}✅ Task completed ← {agent_id} (task: {task_id[:12]}...){Style.RESET_ALL}")

        # Desktop notification (macOS)
        if os.environ.get("STRANDS_CLI_NOTIFICATIONS", "true").lower() == "true":
            try:
                os.system(f'osascript -e \'display notification "Task from {agent_id} completed" with title "Strands CLI"\'')
            except Exception:
                pass

    def on_task_failed(self, task_id: str, agent_id: str, error: str | None = None) -> None:
        """Called when a tracked task fails."""
        self._active_tasks.pop(task_id, None)
        print(f"\n{Fore.RED}❌ Task failed ← {agent_id}: {error or 'unknown error'}{Style.RESET_ALL}")


callback_handler_instance = CallbackHandler()
callback_handler = callback_handler_instance.callback_handler
