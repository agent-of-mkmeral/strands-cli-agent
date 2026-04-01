"""Main CLI entry point — interactive agent with MCP Tasks push-based completion.

This is the core loop. When a task completes, the agent is re-triggered
with the result automatically. The user gets notified through the CLI
without having to ask "is it done?".

Flow:
1. User types a message
2. Agent processes it, may call MCP tools that return Tasks
3. Agent responds (may say "dispatched, I'll let you know when done")
4. User can continue chatting or wait
5. Background: TaskManager polls/receives notifications for task completion
6. When task completes → agent is re-invoked with the result
7. Agent processes the result and displays it to the user
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from strands_cli_agent.handlers.callback_handler import callback_handler, callback_handler_instance
from strands_cli_agent.task_manager import TaskManager, TrackedTask

os.environ["STRANDS_TOOL_CONSOLE_MODE"] = "enabled"

logger = logging.getLogger(__name__)
console = Console()

# Queue for task completion events that need to re-trigger the agent
_completion_queue: queue.Queue[TrackedTask] = queue.Queue()


def _on_task_completed(task: TrackedTask) -> None:
    """Called by TaskManager when a task completes. Queues it for agent re-trigger."""
    _completion_queue.put(task)
    callback_handler_instance.on_task_completed(task.task_id, task.server_name)


def _on_task_failed(task: TrackedTask) -> None:
    """Called by TaskManager when a task fails."""
    _completion_queue.put(task)
    callback_handler_instance.on_task_failed(
        task.task_id, task.server_name, task.status_message
    )


def _on_task_status_changed(task: TrackedTask) -> None:
    """Called by TaskManager on any status change."""
    pass  # Notification handler already displays this


def _format_task_result(task: TrackedTask) -> str:
    """Format a completed task result as a message to feed back into the agent."""
    parts = [
        f"[Task Completed] The background task '{task.task_id}' from server '{task.server_name}' "
        f"(tool: {task.tool_name}) has {task.status}."
    ]

    if task.status == "completed" and task.result:
        content = task.result.get("content", [])
        if content:
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            if text_parts:
                parts.append(f"\nResult:\n{chr(10).join(text_parts)}")
        elif isinstance(task.result, dict):
            parts.append(f"\nResult: {json.dumps(task.result, indent=2)}")

    elif task.status == "failed":
        parts.append(f"\nError: {task.status_message or 'Unknown error'}")

    if task.original_user_message:
        parts.append(f"\n(Original request: {task.original_user_message})")

    return "\n".join(parts)


def _load_system_prompt() -> str:
    """Load system prompt with task-awareness instructions."""
    # Check env var first
    prompt = os.getenv("STRANDS_SYSTEM_PROMPT", "")
    if not prompt:
        prompt_file = Path.cwd() / ".prompt"
        if prompt_file.exists():
            prompt = prompt_file.read_text().strip()

    if not prompt:
        prompt = "You are a helpful assistant."

    # Add task-awareness context
    task_context = """

When you call tools on MCP servers that support tasks, those tools may run 
asynchronously in the background. The system will automatically notify you 
when tasks complete and feed the results back to you. You should:

1. When you dispatch a long-running task, let the user know it's been sent
2. When you receive a [Task Completed] message, process the results and 
   present them to the user in a clear, summarized format
3. When you receive a [Task Failed] message, inform the user and suggest 
   next steps

You have access to MCP servers that may spawn background agents. Use them
for research, code review, data analysis, etc. Results will come back 
automatically — you don't need to poll.
"""
    return prompt + task_context


def _show_welcome() -> None:
    """Display welcome message."""
    title = Text("Strands CLI Agent", style="bold cyan")
    subtitle = Text("MCP Tasks + Notifications • Push-based completion", style="dim")

    welcome = Text()
    welcome.append("Connected to MCP servers with task support.\n", style="dim")
    welcome.append("• Background tasks complete automatically\n", style="dim")
    welcome.append("• Type ", style="dim")
    welcome.append("/tasks", style="bold cyan")
    welcome.append(" to see active tasks\n", style="dim")
    welcome.append("• Type ", style="dim")
    welcome.append("/clear", style="bold cyan")
    welcome.append(" to clear completed tasks\n", style="dim")
    welcome.append("• Type ", style="dim")
    welcome.append("exit", style="bold cyan")
    welcome.append(" to quit\n", style="dim")

    panel = Panel(
        welcome,
        title=title,
        subtitle=subtitle,
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)


def _show_tasks(task_manager: TaskManager) -> None:
    """Display task status table."""
    tasks = task_manager.get_all_tasks()
    if not tasks:
        console.print("  [dim]No tracked tasks[/dim]")
        return

    console.print(f"\n  [bold]Tracked Tasks[/bold] ({len(tasks)} total)\n")

    status_style = {
        "working": "yellow",
        "completed": "green",
        "failed": "red",
        "cancelled": "dim",
    }

    for task in sorted(tasks, key=lambda t: t.created_at, reverse=True):
        style = status_style.get(task.status, "white")
        emoji = {"working": "⏳", "completed": "✅", "failed": "❌", "cancelled": "🚫"}.get(task.status, "📋")
        console.print(
            f"  {emoji} [{style}]{task.status:10s}[/{style}] "
            f"[cyan]{task.task_id[:12]}...[/cyan] "
            f"[dim]{task.server_name}/{task.tool_name}[/dim]"
        )
        if task.status_message:
            console.print(f"     [dim]{task.status_message[:80]}[/dim]")
    console.print()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Strands CLI Agent — MCP Tasks & Notifications powered",
    )
    parser.add_argument("query", nargs="*", help="Query to process (non-interactive mode)")
    parser.add_argument("--mcp-config", help="Path to MCP config file (mcp.json)")
    parser.add_argument(
        "--model-provider", default="bedrock",
        help="Model provider (default: bedrock)",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Model ID to use",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--no-tasks", action="store_true",
        help="Disable task tracking and push notifications",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # Initialize TaskManager
    task_manager = TaskManager(
        on_task_completed=_on_task_completed,
        on_task_failed=_on_task_failed,
        on_task_status_changed=_on_task_status_changed,
    ) if not args.no_tasks else None

    # Load MCP servers
    from strands_cli_agent.mcp_loader import create_mcp_clients, load_mcp_config

    mcp_config = load_mcp_config(args.mcp_config)
    mcp_clients = create_mcp_clients(mcp_config)

    # Build tools list
    tools: list = []

    # Add strands-tools (shell, editor, etc.) if available
    try:
        from strands_tools import editor, file_read, file_write, shell, think

        tools.extend([shell, editor, file_read, file_write, think])
    except ImportError:
        logger.info("strands-tools not available")

    # Add MCP clients
    tools.extend(mcp_clients)

    # Add custom tools from ./tools directory
    tools_dir = Path.cwd() / "tools"
    load_from_dir = tools_dir.exists()

    # Create model
    from strands_tools.utils.models.model import create_model

    model_kwargs = {}
    if args.model_id:
        os.environ["STRANDS_MODEL_ID"] = args.model_id

    try:
        model = create_model(provider=args.model_provider)
    except Exception:
        # Fallback to default bedrock
        from strands.models.bedrock import BedrockModel

        model = BedrockModel()

    # Create agent
    from strands import Agent

    system_prompt = _load_system_prompt()

    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        callback_handler=callback_handler,
        load_tools_from_directory=load_from_dir,
    )

    # Non-interactive mode
    if args.query:
        query = " ".join(args.query)
        agent(query)

        # If tasks were created, wait for them
        if task_manager and task_manager.get_active_tasks():
            console.print("\n[dim]Waiting for background tasks to complete...[/dim]")
            import time

            while task_manager.get_active_tasks():
                # Process any completed tasks
                while not _completion_queue.empty():
                    completed = _completion_queue.get_nowait()
                    result_msg = _format_task_result(completed)
                    agent(result_msg)
                time.sleep(1)

        if task_manager:
            task_manager.stop()
        return

    # Interactive mode
    _show_welcome()

    from strands_tools.utils.user_input import get_user_input

    while True:
        try:
            # Check for completed tasks before prompting
            while not _completion_queue.empty():
                completed = _completion_queue.get_nowait()
                result_msg = _format_task_result(completed)
                console.print(f"\n[bold cyan]🔔 Processing completed task result...[/bold cyan]")
                agent(result_msg)

            user_input = get_user_input("\n~ ", default="", keyboard_interrupt_return_default=False)

            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            # Slash commands
            if user_input == "/tasks":
                if task_manager:
                    _show_tasks(task_manager)
                else:
                    console.print("  [dim]Task tracking disabled[/dim]")
                continue

            if user_input == "/clear":
                if task_manager:
                    n = task_manager.clear_completed()
                    console.print(f"  [dim]Cleared {n} completed tasks[/dim]")
                continue

            if user_input.startswith("!"):
                # Shell command shortcut
                shell_command = user_input[1:]
                print(f"$ {shell_command}")
                try:
                    agent.tool.shell(
                        command=shell_command,
                        user_message_override=user_input,
                        non_interactive_mode=True,
                    )
                    print()
                except Exception as e:
                    print(f"Shell error: {e}")
                continue

            if user_input.strip():
                # Store the current user message for task context
                if task_manager:
                    task_manager._current_user_message = user_input

                agent(user_input)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            callback_handler(force_stop=True)
            console.print(f"\n[red]Error: {e}[/red]")

    # Cleanup
    if task_manager:
        task_manager.stop()


if __name__ == "__main__":
    main()
