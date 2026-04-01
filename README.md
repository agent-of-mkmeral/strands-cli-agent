# Strands CLI Agent

A CLI-based agent with **MCP Tasks and Notifications** support. When MCP servers complete background tasks, the agent is automatically re-triggered with results — push, not pull.

## Why This Exists

Most MCP clients (Kiro, Claude Desktop) don't support MCP Tasks yet. This CLI agent demonstrates the **client side** of MCP Tasks:

1. **Task-Aware Execution** — When you call a tool that returns an MCP Task, the agent tracks it automatically
2. **Push-Based Completion** — When the task completes, the agent is re-triggered with the result. No polling by the user.
3. **Notifications** — Real-time server logs and status updates displayed in the CLI
4. **Re-trigger Loop** — Task completion → result fetched → fed back into agent → agent processes and responds

## How It Works

```
User: "Research all open MCP issues"
  │
  ▼
Agent calls send_message("researcher", "find all open MCP issues")
  │
  ▼
MCP Server returns: Task { taskId: "task-123", status: "working" }
  │
  ▼
Agent: "I've dispatched a researcher agent. I'll let you know when it's done."
  │
  ▼
... user can keep chatting or wait ...
  │
  ▼
[Background] TaskManager detects task completed via polling/notification
  │
  ▼
Agent is re-invoked: "[Task Completed] researcher found 12 issues..."
  │
  ▼
Agent: "The researcher found 12 open MCP issues. Here's the summary: ..."
```

## Quick Start

```bash
# Install
pip install -e .

# Configure MCP servers (same format as Kiro/Claude Desktop)
cp mcp.json.example ~/.strands-cli/mcp.json
# Edit with your MCP server configs

# Run
strands-cli
```

## MCP Config

Uses the same format as Kiro/Claude Desktop. Place in any of:
- `~/.strands-cli/mcp.json`
- `./mcp.json` (current directory)
- `~/.kiro/settings/mcp.json` (Kiro config)
- `STRANDS_MCP_CONFIG` env var (path)
- `MCP_SERVERS` env var (inline JSON)

### Environment Variable Pass-Through

Use `${VAR_NAME}` syntax in config values to reference environment variables. This way you can share your config file without exposing tokens:

```json
{
  "mcpServers": {
    "containerized-strands-agents": {
      "command": "containerized-strands-agents-server",
      "env": {
        "CONTAINERIZED_AGENTS_GITHUB_TOKEN": "${STRANDS_CODER_TOKEN}",
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
      }
    },
    "github": {
      "command": "uvx",
      "args": ["mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "remote-api": {
      "url": "https://${API_HOST}/mcp",
      "headers": {
        "Authorization": "Bearer ${API_TOKEN}"
      }
    }
  }
}
```

Env var resolution works in:
- `env` values — `"${MY_TOKEN}"` → actual token value
- `args` — `["--token", "${MY_TOKEN}"]`
- `command` — `"${BINARY_PATH}"`
- `url` — `"https://${HOST}/mcp"`
- `headers` — `"Bearer ${TOKEN}"`

If an env var is not set, the `${VAR_NAME}` reference is left as-is and a warning is logged.

## Usage

### Interactive Mode
```bash
strands-cli
```

Commands:
- `/tasks` — Show all tracked tasks and their status
- `/clear` — Clear completed/failed tasks
- `!<command>` — Run a shell command
- `exit` — Quit

### One-Shot Mode
```bash
strands-cli "Research the top MCP servers and summarize"
```

In one-shot mode, if tasks are created, the CLI waits for them to complete before exiting.

### Options
```bash
strands-cli --mcp-config ./my-mcp.json     # Custom MCP config
strands-cli --model-provider bedrock         # Model provider
strands-cli --model-id us.anthropic.claude-sonnet-4-5-20250929-v1:0  # Model
strands-cli -v                                # Verbose logging
strands-cli --no-tasks                        # Disable task tracking
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                CLI (cli.py)                  │
│  ┌─────────┐  ┌──────────┐  ┌───────────┐  │
│  │  Agent   │  │ Callback │  │  Welcome   │  │
│  │ (Strands)│  │ Handler  │  │  /Tasks UI │  │
│  └────┬─────┘  └──────────┘  └───────────┘  │
│       │                                      │
│  ┌────┼────────────────────────────────────┐ │
│  │         MCP Clients (mcp_loader.py)     │ │
│  │  ┌──────────┐  ┌──────────┐             │ │
│  │  │ Server A │  │ Server B │  ...        │ │
│  │  │(tasks ✓) │  │(tasks ✓) │             │ │
│  │  └────┬─────┘  └────┬─────┘             │ │
│  └───────┼──────────────┼──────────────────┘ │
│          │              │                    │
│  ┌───────┼──────────────┼──────────────────┐ │
│  │       TaskManager (task_manager.py)     │ │
│  │                                         │ │
│  │  • Tracks tasks from all servers        │ │
│  │  • Polls via MCP tasks/get protocol     │ │
│  │  • Receives notifications/tasks/status  │ │
│  │  • On completion → re-triggers agent    │ │
│  │  • Persists to ~/.strands-cli/tasks.json│ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │    NotificationHandler                  │ │
│  │  • notifications/tasks/status → push    │ │
│  │  • notifications/message → CLI logs     │ │
│  │  • notifications/progress → progress bar│ │
│  └─────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

## MCP Features Used (Client Side)

| MCP Feature | How We Use It |
|-------------|--------------|
| **Tools** | Call tools on connected MCP servers |
| **Tasks** | Track long-running tool calls via `tasks/get`, `tasks/result`, `tasks/cancel` |
| **Task Notifications** | Receive `notifications/tasks/status` for push-based updates |
| **Logging Notifications** | Display real-time server logs in the CLI |
| **Progress Notifications** | Show progress indicators for long operations |
| **Task-Augmented Execution** | Use `call_tool_as_task` for servers that support it |

## Pairing with containerized-strands-agents

This CLI is designed to work with [containerized-strands-agents](https://github.com/mkmeral/containerized-strands-agents) as the MCP server:

```json
{
  "mcpServers": {
    "agents": {
      "command": "containerized-strands-agents-server",
      "env": {
        "CONTAINERIZED_AGENTS_GITHUB_TOKEN": "${STRANDS_CODER_TOKEN}",
        "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}"
      }
    }
  }
}
```

Then:
```
~ send a message to researcher to find all open MCP issues
📋 Task dispatched → agents (task: task-abc123...)
I've dispatched a researcher agent to find open MCP issues. I'll let you know when results come back.

... time passes ...

✅ Task completed ← agents (task: task-abc123...)
🔔 Processing completed task result...
The researcher found 12 open issues related to MCP. Here's the breakdown:
1. ...
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_MCP_CONFIG` | Path to MCP config file | `~/.strands-cli/mcp.json` |
| `MCP_SERVERS` | Inline MCP config JSON | - |
| `STRANDS_SYSTEM_PROMPT` | Custom system prompt | - |
| `STRANDS_MODEL_ID` | Model identifier | `us.anthropic.claude-sonnet-4-5-20250929-v1:0` |
| `STRANDS_CLI_NOTIFICATIONS` | Enable desktop notifications | `true` |

## License

Apache-2.0
