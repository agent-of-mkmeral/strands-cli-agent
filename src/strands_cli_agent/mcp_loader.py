"""MCP Server loader with task-aware notification handling.

Loads MCP servers from config (same format as Kiro/Claude Desktop mcp.json),
but wraps them with:
1. Task-aware execution via Strands MCPClient TasksConfig
2. Notification callbacks that feed into the TaskManager
3. Logging callbacks that display real-time server messages in the CLI
4. Environment variable pass-through (${VAR_NAME} syntax in config values)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Pattern to match ${VAR_NAME} references in config values
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def resolve_env_vars(value: str) -> str:
    """Resolve ${VAR_NAME} references in a string value.

    Replaces all occurrences of ${VAR_NAME} with the corresponding
    environment variable value. If the variable is not set, leaves
    the reference as-is and logs a warning.

    Supports:
    - Full replacement: "${MY_TOKEN}" → "actual_token_value"
    - Partial replacement: "Bearer ${MY_TOKEN}" → "Bearer actual_token_value"
    - Multiple references: "${HOST}:${PORT}" → "localhost:8080"
    - Nested-safe: only matches ${...} (not $VAR or {{VAR}})

    Args:
        value: String that may contain ${VAR_NAME} references

    Returns:
        String with env var references resolved
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            logger.warning(f"Environment variable '{var_name}' not set (referenced as ${{{var_name}}})")
            return match.group(0)  # Leave as-is
        return env_value

    return _ENV_VAR_PATTERN.sub(_replace, value)


def resolve_env_vars_in_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ${VAR_NAME} references in dict values.

    Processes string values, lists, and nested dicts. Non-string
    values (int, bool, None) are left unchanged.

    Args:
        d: Dictionary with potentially unresolved env var references

    Returns:
        New dictionary with all string values resolved
    """
    resolved = {}
    for key, value in d.items():
        if isinstance(value, str):
            resolved[key] = resolve_env_vars(value)
        elif isinstance(value, dict):
            resolved[key] = resolve_env_vars_in_dict(value)
        elif isinstance(value, list):
            resolved[key] = [
                resolve_env_vars(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            resolved[key] = value
    return resolved


def load_mcp_config(config_path: str | None = None) -> dict[str, Any]:
    """Load MCP server configuration.

    Priority:
    1. Explicit config_path argument
    2. STRANDS_MCP_CONFIG env var (path to file)
    3. MCP_SERVERS env var (inline JSON)
    4. ~/.strands-cli/mcp.json
    5. ~/.kiro/settings/mcp.json (Kiro config)

    Returns:
        Dict with mcpServers config
    """
    # 1. Explicit path
    if config_path:
        path = Path(config_path).expanduser()
        if path.exists():
            logger.info(f"Loading MCP config from {path}")
            return json.loads(path.read_text())

    # 2. STRANDS_MCP_CONFIG env var
    env_config_path = os.getenv("STRANDS_MCP_CONFIG")
    if env_config_path:
        path = Path(env_config_path).expanduser()
        if path.exists():
            logger.info(f"Loading MCP config from STRANDS_MCP_CONFIG={path}")
            return json.loads(path.read_text())

    # 3. MCP_SERVERS env var (inline JSON, same as agent-builder)
    mcp_json = os.getenv("MCP_SERVERS")
    if mcp_json:
        logger.info("Loading MCP config from MCP_SERVERS env var")
        return json.loads(mcp_json)

    # 4. Default locations
    default_paths = [
        Path.home() / ".strands-cli" / "mcp.json",
        Path.cwd() / "mcp.json",
        Path.home() / ".kiro" / "settings" / "mcp.json",
    ]
    for path in default_paths:
        if path.exists():
            logger.info(f"Loading MCP config from {path}")
            return json.loads(path.read_text())

    logger.info("No MCP config found")
    return {}


def create_mcp_clients(
    config: dict[str, Any],
    logging_callback: Callable | None = None,
    message_handler: Callable | None = None,
) -> list[Any]:
    """Create MCPClient instances from config with task support enabled.

    Environment variable references (${VAR_NAME}) in env, args, command,
    url, and headers values are automatically resolved from the process
    environment. This allows sharing config files without exposing tokens:

        {
          "env": {
            "GITHUB_TOKEN": "${MY_GITHUB_TOKEN}",
            "API_KEY": "${MY_API_KEY}"
          }
        }

    Args:
        config: MCP config dict (with mcpServers key)
        logging_callback: Callback for MCP log messages (notifications/message)
        message_handler: Callback for all MCP messages (including task notifications)

    Returns:
        List of MCPClient instances ready to be used as agent tools
    """
    from datetime import timedelta

    from mcp import StdioServerParameters, stdio_client
    from strands.tools.mcp import MCPClient, TasksConfig

    servers = config.get("mcpServers", {})
    clients = []

    for name, cfg in servers.items():
        try:
            # Skip disabled servers
            if cfg.get("disabled", False):
                logger.info(f"⏭  MCP server '{name}' (disabled)")
                continue

            # Build tool filters for disabled tools
            tool_filters = None
            disabled_tools = cfg.get("disabledTools", [])
            if disabled_tools:
                from strands.tools.mcp import ToolFilters

                tool_filters = ToolFilters(rejected=disabled_tools)

            # Build transport callable
            transport = None
            if "command" in cfg:
                command = resolve_env_vars(cfg["command"])
                args = [resolve_env_vars(a) if isinstance(a, str) else a for a in cfg.get("args", [])]

                # Resolve env var references in the env dict
                env = None
                if cfg.get("env"):
                    env = resolve_env_vars_in_dict(cfg["env"])

                def make_transport(_cmd=command, _args=args, _env=env):
                    return stdio_client(StdioServerParameters(command=_cmd, args=_args, env=_env))

                transport = make_transport

            elif "url" in cfg:
                url = resolve_env_vars(cfg["url"])

                # Resolve env var references in headers
                headers = None
                if cfg.get("headers"):
                    headers = resolve_env_vars_in_dict(cfg["headers"])

                def make_http_transport(_url=url, _headers=headers):
                    if "/sse" in _url:
                        from mcp.client.sse import sse_client

                        return sse_client(_url)
                    else:
                        from mcp.client.streamable_http import streamablehttp_client

                        return streamablehttp_client(url=_url, headers=_headers)

                transport = make_http_transport
            else:
                logger.warning(f"MCP server '{name}': no command or url, skipping")
                continue

            # Enable task-augmented execution
            # This tells Strands MCPClient to use call_tool_as_task for supported tools
            tasks_config = TasksConfig(
                ttl=timedelta(hours=1),
                poll_timeout=timedelta(hours=1),
            )

            client = MCPClient(
                transport_callable=transport,
                prefix=cfg.get("prefix", name),
                tool_filters=tool_filters,
                tasks_config=tasks_config,
            )

            clients.append(client)

            if disabled_tools:
                logger.info(f"✓ MCP server '{name}' (tasks enabled, disabled: {', '.join(disabled_tools)})")
            else:
                logger.info(f"✓ MCP server '{name}' (tasks enabled)")

        except Exception as e:
            logger.error(f"✗ MCP server '{name}': {e}")

    return clients
