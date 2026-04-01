"""Tests for MCP config loading with environment variable pass-through."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from strands_cli_agent.mcp_loader import (
    load_mcp_config,
    resolve_env_vars,
    resolve_env_vars_in_dict,
)


class TestResolveEnvVars:
    """Tests for ${VAR_NAME} resolution in string values."""

    def test_simple_replacement(self):
        with patch.dict(os.environ, {"MY_TOKEN": "secret123"}):
            assert resolve_env_vars("${MY_TOKEN}") == "secret123"

    def test_partial_replacement(self):
        with patch.dict(os.environ, {"MY_TOKEN": "secret123"}):
            assert resolve_env_vars("Bearer ${MY_TOKEN}") == "Bearer secret123"

    def test_multiple_references(self):
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            assert resolve_env_vars("${HOST}:${PORT}") == "localhost:8080"

    def test_unset_variable_left_as_is(self):
        # Make sure the var is NOT set
        env = os.environ.copy()
        env.pop("NONEXISTENT_VAR_12345", None)
        with patch.dict(os.environ, env, clear=True):
            result = resolve_env_vars("${NONEXISTENT_VAR_12345}")
            assert result == "${NONEXISTENT_VAR_12345}"

    def test_mixed_set_and_unset(self):
        env = os.environ.copy()
        env["SET_VAR"] = "hello"
        env.pop("UNSET_VAR_99999", None)
        with patch.dict(os.environ, env, clear=True):
            result = resolve_env_vars("${SET_VAR}-${UNSET_VAR_99999}")
            assert result == "hello-${UNSET_VAR_99999}"

    def test_no_reference_returns_unchanged(self):
        assert resolve_env_vars("plain string") == "plain string"

    def test_empty_string(self):
        assert resolve_env_vars("") == ""

    def test_non_string_passthrough(self):
        assert resolve_env_vars(42) == 42
        assert resolve_env_vars(None) is None
        assert resolve_env_vars(True) is True

    def test_dollar_without_braces_ignored(self):
        assert resolve_env_vars("$MY_TOKEN") == "$MY_TOKEN"

    def test_double_braces_ignored(self):
        assert resolve_env_vars("{{MY_TOKEN}}") == "{{MY_TOKEN}}"

    def test_empty_var_name(self):
        # ${} should not match (regex requires at least one char)
        assert resolve_env_vars("${}") == "${}"

    def test_var_with_underscores_and_numbers(self):
        with patch.dict(os.environ, {"MY_API_KEY_V2": "key123"}):
            assert resolve_env_vars("${MY_API_KEY_V2}") == "key123"

    def test_empty_env_var_value(self):
        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            assert resolve_env_vars("${EMPTY_VAR}") == ""

    def test_env_var_with_special_chars_in_value(self):
        with patch.dict(os.environ, {"TOKEN": "abc+def/ghi="}):
            assert resolve_env_vars("${TOKEN}") == "abc+def/ghi="


class TestResolveEnvVarsInDict:
    """Tests for recursive dict resolution."""

    def test_flat_dict(self):
        with patch.dict(os.environ, {"TOKEN": "secret"}):
            result = resolve_env_vars_in_dict({"key": "${TOKEN}"})
            assert result == {"key": "secret"}

    def test_nested_dict(self):
        with patch.dict(os.environ, {"TOKEN": "secret"}):
            result = resolve_env_vars_in_dict({
                "outer": {
                    "inner": "${TOKEN}"
                }
            })
            assert result == {"outer": {"inner": "secret"}}

    def test_list_values(self):
        with patch.dict(os.environ, {"ARG": "value"}):
            result = resolve_env_vars_in_dict({"args": ["--token", "${ARG}"]})
            assert result == {"args": ["--token", "value"]}

    def test_non_string_values_unchanged(self):
        result = resolve_env_vars_in_dict({
            "port": 8080,
            "disabled": False,
            "timeout": None,
        })
        assert result == {"port": 8080, "disabled": False, "timeout": None}

    def test_mixed_types(self):
        with patch.dict(os.environ, {"HOST": "localhost"}):
            result = resolve_env_vars_in_dict({
                "url": "${HOST}",
                "port": 443,
                "tags": ["${HOST}", "prod"],
                "nested": {"host": "${HOST}"},
            })
            assert result == {
                "url": "localhost",
                "port": 443,
                "tags": ["localhost", "prod"],
                "nested": {"host": "localhost"},
            }

    def test_does_not_mutate_original(self):
        original = {"key": "${TOKEN}"}
        with patch.dict(os.environ, {"TOKEN": "secret"}):
            result = resolve_env_vars_in_dict(original)
        assert original == {"key": "${TOKEN}"}
        assert result == {"key": "secret"}

    def test_real_world_mcp_env_config(self):
        """Test the exact pattern from the user's request."""
        with patch.dict(os.environ, {
            "STRANDS_CODER_TOKEN": "ghp_realtoken123",
            "PERPLEXITY_API_KEY": "pplx-realkey456",
        }):
            config_env = {
                "CONTAINERIZED_AGENTS_GITHUB_TOKEN": "${STRANDS_CODER_TOKEN}",
                "PERPLEXITY_API_KEY": "${PERPLEXITY_API_KEY}",
            }
            result = resolve_env_vars_in_dict(config_env)
            assert result == {
                "CONTAINERIZED_AGENTS_GITHUB_TOKEN": "ghp_realtoken123",
                "PERPLEXITY_API_KEY": "pplx-realkey456",
            }


class TestLoadMcpConfig:
    """Tests for config file loading."""

    def test_load_from_explicit_path(self, tmp_path):
        config = {"mcpServers": {"test": {"command": "echo"}}}
        config_file = tmp_path / "mcp.json"
        config_file.write_text(json.dumps(config))

        result = load_mcp_config(str(config_file))
        assert result == config

    def test_load_from_env_var_path(self, tmp_path):
        config = {"mcpServers": {"test": {"command": "echo"}}}
        config_file = tmp_path / "custom_mcp.json"
        config_file.write_text(json.dumps(config))

        with patch.dict(os.environ, {"STRANDS_MCP_CONFIG": str(config_file)}):
            result = load_mcp_config()
        assert result == config

    def test_load_from_inline_json(self):
        config = {"mcpServers": {"test": {"command": "echo"}}}
        with patch.dict(os.environ, {"MCP_SERVERS": json.dumps(config)}):
            result = load_mcp_config()
        assert result == config

    def test_missing_file_returns_empty(self):
        result = load_mcp_config("/nonexistent/path/mcp.json")
        assert result == {}

    def test_no_config_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.chdir(tmp_path)
        # Clear env vars
        monkeypatch.delenv("STRANDS_MCP_CONFIG", raising=False)
        monkeypatch.delenv("MCP_SERVERS", raising=False)
        result = load_mcp_config()
        assert result == {}
