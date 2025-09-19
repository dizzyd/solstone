#!/usr/bin/env python3
"""Tests for HTTP MCP integration in utils.py."""

import sys
from unittest.mock import patch

import pytest

from think.utils import create_mcp_client


class TestCreateMCPClientHTTP:
    """Test HTTP MCP client creation and URI handling."""

    def setup_method(self):
        """Clean up any stubbed fastmcp module from other tests."""
        if "fastmcp" in sys.modules:
            del sys.modules["fastmcp"]
        if "fastmcp.fastmcp" in sys.modules:
            del sys.modules["fastmcp.fastmcp"]

    def test_with_explicit_url(self):
        """Client uses explicitly provided URL."""
        with patch("fastmcp.Client") as mock_client:
            result = create_mcp_client(" http://127.0.0.1:6270/mcp/ ")

            mock_client.assert_called_once_with(
                "http://127.0.0.1:6270/mcp/", timeout=15.0
            )
            assert result == mock_client.return_value

    def test_empty_url_error(self):
        """Error is raised when provided URL is empty or whitespace."""
        with pytest.raises(RuntimeError, match="MCP server URL not provided"):
            create_mcp_client("   ")
