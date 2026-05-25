"""MCP server package for Memory Etch.

Provides a FastMCP stdio server exposing EtchStore as 9 MCP tools.
"""

from .server import get_store, server

__all__ = ["server", "get_store"]
