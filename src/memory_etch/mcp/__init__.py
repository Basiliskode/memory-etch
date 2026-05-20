"""MCP server package for Memory Etch.

Provides a FastMCP stdio server exposing EtchStore as 6 MCP tools.
"""

from .server import server, get_store

__all__ = ["server", "get_store"]
