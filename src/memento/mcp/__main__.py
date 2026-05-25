"""Entry point for ``python -m memento.mcp``.

Runs the FastMCP stdio server.
"""

from .server import server

if __name__ == "__main__":
    server.run(transport="stdio")
