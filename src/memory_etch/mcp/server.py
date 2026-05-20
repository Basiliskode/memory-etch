"""FastMCP stdio server for Memory Etch.

Exposes ``EtchStore`` as 6 MCP tools:

- ``add_fact``
- ``search_facts``
- ``get_fact``
- ``delete_fact``
- ``get_timeline``
- ``search_similar``

The store is a module-level singleton initialized from the
``MEMORY_ETCH_DB_PATH`` environment variable.

Usage:
    MEMORY_ETCH_DB_PATH=~/.memory-etch/etch.db python -m memory_etch.mcp
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from memory_etch import EtchStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton store
# ---------------------------------------------------------------------------

_store: Optional[EtchStore] = None


def get_store() -> EtchStore:
    """Get or create the singleton EtchStore instance.

    The database path is read from the ``MEMORY_ETCH_DB_PATH`` environment
    variable.  If not set, defaults to ``:memory:`` (useful for testing).
    For production, set it to ``~/.memory-etch/etch.db``.

    The store is created once and cached for the lifetime of the process.
    """
    global _store
    if _store is not None:
        return _store

    db_path = os.environ.get("MEMORY_ETCH_DB_PATH")
    if not db_path:
        db_path = ":memory:"

    _store = EtchStore(db_path, auto_migrate=True)
    return _store


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

server = FastMCP("memory-etch", log_level="WARNING")


@server.tool()
def add_fact(
    content: str,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    topic_key: Optional[str] = None,
    source: Optional[str] = None,
    metadata: Optional[str] = None,
) -> str:
    """Add a fact to the memory store.

    Args:
        content: Fact text content.
        project: Optional project namespace.
        session_id: Optional session identifier.
        topic_key: Optional topic key for upsert behavior.
        source: Optional source description (stored in ``what`` field).
        metadata: Optional JSON metadata string (stored in ``learned`` field).

    Returns:
        JSON string with ``{"id": int, "status": "created"|"updated"}``.
    """
    store = get_store()
    what_text = source or ""
    learned_text = metadata or ""
    fid = store.add_fact(
        content=content,
        project=project or "",
        session_id=session_id or "",
        topic_key=topic_key or "",
        what=what_text,
        learned=learned_text,
    )
    # Determine status: if topic_key was provided and content differs → "updated"
    status = "updated" if (topic_key and store.get_fact(fid) and store.get_fact(fid)["content"] == content) else "created"  # noqa: E501
    # Simpler: always "created" unless we can detect upsert.  Use revision_count.
    fact = store.get_fact(fid)
    if fact and fact.get("revision_count", 0) > 0:
        status = "updated"
    return json.dumps({"id": fid, "status": status})


@server.tool()
def search_facts(
    query: str,
    limit: int = 10,
    project: Optional[str] = None,
    mode: str = "auto",
) -> str:
    """Search facts by full-text query.

    Args:
        query: Full-text search query.
        limit: Max results (default: 10).
        project: Optional project filter.
        mode: Search mode (default: "auto").

    Returns:
        JSON array of fact dicts with ``id``, ``content``, ``score``,
        ``project``, ``summary`` keys.
    """
    store = get_store()
    results = store.search_facts(query=query, limit=limit, project=project or "")
    output = []
    for r in results:
        content = r.get("content", "")
        output.append({
            "id": r["fact_id"],
            "content": content,
            "score": r.get("trust_score", 0.0),
            "project": r.get("project", ""),
            "summary": content[:200] if content else "",
        })
    return json.dumps(output)


@server.tool()
def get_fact(fact_id: int) -> str:
    """Get a single fact by its ID.

    Args:
        fact_id: The fact ID to retrieve.

    Returns:
        JSON string with the full fact dict, or ``{"status": "not_found"}``.
    """
    store = get_store()
    fact = store.get_fact(fact_id)
    if fact is None:
        return json.dumps({"status": "not_found"})
    # Remove binary blobs for JSON serialisation
    fact.pop("hrr_vector", None)
    fact.pop("embedding", None)
    return json.dumps(fact, default=str)


@server.tool()
def delete_fact(fact_id: int) -> str:
    """Permanently delete a fact by its ID.

    Args:
        fact_id: The fact ID to delete.

    Returns:
        JSON string with ``{"status": "deleted"|"not_found"}``.
    """
    store = get_store()
    fact = store.get_fact(fact_id)
    if fact is None:
        return json.dumps({"status": "not_found"})
    store.remove_fact(fact_id)
    return json.dumps({"status": "deleted"})


@server.tool()
def get_timeline(project: Optional[str] = None, limit: int = 20) -> str:
    """Get fact timeline, newest first.

    Args:
        project: Optional project filter.
        limit: Max entries (default: 20).

    Returns:
        JSON array of fact dicts.
    """
    store = get_store()
    results = store.list_facts(project=project or "", limit=limit)
    output = []
    for r in results:
        d = dict(r)
        d.pop("hrr_vector", None)
        d.pop("embedding", None)
        output.append(d)
    return json.dumps(output, default=str)


@server.tool()
def search_similar(query: str, limit: int = 5) -> str:
    """Search for facts similar to the given query text.

    Uses full-text search (FTS5) to find semantically related facts.

    Args:
        query: Text to find similar facts for.
        limit: Max results (default: 5).

    Returns:
        JSON array of fact dicts sorted by relevance.
    """
    store = get_store()
    results = store.search_facts(query=query, limit=limit)
    output = []
    for r in results:
        content = r.get("content", "")
        output.append({
            "id": r["fact_id"],
            "content": content,
            "score": r.get("trust_score", 0.0),
            "project": r.get("project", ""),
            "summary": content[:200] if content else "",
        })
    return json.dumps(output, default=str)
