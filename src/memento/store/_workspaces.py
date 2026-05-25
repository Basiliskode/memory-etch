"""Workspace management — create, read, update, delete, list, stats.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import json
import logging
import sqlite3
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _ensure_workspace(store, name: str) -> None:
    """Auto-create workspace if it doesn't exist. Called inside store._lock."""
    if not name:
        return
    store._conn.execute(
        "INSERT OR IGNORE INTO workspaces (name) VALUES (?)",
        (name,),
    )


def create_workspace(
    store,
    name: str,
    description: str = "",
    tags: Optional[list] = None,
    settings: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create a new workspace or return existing if name already taken.

    Args:
        name: Unique workspace name.
        description: Optional description.
        tags: Optional list of tags.
        settings: Optional dict of settings.
        metadata: Optional dict of metadata.

    Returns:
        Workspace dict with parsed JSON fields.
    """
    with store._lock:
        store._conn.execute(
            """INSERT OR IGNORE INTO workspaces (name, description, tags, settings, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (name, description, json.dumps(tags or []),
             json.dumps(settings or {}), json.dumps(metadata or {})),
        )
        store._conn.commit()
        row = store._conn.execute(
            "SELECT * FROM workspaces WHERE name = ?", (name,)
        ).fetchone()
    return _parse_workspace_row(store, row) if row else {}


def get_workspace(store, name: str) -> Optional[dict]:
    """Get a workspace by name.

    Args:
        name: Workspace name.

    Returns:
        Workspace dict with parsed JSON fields, or None if not found.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM workspaces WHERE name = ? AND (deleted IS NULL OR deleted = 0)",
            (name,),
        ).fetchone()
    return _parse_workspace_row(store, row) if row else None


def _parse_workspace_row(store, row: sqlite3.Row) -> dict:
    """Parse a workspace row, converting JSON fields to Python objects."""
    d = dict(row)
    for field in ("tags", "settings", "metadata"):
        try:
            d[field] = json.loads(d.get(field, "{}"))
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def update_workspace(store, name: str, **kwargs) -> bool:
    """Update a workspace's mutable fields.

    Accepts keyword args: ``description``, ``tags``, ``settings``, ``metadata``.
    Tags must be a list; settings and metadata must be dicts.

    Returns:
        True if the workspace was found and updated.
    """
    allowed = {"description", "tags", "settings", "metadata"}
    updates: dict[str, Any] = {}
    for key, val in kwargs.items():
        if key not in allowed:
            raise ValueError(f"Unknown workspace field: '{key}'. Allowed: {allowed}")
        if key == "tags":
            updates[key] = json.dumps(val if val is not None else [])
        elif key in ("settings", "metadata"):
            updates[key] = json.dumps(val if val is not None else {})
        else:
            updates[key] = val

    if not updates:
        return False

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    set_clause += ", updated_at = datetime('now')"
    values = list(updates.values()) + [name]

    with store._lock:
        cur = store._conn.execute(
            f"UPDATE workspaces SET {set_clause} WHERE name = ? AND (deleted IS NULL OR deleted = 0)",
            values,
        )
        store._conn.commit()
    return cur.rowcount > 0


def delete_workspace(store, name: str) -> bool:
    """Soft-delete a workspace.

    Does NOT cascade-delete facts — just marks the workspace as deleted.

    Args:
        name: Workspace name.

    Returns:
        True if the workspace was found and soft-deleted.
    """
    with store._lock:
        cur = store._conn.execute(
            "UPDATE workspaces SET deleted = 1, updated_at = datetime('now')"
            " WHERE name = ? AND (deleted IS NULL OR deleted = 0)",
            (name,),
        )
        store._conn.commit()
    return cur.rowcount > 0


def list_workspaces(store, include_deleted: bool = False, include_stats: bool = False) -> list[dict]:
    """List workspaces.

    Args:
        include_deleted: If True, include soft-deleted workspaces.
        include_stats: If True, include live fact/session counts
            (slower — queries facts and sessions tables).

    Returns:
        List of workspace dicts with parsed JSON fields.
    """
    with store._lock:
        if include_deleted:
            rows = store._conn.execute(
                "SELECT * FROM workspaces ORDER BY name"
            ).fetchall()
        else:
            rows = store._conn.execute(
                "SELECT * FROM workspaces WHERE (deleted IS NULL OR deleted = 0) ORDER BY name"
            ).fetchall()

    result = [_parse_workspace_row(store, r) for r in rows]

    if include_stats:
        for ws in result:
            stats = store.workspace_stats(ws["name"])
            ws["fact_count"] = stats["fact_count"]
            ws["session_count"] = stats["session_count"]
            ws["last_active"] = stats["last_active"]

    return result


def workspace_stats(store, name: str) -> dict:
    """Get live statistics for a workspace.

    Queries facts and sessions tables for current counts.

    Args:
        name: Workspace name.

    Returns:
        Dict with keys ``name``, ``fact_count``, ``session_count``,
        ``last_active``.
    """
    with store._lock:
        fact_count = store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE project = ? AND (deleted IS NULL OR deleted = 0)",
            (name,),
        ).fetchone()[0]
        session_count = store._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE project = ?",
            (name,),
        ).fetchone()[0]
        last_active_row = store._conn.execute(
            "SELECT MAX(created_at) FROM facts WHERE project = ? AND (deleted IS NULL OR deleted = 0)",
            (name,),
        ).fetchone()[0]
    return {
        "name": name,
        "fact_count": fact_count,
        "session_count": session_count,
        "last_active": last_active_row,
    }
