"""Session management — start, end, list, query, summarize sessions.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def start_session(store, session_id: str, project: str = "", metadata: Optional[dict] = None) -> bool:
    """Start a new session.

    Args:
        session_id: Unique session identifier.
        project: Optional project namespace.
        metadata: Optional dict stored as JSON.

    Returns:
        True on success.
    """
    with store._lock:
        store._ensure_workspace(project)
        store._conn.execute(
            """INSERT OR IGNORE INTO sessions (session_id, project, status, metadata)
               VALUES (?, ?, 'active', ?)""",
            (session_id, project, json.dumps(metadata or {})),
        )
        store._conn.commit()
    return True


def end_session(store, session_id: str, summary: str = "") -> bool:
    """End an active session.

    Args:
        session_id: Session to end.
        summary: Optional summary of the session.

    Returns:
        True if the session was found and ended, False otherwise.
    """
    with store._lock:
        # Count facts for this session
        fact_count = store._conn.execute(
            "SELECT COUNT(*) FROM facts WHERE session_id = ? AND (deleted IS NULL OR deleted = 0)",
            (session_id,),
        ).fetchone()[0]
        c = store._conn.execute(
            "UPDATE sessions SET status='ended', ended_at=CURRENT_TIMESTAMP, summary=?, fact_count=? WHERE session_id=?",
            (summary, fact_count, session_id),
        )
        store._conn.commit()
    return c.rowcount > 0


def get_session(store, session_id: str) -> Optional[dict]:
    """Get session details by ID.

    Args:
        session_id: Session identifier.

    Returns:
        Session dict with all columns, or None if not found.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def generate_session_summary(store, session_id: str) -> dict:
    """Generate a structured summary of a session from its facts.

    Best-effort aggregation: missing sections return empty defaults.

    Args:
        session_id: The session identifier to summarize.

    Returns:
        Dict with keys ``goal`` (str), ``discoveries`` (list[str]),
        ``accomplished`` (list[str]), ``next_steps`` (str).
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT content, category FROM facts "
            "WHERE session_id = ? AND (deleted IS NULL OR deleted = 0)",
            (session_id,),
        ).fetchall()

    goal: str = ""
    discoveries: list[str] = []
    accomplished: list[str] = []
    next_steps: str = ""

    for row in rows:
        content = row["content"]
        category = row["category"]

        # All session facts are "accomplished"
        accomplished.append(content)

        # Goal detection — fact starting with "## Goal"
        if not goal and content.lstrip().upper().startswith("## GOAL"):
            goal = content

        # Next steps detection — fact starting with "## Next Steps" or "Next Steps:"
        if content.lstrip().upper().startswith("## NEXT STEPS"):
            next_steps = content
        elif "Next Steps:" in content or "Next steps:" in content:
            if not next_steps:
                next_steps = content

        # Discoveries — facts with category discovery or bugfix
        if category in ("discovery", "bugfix"):
            discoveries.append(content)

    return {
        "goal": goal,
        "discoveries": discoveries,
        "accomplished": accomplished,
        "next_steps": next_steps,
    }


def list_sessions(store, project: str = "", limit: int = 10) -> list[dict]:
    """Get recent ended sessions, newest first.

    Args:
        project: Optional project name to filter.
        limit: Max number of sessions to return (default: 10).

    Returns:
        List of session dicts.
    """
    with store._lock:
        where = ["status = 'ended'"]
        params: list = []
        if project:
            where.append("project = ?")
            params.append(project)
        w = " AND ".join(where)
        rows = store._conn.execute(
            f"SELECT session_id, project, summary, fact_count, started_at, ended_at "
            f"FROM sessions WHERE {w} ORDER BY ended_at DESC, session_id DESC LIMIT ?",
            params + [limit],
        ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("summary") and len(d["summary"]) > 200:
            d["summary"] = d["summary"][:200]
        result.append(d)
    return result
