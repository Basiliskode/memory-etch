"""Backward-compatible API aliases for EtchStore.

These functions replicate the enriched-return / error-raising behavior
of the original EtchStore methods before they were split into sub-modules.
They receive ``store`` (an EtchStore instance) as first argument.
"""

import logging
from typing import Optional

from . import _relations

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# get_timeline — full implementation (not in any other sub-module)
# ------------------------------------------------------------------


def get_timeline(store, fact_id: int, before: int = 5, after: int = 5) -> dict:
    """Get chronological context around a fact.

    Args:
        fact_id: Anchor fact ID.
        before: Number of preceding facts to include (default: 5).
        after: Number of subsequent facts to include (default: 5).

    Returns:
        Dict with keys ``fact``, ``before``, ``after``.
    """
    with store._lock:
        anchor = store._conn.execute(
            "SELECT fact_id, content, session_id FROM facts WHERE fact_id = ?",
            (fact_id,),
        ).fetchone()
        if not anchor:
            return {"fact": None, "before": [], "after": []}

        session_id = anchor["session_id"] or ""
        b4: list = []
        aft: list = []

        if session_id:
            try:
                b4 = store._conn.execute(
                    "SELECT fact_id, content, category, tags, trust_score, created_at FROM facts "
                    "WHERE fact_id < ? AND session_id = ? ORDER BY fact_id DESC LIMIT ?",
                    (fact_id, session_id, before),
                ).fetchall()
                aft = store._conn.execute(
                    "SELECT fact_id, content, category, tags, trust_score, created_at FROM facts "
                    "WHERE fact_id > ? AND session_id = ? ORDER BY fact_id ASC LIMIT ?",
                    (fact_id, session_id, after),
                ).fetchall()
            except Exception:
                pass

    return {
        "fact": dict(anchor),
        "before": [dict(r) for r in b4],
        "after": [dict(r) for r in aft],
    }


# ------------------------------------------------------------------
# session_start — enriched return
# ------------------------------------------------------------------


def session_start(
    store,
    session_id: str,
    project: str = "",
    metadata: Optional[dict] = None,
) -> dict:
    """Alias for ``start_session`` — returns enriched dict.

    Includes prior session count and top facts for the project.
    """
    prior = 0
    if project:
        row = store._conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE project = ?", (project,)
        ).fetchone()
        prior = row[0] if row else 0
    store.start_session(session_id, project, metadata)
    # Include facts matching project OR global facts (no project)
    if project:
        top_rows = store._conn.execute(
            "SELECT fact_id, content, trust_score FROM facts "
            "WHERE (deleted IS NULL OR deleted = 0) AND (project = ? OR project = '') "
            "ORDER BY trust_score DESC LIMIT 5",
            (project,),
        ).fetchall()
    else:
        top_rows = store._conn.execute(
            "SELECT fact_id, content, trust_score FROM facts "
            "WHERE (deleted IS NULL OR deleted = 0) "
            "ORDER BY trust_score DESC LIMIT 5",
        ).fetchall()
    return {
        "session_id": session_id,
        "prior_session_count": prior,
        "top_facts": [dict(r) for r in top_rows],
    }


# ------------------------------------------------------------------
# session_end — simple alias
# ------------------------------------------------------------------


def session_end(store, session_id: str, summary: str = "") -> bool:
    """Alias for ``end_session``."""
    return store.end_session(session_id, summary)


# ------------------------------------------------------------------
# timeline — error-raising alias for get_timeline
# ------------------------------------------------------------------


def timeline(store, fact_id: int, before: int = 5, after: int = 5) -> dict:
    """Alias for ``get_timeline`` with error-raising behavior.

    Raises:
        KeyError: If the fact does not exist.
        ValueError: If the fact has no session association.
    """
    result = store.get_timeline(fact_id, before, after)
    if not result["fact"]:
        raise KeyError(f"fact_id {fact_id} not found")
    if not result["fact"].get("session_id"):
        raise ValueError("no session association")
    return result


# ------------------------------------------------------------------
# judge_relation — thin compat wrapper
# ------------------------------------------------------------------


def judge_relation(
    store,
    fact_id_a: int,
    fact_id_b: int,
    relation_type: str = "related",
    confidence: float = 0.5,
    judged_by: str = "auto",
) -> dict:
    """Alias for ``add_relation`` — returns enriched dict.

    Delegates to ``_relations.judge_relation``.
    """
    return _relations.judge_relation(
        store, fact_id_a, fact_id_b, relation_type, confidence, judged_by,
    )


# ------------------------------------------------------------------
# get_recent_sessions — alias for list_sessions
# ------------------------------------------------------------------


def get_recent_sessions(store, project: str = "", limit: int = 10) -> list[dict]:
    """Alias for ``list_sessions``."""
    return store.list_sessions(project, limit)


# ------------------------------------------------------------------
# search_facts — alias for search
# ------------------------------------------------------------------


def search_facts(
    store,
    query: str,
    limit: int = 10,
    exclude_deleted: bool = True,
    project: str = "",
    scope: str = "canonical",
    source_harness: str = "",
    source_agent: str = "",
    source_kind: str = "",
    fact_type: str = "",
) -> list[dict]:
    """Alias for ``search``."""
    return store.search(
        query, limit=limit, exclude_deleted=exclude_deleted, project=project,
        scope=scope, source_harness=source_harness, source_agent=source_agent,
        source_kind=source_kind, fact_type=fact_type,
    )
