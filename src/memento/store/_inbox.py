"""Inbox management — review, promote, and reject inbox facts.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def list_inbox(
    store,
    project: str = "",
    source_harness: str = "",
    limit: int = 50,
) -> list[dict]:
    """List inbox facts (scope='inbox') for review.

    Returns non-deleted facts where ``scope='inbox'``, filtered by
    project and/or source_harness if provided, newest first.

    Args:
        project: Optional project filter.
        source_harness: Optional source harness filter.
        limit: Max results (default: 50).

    Returns:
        List of fact dicts.
    """
    with store._lock:
        conditions = [
            "(f.deleted IS NULL OR f.deleted = 0)",
            "f.scope = 'inbox'",
        ]
        params: list = []
        if project:
            conditions.append("f.project = ?")
            params.append(project)
        if source_harness:
            conditions.append("f.source_harness = ?")
            params.append(source_harness)
        w = " AND ".join(conditions)
        rows = store._conn.execute(
            f"""SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                       f.created_at, f.updated_at, f.project, f.topic_key,
                       f.revision_count, f.importance, f.session_id,
                       f.source_harness, f.source_agent, f.source_kind, f.scope,
                       f.fact_type, f.deleted, f.deleted_reason
                FROM facts f
                WHERE {w}
                ORDER BY f.fact_id DESC
                LIMIT ?""",
            params + [limit],
        ).fetchall()
    return [dict(r) for r in rows]


def promote_fact(store, fact_id: int) -> bool:
    """Promote an inbox fact to canonical scope.

    Changes ``scope`` from ``'inbox'`` to ``'canonical'`` and updates
    the timestamp. Only affects facts where ``scope='inbox'`` and
    ``(deleted IS NULL OR deleted = 0)``.

    Args:
        fact_id: ID of the inbox fact to promote.

    Returns:
        True if the fact was promoted, False if no matching inbox fact.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        fact_project = row["project"] if row else ""
        cur = store._conn.execute(
            """UPDATE facts SET scope = 'canonical', updated_at = CURRENT_TIMESTAMP
               WHERE fact_id = ? AND scope = 'inbox'
               AND (deleted IS NULL OR deleted = 0)""",
            (fact_id,),
        )
        store._conn.commit()
        if cur.rowcount > 0:
            store._log_event("fact_promoted", fact_id=fact_id, project=fact_project,
                             metadata={"from_scope": "inbox", "to_scope": "canonical"})
        store._invalidate_hrr_cache(fact_id)
    return cur.rowcount > 0


def reject_fact(store, fact_id: int, reason: str = "") -> bool:
    """Reject (soft-delete) an inbox fact.

    Soft-deletes the fact and stores the rejection reason in
    ``deleted_reason``. Only affects non-deleted inbox facts.

    Args:
        fact_id: ID of the inbox fact to reject.
        reason: Optional rejection reason (default: "").

    Returns:
        True if the fact was rejected, False if no matching inbox fact.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        fact_project = row["project"] if row else ""
        cur = store._conn.execute(
            """UPDATE facts SET deleted = 1, deleted_reason = ?,
                    updated_at = CURRENT_TIMESTAMP
               WHERE fact_id = ? AND scope = 'inbox'
               AND (deleted IS NULL OR deleted = 0)""",
            (reason, fact_id),
        )
        store._conn.commit()
        if cur.rowcount > 0:
            store._log_event("fact_rejected", fact_id=fact_id, project=fact_project,
                             metadata={"reason": reason})
        store._invalidate_hrr_cache(fact_id)
    return cur.rowcount > 0
