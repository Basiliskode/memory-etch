"""Event log — mutation journal for EtchStore.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _log_event(
    store,
    event_type: str,
    fact_id: Optional[int] = None,
    project: str = "",
    metadata: Optional[dict] = None,
) -> int:
    """Record a mutation event in the append-only event log.

    Caller MUST hold ``store._lock`` — all mutation methods already do.

    Args:
        event_type: One of the ``fact_*`` or ``relation_*`` event types.
        fact_id: Optional fact ID the event relates to.
        project: Optional project name.
        metadata: Optional JSON-compatible dict with event-specific fields.

    Returns:
        The ``event_id`` of the newly inserted row.
    """
    cur = store._conn.execute(
        """INSERT INTO event_log (event_type, fact_id, project, metadata)
           VALUES (?, ?, ?, ?)""",
        (event_type, fact_id, project, json.dumps(metadata or {})),
    )
    return cur.lastrowid


def get_event_log(
    store,
    event_type: Optional[str] = None,
    fact_id: Optional[int] = None,
    project: str = "",
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get events from the mutation journal.

    Args:
        event_type: Optional filter by event type.
        fact_id: Optional filter by fact_id.
        project: Optional filter by project.
        limit: Max results (default: 50).
        offset: Pagination offset (default: 0).

    Returns:
        List of event dicts, newest first. The ``metadata`` field is
        parsed from JSON into a dict.
    """
    with store._lock:
        conditions: list[str] = []
        params: list = []
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if fact_id is not None:
            conditions.append("fact_id = ?")
            params.append(fact_id)
        if project:
            conditions.append("project = ?")
            params.append(project)

        where = ""
        if conditions:
            where = " WHERE " + " AND ".join(conditions)

        sql = f"SELECT * FROM event_log{where} ORDER BY event_id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = store._conn.execute(sql, params).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d["metadata"] = json.loads(d.get("metadata", "{}"))
        except (json.JSONDecodeError, TypeError):
            d["metadata"] = {}
        result.append(d)
    return result
