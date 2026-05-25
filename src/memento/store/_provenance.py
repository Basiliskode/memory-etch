"""Provenance tracking — derivation links, provenance queries, derivation trees.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import json
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


def _add_derivation_link(
    store,
    source_fact_id: int,
    derived_fact_id: int,
    judged_by: str = "auto",
) -> bool:
    """Record that *derived_fact_id* was derived from *source_fact_id*.

    This is a convenience wrapper around ``add_relation`` that enforces
    the ``derived_from`` relation type and logs a structured event.

    Args:
        source_fact_id: The original fact ID.
        derived_fact_id: The fact that was created based on the source.
        judged_by: Who or what established the derivation (default: "auto").

    Returns:
        True if the relation was inserted, False if it already exists.
    """
    result = store.add_relation(
        source_fact_id,
        derived_fact_id,
        relation_type="derived_from",
        judged_by=judged_by,
    )
    if result:
        with store._lock:
            store._log_event(
                "derivation_added",
                fact_id=derived_fact_id,
                metadata={
                    "source_fact_id": source_fact_id,
                    "derived_fact_id": derived_fact_id,
                    "judged_by": judged_by,
                },
            )
    return result


def get_provenance(store, fact_id: int) -> dict:
    """Return the COMPLETE provenance trail for a single fact.

    Includes fact metadata, recursive derivation ancestors, event log
    entries, session information (if available), and a count of facts
    derived from this one.

    Args:
        fact_id: Fact ID to look up.

    Returns:
        Dict with keys ``fact_id``, ``content``, ``created_at``,
        ``provenance``, ``derivation_ancestors``, ``event_log``,
        ``session`` (optional), and ``derived_facts_count``.

    Raises:
        KeyError: If *fact_id* does not exist.
    """
    with store._lock:
        row = store._conn.execute(
            """SELECT fact_id, content, created_at,
                      source_harness, source_agent, source_kind, scope,
                      session_id, project
               FROM facts WHERE fact_id = ?""",
            (fact_id,),
        ).fetchone()
    if not row:
        raise KeyError(f"fact_id {fact_id} not found")

    fact = dict(row)

    # --- Provenance metadata ---
    provenance = {
        "source_harness": fact["source_harness"] or "",
        "source_agent": fact["source_agent"] or "",
        "source_kind": fact["source_kind"] or "",
        "scope": fact["scope"] or "canonical",
        "session_id": fact["session_id"] or "",
        "project": fact["project"] or "",
    }

    # --- Derivation ancestors (recursive walk backward) ---
    ancestors: list[dict] = []
    visited: set[int] = {fact_id}
    queue: deque = deque()
    # Find immediate parents
    with store._lock:
        rows = store._conn.execute(
            """SELECT r.fact_id_a, r.confidence, r.judged_by,
                      f.content, f.source_harness, f.source_agent
               FROM fact_relations r
               JOIN facts f ON f.fact_id = r.fact_id_a
               WHERE r.fact_id_b = ? AND r.relation_type = 'derived_from'""",
            (fact_id,),
        ).fetchall()
    for r in rows:
        parent_id = r["fact_id_a"]
        visited.add(parent_id)
        entry = {
            "fact_id": parent_id,
            "content": r["content"],
            "relation_type": "derived_from",
            "depth": 1,
            "confidence": r["confidence"],
            "judged_by": r["judged_by"],
            "source_harness": r["source_harness"] or "",
            "source_agent": r["source_agent"] or "",
        }
        ancestors.append(entry)
        queue.append((parent_id, 1))

    # BFS to find grand-parents, etc. (up to max_depth=10)
    max_depth = 10
    while queue and len(ancestors) < 100:
        current_id, depth = queue.popleft()
        if depth >= max_depth:
            continue
        with store._lock:
            rows = store._conn.execute(
                """SELECT r.fact_id_a, r.confidence, r.judged_by,
                          f.content, f.source_harness, f.source_agent
                   FROM fact_relations r
                   JOIN facts f ON f.fact_id = r.fact_id_a
                   WHERE r.fact_id_b = ? AND r.relation_type = 'derived_from'""",
                (current_id,),
            ).fetchall()
        for r in rows:
            parent_id = r["fact_id_a"]
            if parent_id in visited:
                continue
            visited.add(parent_id)
            entry = {
                "fact_id": parent_id,
                "content": r["content"],
                "relation_type": "derived_from",
                "depth": depth + 1,
                "confidence": r["confidence"],
                "judged_by": r["judged_by"],
                "source_harness": r["source_harness"] or "",
                "source_agent": r["source_agent"] or "",
            }
            ancestors.append(entry)
            queue.append((parent_id, depth + 1))

    # --- Event log entries for this fact ---
    event_entries: list[dict] = []
    with store._lock:
        events = store._conn.execute(
            """SELECT event_type, created_at, metadata
               FROM event_log
               WHERE fact_id = ?
               ORDER BY created_at DESC""",
            (fact_id,),
        ).fetchall()
    for ev in events:
        parsed_meta = {}
        try:
            parsed_meta = json.loads(ev["metadata"]) if ev["metadata"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        event_entries.append({
            "event_type": ev["event_type"],
            "created_at": ev["created_at"],
            "metadata": parsed_meta,
        })

    # --- Session info ---
    session_info: Optional[dict] = None
    if fact.get("session_id"):
        with store._lock:
            srow = store._conn.execute(
                """SELECT session_id, started_at, ended_at, fact_count, summary
                   FROM sessions WHERE session_id = ?""",
                (fact["session_id"],),
            ).fetchone()
        if srow:
            session_info = {
                "session_id": srow["session_id"],
                "started_at": srow["started_at"],
                "ended_at": srow["ended_at"],
                "fact_count": srow["fact_count"],
                "summary": srow["summary"] or "",
            }

    # --- Count of facts derived FROM this fact ---
    with store._lock:
        derived_count_row = store._conn.execute(
            """SELECT COUNT(*) AS cnt FROM fact_relations
               WHERE fact_id_a = ? AND relation_type = 'derived_from'""",
            (fact_id,),
        ).fetchone()
    derived_facts_count = derived_count_row["cnt"] if derived_count_row else 0

    result = {
        "fact_id": fact["fact_id"],
        "content": fact["content"],
        "created_at": fact["created_at"],
        "provenance": provenance,
        "derivation_ancestors": ancestors,
        "event_log": event_entries,
        "derived_facts_count": derived_facts_count,
    }
    if session_info is not None:
        result["session"] = session_info
    return result


def get_derivation_tree(store, fact_id: int, max_depth: int = 5) -> dict:
    """Return a recursive tree of facts DERIVED FROM this fact.

    Performs a BFS forward through ``derived_from`` relations
    (fact_id_a = source), building a nested tree structure.

    Args:
        fact_id: Root fact ID.
        max_depth: Maximum depth of the tree (default: 5).

    Returns:
        Dict with ``root`` (fact metadata) and ``derivations`` (list of
        child nodes, each with nested ``children``).

    Raises:
        KeyError: If *fact_id* does not exist.
    """
    with store._lock:
        row = store._conn.execute(
            """SELECT fact_id, content, category, created_at
               FROM facts WHERE fact_id = ?""",
            (fact_id,),
        ).fetchone()
    if not row:
        raise KeyError(f"fact_id {fact_id} not found")

    root = {
        "fact_id": row["fact_id"],
        "content": row["content"],
        "category": row["category"],
        "created_at": row["created_at"],
    }

    def _build_children(current_id: int, depth: int, visited: set[int]) -> list[dict]:
        """Recursively build child nodes for *current_id*."""
        if depth >= max_depth:
            return []
        children: list[dict] = []
        with store._lock:
            rows = store._conn.execute(
                """SELECT r.fact_id_b, r.confidence, r.judged_by,
                          f.content, f.source_harness, f.source_agent
                   FROM fact_relations r
                   JOIN facts f ON f.fact_id = r.fact_id_b
                   WHERE r.fact_id_a = ? AND r.relation_type = 'derived_from'""",
                (current_id,),
            ).fetchall()
        for r in rows:
            child_id = r["fact_id_b"]
            if child_id in visited:
                continue
            visited.add(child_id)
            node = {
                "fact_id": child_id,
                "content": r["content"],
                "relation_type": "derived_from",
                "confidence": r["confidence"],
                "judged_by": r["judged_by"],
                "depth": depth + 1,
                "source_harness": r["source_harness"] or "",
                "source_agent": r["source_agent"] or "",
                "children": _build_children(child_id, depth + 1, visited),
            }
            children.append(node)
        return children

    derivations = _build_children(fact_id, 0, {fact_id})

    return {"root": root, "derivations": derivations}
