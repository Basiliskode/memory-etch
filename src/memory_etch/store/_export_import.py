"""Export/Import, Stats, Projects — lifecycle and analytics.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import json
import logging

logger = logging.getLogger(__name__)


def export_memory(store, path: str) -> dict:
    """Export all memory data to a JSON file.

    Includes all active facts, sessions, fact relations, and turn buffer
    entries. HRR vectors and embeddings are excluded from the dump.

    Args:
        path: File path for the JSON export.

    Returns:
        Stats dict with counts of exported items.
    """
    with store._lock:
        facts = store._conn.execute(
            "SELECT fact_id, content, category, tags, trust_score, importance, "
            "project, session_id, topic_key, revision_count, retrieval_count, "
            "consolidated, deleted, deleted_reason, created_at, updated_at, "
            "what, why, where_text, learned, scope, fact_type "
            "FROM facts ORDER BY fact_id"
        ).fetchall()

        sessions = store._conn.execute(
            "SELECT session_id, project, status, fact_count, summary, "
            "metadata, started_at, ended_at FROM sessions ORDER BY session_id"
        ).fetchall()

        relations = store._conn.execute(
            "SELECT relation_id, fact_id_a, fact_id_b, relation_type, "
            "confidence, judged_by, created_at FROM fact_relations ORDER BY relation_id"
        ).fetchall()

        turns = store._conn.execute(
            "SELECT turn_id, session_id, role, content, meaningful, created_at "
            "FROM turn_buffer ORDER BY turn_id"
        ).fetchall()

    exported_facts = []
    for row in facts:
        fact = dict(row)
        fact["where"] = fact.get("where_text", "")
        exported_facts.append(fact)

    data = {
        "version": 1,
        "facts": exported_facts,
        "sessions": [dict(r) for r in sessions],
        "relations": [dict(r) for r in relations],
        "turns": [dict(r) for r in turns],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

    return {
        "facts": len(data["facts"]),
        "sessions": len(data["sessions"]),
        "relations": len(data["relations"]),
        "turns": len(data["turns"]),
    }


def import_memory(store, path: str) -> dict:
    """Import memory data from a JSON file created by ``export_memory``.

    Facts are inserted via ``add_fact`` (respecting content dedup/topic upsert).
    Sessions, relations, and turn buffer entries are inserted directly.

    Args:
        path: File path to the JSON export.

    Returns:
        Stats dict with counts of imported items.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    imported = {"facts": 0, "sessions": 0, "relations": 0, "turns": 0}

    for row in data.get("facts", []):
        store.add_fact(
            content=row["content"],
            category=row.get("category", "general"),
            tags=row.get("tags", ""),
            trust_score=row.get("trust_score", 0.5),
            importance=row.get("importance", 0.5),
            project=row.get("project", ""),
            session_id=row.get("session_id", ""),
            topic_key=row.get("topic_key", ""),
            what=row.get("what", ""),
            why=row.get("why", ""),
            where_text=row.get("where", row.get("where_text", "")),
            learned=row.get("learned", ""),
            scope=row.get("scope", "canonical"),
            fact_type=row.get("fact_type", ""),
        )
        imported["facts"] += 1

    with store._lock:
        for row in data.get("sessions", []):
            store._conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (session_id, project, status, fact_count, summary, metadata, started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    row["session_id"], row.get("project", ""),
                    row.get("status", "ended"), row.get("fact_count", 0),
                    row.get("summary", ""), row.get("metadata", "{}"),
                    row.get("started_at"), row.get("ended_at"),
                ),
            )
            imported["sessions"] += 1

        for row in data.get("relations", []):
            store._conn.execute(
                """INSERT OR IGNORE INTO fact_relations
                   (fact_id_a, fact_id_b, relation_type, confidence, judged_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    row["fact_id_a"], row["fact_id_b"], row["relation_type"],
                    row.get("confidence", 1.0), row.get("judged_by", "import"),
                    row.get("created_at"),
                ),
            )
            imported["relations"] += 1

        for row in data.get("turns", []):
            store._conn.execute(
                """INSERT OR IGNORE INTO turn_buffer
                   (session_id, role, content, meaningful, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    row["session_id"], row["role"], row["content"],
                    row.get("meaningful", 0), row.get("created_at"),
                ),
            )
            imported["turns"] += 1

        store._conn.commit()

    return imported


def stats(store) -> dict:
    """Get database statistics.

    Returns:
        Dict with keys ``fact_count``, ``session_count``,
        ``relation_count``, ``extraction_count``, ``active_sessions``.
    """
    with store._lock:
        facts = store._conn.execute("SELECT COUNT(*) FROM facts WHERE (deleted IS NULL OR deleted = 0)").fetchone()[0]
        sessions = store._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        relations = 0
        extractions = 0
        active = 0
        try:
            relations = store._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
            extractions = store._conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
            active = store._conn.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0]
        except Exception:
            pass
    return {
        "fact_count": facts,
        "session_count": sessions,
        "relation_count": relations,
        "extraction_count": extractions,
        "active_sessions": active,
    }


def projects(store) -> list[str]:
    """List distinct project names that have facts.

    Returns:
        Sorted list of non-empty project names.
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT DISTINCT project FROM facts WHERE project != '' AND (deleted IS NULL OR deleted = 0) ORDER BY project"
        ).fetchall()
    return [r[0] for r in rows]
