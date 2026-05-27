"""Snapshots — checkpoints of the entire memory store.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_snapshot(
    store,
    name: str,
    description: str = "",
    tags: Optional[list[str]] = None,
    project: str = "",
) -> dict:
    """Create a new snapshot of the entire memory store.

    Captures all facts, sessions, relations, turns, event_log entries,
    and workspaces into a versioned JSON blob. Excludes HRR vectors
    and embeddings from facts.

    Args:
        name: Unique name for the snapshot.
        description: Optional human-readable description.
        tags: Optional list of tag strings.
        project: If non-empty, only facts for this project are captured.

    Returns:
        Metadata dict with keys: name, fact_count, session_count,
        workspace_count, relation_count, turn_count, event_count,
        state_hash, created_at.

    Raises:
        ValueError: If name is empty or a snapshot with this name exists.
    """
    if not name or not name.strip():
        raise ValueError("Snapshot name must not be empty")

    with store._lock:
        existing = store._conn.execute(
            "SELECT name FROM snapshots WHERE name = ?", (name.strip(),)
        ).fetchone()
        if existing:
            raise ValueError(f"Snapshot '{name}' already exists")

        # Query facts (exclude hrr_vector and embedding)
        facts = store._conn.execute(
            "SELECT fact_id, content, category, tags, trust_score, importance, "
            "project, session_id, topic_key, revision_count, retrieval_count, "
            "consolidated, deleted, deleted_reason, created_at, updated_at "
            "FROM facts ORDER BY fact_id",
        ).fetchall()

        sessions = store._conn.execute(
            "SELECT * FROM sessions ORDER BY session_id",
        ).fetchall()

        relations = store._conn.execute(
            "SELECT * FROM fact_relations ORDER BY relation_id",
        ).fetchall()

        turns = store._conn.execute(
            "SELECT * FROM turn_buffer ORDER BY turn_id",
        ).fetchall()

        event_log = store._conn.execute(
            "SELECT * FROM event_log ORDER BY event_id",
        ).fetchall()

        workspaces = store._conn.execute(
            "SELECT * FROM workspaces ORDER BY workspace_id",
        ).fetchall()

        atlas_maps = store._conn.execute(
            "SELECT * FROM atlas_maps ORDER BY map_id",
        ).fetchall()

        atlas_regions = store._conn.execute(
            "SELECT * FROM atlas_regions ORDER BY region_id",
        ).fetchall()

        atlas_edges = store._conn.execute(
            "SELECT * FROM atlas_edges ORDER BY edge_id",
        ).fetchall()

    # Build data dict (outside lock — no DB access)
    data = {
        "version": 2,
        "snapshot_name": name.strip(),
        "project": project,
        "facts": [dict(r) for r in facts],
        "sessions": [dict(r) for r in sessions],
        "relations": [dict(r) for r in relations],
        "turns": [dict(r) for r in turns],
        "event_log": [dict(r) for r in event_log],
        "workspaces": [dict(r) for r in workspaces],
        "atlas_maps": [dict(r) for r in atlas_maps],
        "atlas_regions": [dict(r) for r in atlas_regions],
        "atlas_edges": [dict(r) for r in atlas_edges],
    }

    # If project is non-empty, filter facts to only that project
    if project:
        data["facts"] = [f for f in data["facts"] if f.get("project") == project]

    json_str = json.dumps(data, default=str)
    state_hash = hashlib.sha256(json_str.encode()).hexdigest()
    tags_json = json.dumps(tags or [])

    fact_count = len(data["facts"])
    session_count = len(data["sessions"])
    workspace_count = len(data["workspaces"])
    relation_count = len(data["relations"])
    turn_count = len(data["turns"])
    event_count = len(data["event_log"])

    with store._lock:
        store._conn.execute(
            """INSERT INTO snapshots
               (name, description, tags, project, data, state_hash,
                fact_count, session_count, workspace_count,
                relation_count, turn_count, event_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name.strip(), description, tags_json, project,
                json_str, state_hash,
                fact_count, session_count, workspace_count,
                relation_count, turn_count, event_count,
            ),
        )
        store._conn.commit()
        store._log_event(
            "snapshot_created",
            project=project,
            metadata={
                "name": name.strip(),
                "fact_count": fact_count,
                "session_count": session_count,
            },
        )

    return {
        "name": name.strip(),
        "description": description,
        "tags": tags or [],
        "project": project,
        "fact_count": fact_count,
        "session_count": session_count,
        "workspace_count": workspace_count,
        "relation_count": relation_count,
        "turn_count": turn_count,
        "event_count": event_count,
        "state_hash": state_hash,
        "created_at": None,  # caller doesn't have this yet; get_snapshot fills it
    }


def restore_snapshot(store, name: str, merge: bool = False) -> dict:
    """Restore memory state from a snapshot.

    In replace mode (merge=False), all existing data is wiped before
    restoring. In merge mode, facts go through dedup (add_fact) and
    other tables use INSERT OR IGNORE.

    Args:
        name: Snapshot name to restore from.
        merge: If True, merge into existing data instead of replacing.

    Returns:
        Stats dict with counts of restored items.

    Raises:
        ValueError: If the snapshot is not found.
    """
    # Step 1: Read snapshot data under lock
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM snapshots WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            raise ValueError(f"Snapshot '{name}' not found")
        data = json.loads(row["data"])

        # Pre-extract for use outside lock
        facts_data = data.get("facts", [])
        sessions_data = data.get("sessions", [])
        relations_data = data.get("relations", [])
        turns_data = data.get("turns", [])
        workspaces_data = data.get("workspaces", [])
        atlas_maps_data = data.get("atlas_maps", [])
        atlas_regions_data = data.get("atlas_regions", [])
        atlas_edges_data = data.get("atlas_edges", [])
        restored_event_log = data.get("event_log", [])

        if not merge:
            for table in ("facts", "sessions", "fact_relations", "turn_buffer", "event_log", "workspaces",
                          "atlas_maps", "atlas_regions", "atlas_edges"):
                store._conn.execute(f"DELETE FROM {table}")
            store._conn.commit()

    # Step 2: Import facts through add_fact (outside lock — add_fact acquires its own)
    fact_count = 0
    for fact_row in facts_data:
        store.add_fact(
            content=fact_row["content"],
            category=fact_row.get("category", "general"),
            tags=fact_row.get("tags", ""),
            trust_score=fact_row.get("trust_score", 0.5),
            importance=fact_row.get("importance", 0.5),
            project=fact_row.get("project", ""),
            session_id=fact_row.get("session_id", ""),
            topic_key=fact_row.get("topic_key", ""),
        )
        fact_count += 1

    # Step 3: Direct-insert sessions, relations, turns, workspaces
    with store._lock:
        for s_row in sessions_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO sessions
                   (session_id, project, status, fact_count, summary, metadata, started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    s_row["session_id"], s_row.get("project", ""),
                    s_row.get("status", "ended"), s_row.get("fact_count", 0),
                    s_row.get("summary", ""), s_row.get("metadata", "{}"),
                    s_row.get("started_at"), s_row.get("ended_at"),
                ),
            )

        for r_row in relations_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO fact_relations
                   (fact_id_a, fact_id_b, relation_type, confidence, judged_by, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    r_row["fact_id_a"], r_row["fact_id_b"], r_row["relation_type"],
                    r_row.get("confidence", 1.0), r_row.get("judged_by", "import"),
                    r_row.get("created_at"),
                ),
            )

        for t_row in turns_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO turn_buffer
                   (session_id, role, content, meaningful, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    t_row["session_id"], t_row["role"], t_row["content"],
                    t_row.get("meaningful", 0), t_row.get("created_at"),
                ),
            )

        for w_row in workspaces_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO workspaces
                   (name, description, tags, settings, metadata, fact_count,
                    last_active, created_at, updated_at, deleted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    w_row["name"], w_row.get("description", ""),
                    w_row.get("tags", "[]"), w_row.get("settings", "{}"),
                    w_row.get("metadata", "{}"), w_row.get("fact_count", 0),
                    w_row.get("last_active"), w_row.get("created_at"),
                    w_row.get("updated_at"), w_row.get("deleted", 0),
                ),
            )

        for am_row in atlas_maps_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO atlas_maps
                   (map_id, name, description, tags, project, metadata,
                    node_count, created_at, updated_at, deleted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    am_row["map_id"], am_row["name"],
                    am_row.get("description", ""), am_row.get("tags", ""),
                    am_row.get("project", ""), am_row.get("metadata", "{}"),
                    am_row.get("node_count", 0), am_row.get("created_at"),
                    am_row.get("updated_at"), am_row.get("deleted", 0),
                ),
            )

        for ar_row in atlas_regions_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO atlas_regions
                   (region_id, map_id, parent_region_id, name, description,
                    tags, fact_count, created_at, deleted)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ar_row["region_id"], ar_row["map_id"],
                    ar_row.get("parent_region_id"), ar_row["name"],
                    ar_row.get("description", ""), ar_row.get("tags", ""),
                    ar_row.get("fact_count", 0), ar_row.get("created_at"),
                    ar_row.get("deleted", 0),
                ),
            )

        for ae_row in atlas_edges_data:
            store._conn.execute(
                """INSERT OR IGNORE INTO atlas_edges
                   (edge_id, map_id, source_type, source_id, target_type,
                    target_id, relation_type, weight, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ae_row["edge_id"], ae_row["map_id"],
                    ae_row["source_type"], ae_row["source_id"],
                    ae_row["target_type"], ae_row["target_id"],
                    ae_row.get("relation_type", "contains"),
                    ae_row.get("weight", 0.5),
                    ae_row.get("metadata", "{}"),
                    ae_row.get("created_at"),
                ),
            )

        store._conn.commit()
        store._log_event(
            "snapshot_restored",
            metadata={"name": name, "merge": merge},
        )

    return {
        "facts_restored": fact_count,
        "sessions_restored": len(sessions_data),
        "relations_restored": len(relations_data),
        "turns_restored": len(turns_data),
        "workspaces_restored": len(workspaces_data),
        "atlas_maps_restored": len(atlas_maps_data),
        "atlas_regions_restored": len(atlas_regions_data),
        "atlas_edges_restored": len(atlas_edges_data),
    }


def list_snapshots(store, project: str = "") -> list[dict]:
    """List all snapshots with metadata (excludes the full data column).

    Args:
        project: If non-empty, only snapshots for this project.

    Returns:
        List of snapshot metadata dicts ordered by created_at DESC.
    """
    with store._lock:
        if project:
            rows = store._conn.execute(
                """SELECT name, description, tags, project, state_hash,
                          fact_count, session_count, workspace_count,
                          relation_count, turn_count, event_count, created_at
                   FROM snapshots WHERE project = ?
                   ORDER BY created_at DESC""",
                (project,),
            ).fetchall()
        else:
            rows = store._conn.execute(
                """SELECT name, description, tags, project, state_hash,
                          fact_count, session_count, workspace_count,
                          relation_count, turn_count, event_count, created_at
                   FROM snapshots
                   ORDER BY created_at DESC""",
            ).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        d["tags"] = json.loads(d.get("tags", "[]"))
        result.append(d)
    return result


def get_snapshot(store, name: str) -> dict:
    """Get a full snapshot by name, including the parsed data.

    Args:
        name: Snapshot name.

    Returns:
        Full snapshot dict with parsed ``tags`` and ``data`` fields.

    Raises:
        ValueError: If the snapshot is not found.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM snapshots WHERE name = ?", (name,)
        ).fetchone()
        if not row:
            raise ValueError(f"Snapshot '{name}' not found")
        result = dict(row)
        result["tags"] = json.loads(result.get("tags", "[]"))
        result["data"] = json.loads(result["data"])
        return result


def delete_snapshot(store, name: str) -> bool:
    """Delete a snapshot by name.

    Args:
        name: Snapshot name to delete.

    Returns:
        True if the snapshot was deleted, False if not found.
    """
    with store._lock:
        cur = store._conn.execute(
            "DELETE FROM snapshots WHERE name = ?", (name,)
        )
        deleted = cur.rowcount > 0
        store._conn.commit()
        if deleted:
            store._log_event(
                "snapshot_deleted",
                metadata={"name": name},
            )
        return deleted


def snapshot_diff(store, name_a: str, name_b: str) -> dict:
    """Compare two snapshots by table counts.

    Args:
        name_a: First snapshot name.
        name_b: Second snapshot name.

    Returns:
        Dict with per-table counts and deltas.

    Raises:
        ValueError: If either snapshot is not found.
    """
    with store._lock:
        row_a = store._conn.execute(
            "SELECT * FROM snapshots WHERE name = ?", (name_a,)
        ).fetchone()
        if not row_a:
            raise ValueError(f"Snapshot '{name_a}' not found")
        row_b = store._conn.execute(
            "SELECT * FROM snapshots WHERE name = ?", (name_b,)
        ).fetchone()
        if not row_b:
            raise ValueError(f"Snapshot '{name_b}' not found")

    data_a = json.loads(row_a["data"])
    data_b = json.loads(row_b["data"])

    def _counts(key: str) -> dict:
        ca = len(data_a.get(key, []))
        cb = len(data_b.get(key, []))
        return {"a": ca, "b": cb, "delta": cb - ca}

    return {
        "name_a": name_a,
        "name_b": name_b,
        "facts": _counts("facts"),
        "sessions": _counts("sessions"),
        "relations": _counts("relations"),
        "turns": _counts("turns"),
        "event_log": _counts("event_log"),
        "workspaces": _counts("workspaces"),
        "atlas_maps": _counts("atlas_maps"),
        "atlas_regions": _counts("atlas_regions"),
        "atlas_edges": _counts("atlas_edges"),
    }
