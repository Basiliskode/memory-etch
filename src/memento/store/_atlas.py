"""Atlas — semantic fact map layer.

Layers a navigable graph (maps → regions → edges) over existing facts
in EtchStore. Three tables:

- ``atlas_maps`` — top-level named containers with FTS5 search.
- ``atlas_regions`` — hierarchical sub-divisions within a map.
- ``atlas_edges`` — typed, weighted links between any entities
  (map, region, fact) with adjacency-list traversal.

Module-level functions receive ``store`` (the EtchStore instance) as
first argument, matching the existing delegation pattern.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Maps CRUD
# ---------------------------------------------------------------------------


def create_map(
    store,
    name: str,
    description: str = "",
    tags: str = "",
    project: str = "",
    metadata: str = "{}",
) -> int:
    """Create a new atlas map.

    Args:
        name: Human-readable name for the map.
        description: Optional description.
        tags: Comma-separated tags.
        project: Optional project namespace.
        metadata: JSON metadata string.

    Returns:
        The new map_id.

    Raises:
        ValueError: If name is empty.
    """
    if not name or not name.strip():
        raise ValueError("Map name must not be empty")
    with store._lock:
        cur = store._conn.execute(
            """INSERT INTO atlas_maps (name, description, tags, project, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (name.strip(), description, tags, project, metadata),
        )
        store._conn.commit()
        return cur.lastrowid


def get_map(store, map_id: int) -> Optional[dict]:
    """Get a single map by ID (excluding soft-deleted).

    Args:
        map_id: The map ID.

    Returns:
        Map dict or None if not found or deleted.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM atlas_maps WHERE map_id = ? AND (deleted IS NULL OR deleted = 0)",
            (map_id,),
        ).fetchone()
    return dict(row) if row else None


def update_map(store, map_id: int, **kwargs) -> bool:
    """Update a map's mutable fields.

    Accepts keyword args: ``name``, ``description``, ``tags``, ``metadata``.

    Args:
        map_id: The map ID to update.

    Returns:
        True if the map was found and updated, False otherwise.
    """
    allowed = {"name", "description", "tags", "metadata"}
    updates = {}
    for key, val in kwargs.items():
        if key not in allowed:
            raise ValueError(f"Unknown map field: '{key}'. Allowed: {allowed}")
        updates[key] = val

    if not updates:
        return True  # No-op is still "success"

    with store._lock:
        pairs = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [map_id]
        cur = store._conn.execute(
            f"UPDATE atlas_maps SET {pairs}, updated_at = CURRENT_TIMESTAMP "
            f"WHERE map_id = ? AND (deleted IS NULL OR deleted = 0)",
            values,
        )
        store._conn.commit()
        return cur.rowcount > 0


def delete_map(store, map_id: int) -> bool:
    """Soft-delete a map.

    Args:
        map_id: The map ID to delete.

    Returns:
        True if the map was found and soft-deleted.
    """
    with store._lock:
        cur = store._conn.execute(
            "UPDATE atlas_maps SET deleted = 1, updated_at = CURRENT_TIMESTAMP "
            "WHERE map_id = ? AND (deleted IS NULL OR deleted = 0)",
            (map_id,),
        )
        store._conn.commit()
        return cur.rowcount > 0


def list_maps(store, project: str = "", limit: int = 50) -> list[dict]:
    """List maps, excluding soft-deleted.

    Args:
        project: If non-empty, filter by project.
        limit: Max results (default 50).

    Returns:
        List of map dicts.
    """
    with store._lock:
        if project:
            rows = store._conn.execute(
                "SELECT * FROM atlas_maps "
                "WHERE (deleted IS NULL OR deleted = 0) AND project = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            ).fetchall()
        else:
            rows = store._conn.execute(
                "SELECT * FROM atlas_maps "
                "WHERE (deleted IS NULL OR deleted = 0) "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Regions CRUD
# ---------------------------------------------------------------------------


def create_region(
    store,
    map_id: int,
    name: str,
    description: str = "",
    tags: str = "",
    parent_region_id: Optional[int] = None,
) -> int:
    """Create a region within a map.

    Args:
        map_id: Parent map ID.
        name: Region name.
        description: Optional description.
        tags: Comma-separated tags.
        parent_region_id: Optional parent region for hierarchy.

    Returns:
        The new region_id.
    """
    if not name or not name.strip():
        raise ValueError("Region name must not be empty")
    with store._lock:
        cur = store._conn.execute(
            """INSERT INTO atlas_regions
               (map_id, parent_region_id, name, description, tags)
               VALUES (?, ?, ?, ?, ?)""",
            (map_id, parent_region_id, name.strip(), description, tags),
        )
        store._conn.commit()
        return cur.lastrowid


def get_region(store, region_id: int) -> Optional[dict]:
    """Get a single region by ID (excluding soft-deleted).

    Args:
        region_id: The region ID.

    Returns:
        Region dict or None.
    """
    with store._lock:
        row = store._conn.execute(
            "SELECT * FROM atlas_regions WHERE region_id = ? AND (deleted IS NULL OR deleted = 0)",
            (region_id,),
        ).fetchone()
    return dict(row) if row else None


def list_regions(store, map_id: int) -> list[dict]:
    """List all non-deleted regions for a map.

    Args:
        map_id: The parent map ID.

    Returns:
        List of region dicts.
    """
    with store._lock:
        rows = store._conn.execute(
            "SELECT * FROM atlas_regions "
            "WHERE map_id = ? AND (deleted IS NULL OR deleted = 0) "
            "ORDER BY region_id",
            (map_id,),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Edges CRUD
# ---------------------------------------------------------------------------


def link_fact(
    store,
    map_id: int,
    fact_id: int,
    region_id: Optional[int] = None,
    relation_type: str = "contains",
    weight: float = 0.5,
) -> int:
    """Link a fact to a map (optionally to a specific region).

    Args:
        map_id: The map ID.
        fact_id: The fact ID (from the facts table).
        region_id: Optional target region within the map.
        relation_type: Edge type label (default: 'contains').
        weight: Edge weight 0.0–1.0 (default: 0.5).

    Returns:
        The new edge_id.
    """
    source_id = region_id if region_id is not None else fact_id
    source_type = "region" if region_id is not None else "fact"
    with store._lock:
        cur = store._conn.execute(
            """INSERT OR IGNORE INTO atlas_edges
               (map_id, source_type, source_id, target_type, target_id,
                relation_type, weight)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (map_id, source_type, source_id, "fact", fact_id,
             relation_type, weight),
        )
        store._conn.commit()
        return cur.lastrowid or 0


def link_region(
    store,
    map_id: int,
    source_region_id: int,
    target_region_id: int,
    relation_type: str = "adjacent",
    weight: float = 0.5,
) -> int:
    """Create an edge between two regions within a map.

    Args:
        map_id: The map ID.
        source_region_id: Source region ID.
        target_region_id: Target region ID.
        relation_type: Edge type label (default: 'adjacent').
        weight: Edge weight 0.0–1.0 (default: 0.5).

    Returns:
        The new edge_id.
    """
    with store._lock:
        cur = store._conn.execute(
            """INSERT OR IGNORE INTO atlas_edges
               (map_id, source_type, source_id, target_type, target_id,
                relation_type, weight)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (map_id, "region", source_region_id, "region", target_region_id,
             relation_type, weight),
        )
        store._conn.commit()
        return cur.lastrowid or 0


def unlink(store, edge_id: int) -> bool:
    """Remove an edge by ID.

    Args:
        edge_id: The edge ID to remove.

    Returns:
        True if the edge was found and removed.
    """
    with store._lock:
        cur = store._conn.execute(
            "DELETE FROM atlas_edges WHERE edge_id = ?",
            (edge_id,),
        )
        store._conn.commit()
        return cur.rowcount > 0


def get_edges(
    store,
    node_type: str = "",
    node_id: int = 0,
) -> list[dict]:
    """Get edges, optionally filtered by source node.

    Args:
        node_type: If non-empty, filter by source_type.
        node_id: If non-zero, filter by source_id (only when node_type set).

    Returns:
        List of edge dicts.
    """
    with store._lock:
        if node_type and node_id:
            rows = store._conn.execute(
                "SELECT * FROM atlas_edges "
                "WHERE source_type = ? AND source_id = ? "
                "ORDER BY edge_id",
                (node_type, node_id),
            ).fetchall()
        elif node_type:
            rows = store._conn.execute(
                "SELECT * FROM atlas_edges WHERE source_type = ? ORDER BY edge_id",
                (node_type,),
            ).fetchall()
        else:
            rows = store._conn.execute(
                "SELECT * FROM atlas_edges ORDER BY edge_id"
            ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_map(
    store,
    query: str,
    limit: int = 20,
    project: str = "",
    scope: str = "canonical",
) -> list[dict]:
    """Full-text search across maps.

    Searches the ``atlas_maps_fts`` index. Returns summary info
    (name, description snippet, tags) — full content via ``get_map_node``.

    Args:
        query: FTS5 search query (plain text, auto-sanitized).
        limit: Max results (default 20).
        project: If non-empty, filter by project.
        scope: Ignored for atlas (kept for API compatibility).

    Returns:
        List of map summary dicts with keys: map_id, name, description, tags, score.
    """
    if not query or not query.strip():
        return []

    # Sanitize query like _sanitize_fts5 does
    import re
    safe = re.sub(r"""[?!'".;:\-+=~`@#$%^&*()\[\]{}|,<>]""", " ", query)
    safe = " ".join(safe.split())
    if not safe:
        return []

    with store._lock:
        if project:
            rows = store._conn.execute(
                """SELECT m.map_id, m.name, m.description, m.tags, m.project,
                          fts.rank AS score
                   FROM atlas_maps m
                   JOIN atlas_maps_fts fts ON fts.rowid = m.map_id
                   WHERE atlas_maps_fts MATCH ?
                     AND (m.deleted IS NULL OR m.deleted = 0)
                     AND m.project = ?
                   ORDER BY score
                   LIMIT ?""",
                (safe, project, limit),
            ).fetchall()
        else:
            rows = store._conn.execute(
                """SELECT m.map_id, m.name, m.description, m.tags, m.project,
                          fts.rank AS score
                   FROM atlas_maps m
                   JOIN atlas_maps_fts fts ON fts.rowid = m.map_id
                   WHERE atlas_maps_fts MATCH ?
                     AND (m.deleted IS NULL OR m.deleted = 0)
                   ORDER BY score
                   LIMIT ?""",
                (safe, limit),
            ).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        # Summary: first 200 chars of description
        desc = d.get("description", "") or ""
        d["summary"] = desc[:200]
        results.append(d)
    return results


def get_map_node(store, node_type: str, node_id: int) -> Optional[dict]:
    """Get the full record for any atlas node type.

    Args:
        node_type: One of ``"map"``, ``"region"``.
        node_id: The node ID.

    Returns:
        Full node dict or None.
    """
    with store._lock:
        if node_type == "map":
            row = store._conn.execute(
                "SELECT * FROM atlas_maps WHERE map_id = ? AND (deleted IS NULL OR deleted = 0)",
                (node_id,),
            ).fetchone()
        elif node_type == "region":
            row = store._conn.execute(
                "SELECT * FROM atlas_regions WHERE region_id = ? AND (deleted IS NULL OR deleted = 0)",
                (node_id,),
            ).fetchone()
        else:
            return None
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def traverse_path(
    store,
    start_map_id: int,
    end_map_id: int,
    max_depth: int = 5,
) -> list[dict]:
    """Find a path between two maps via recursive CTE on edges.

    Uses BFS-like recursive traversal on ``atlas_edges`` where both
    source and target are maps.

    Args:
        start_map_id: Starting map ID.
        end_map_id: Target map ID.
        max_depth: Maximum recursion depth (default: 5).

    Returns:
        List of node dicts on the path, or empty list if unreachable.
    """
    with store._lock:
        rows = store._conn.execute(
            """WITH RECURSIVE path AS (
                SELECT edge_id, source_type, source_id, target_type, target_id,
                       1 AS depth, CAST(source_id AS TEXT) AS path_str
                FROM atlas_edges
                WHERE source_type = 'region'
                  AND source_id = ?
                UNION ALL
                SELECT e.edge_id, e.source_type, e.source_id,
                       e.target_type, e.target_id,
                       p.depth + 1,
                       p.path_str || ',' || CAST(e.source_id AS TEXT)
                FROM atlas_edges e
                JOIN path p ON e.source_type = p.target_type
                           AND e.source_id = p.target_id
                WHERE p.depth < ?
                  AND instr(',' || p.path_str || ',',
                            ',' || CAST(e.source_id AS TEXT) || ',') = 0
            )
            SELECT edge_id, source_type, source_id, target_type, target_id, depth
            FROM path
            WHERE target_type = 'region' AND target_id = ?
            ORDER BY depth
            LIMIT 1""",
            (start_map_id, max_depth, end_map_id),
        ).fetchall()

    if not rows:
        # Fallback: try direct map-to-map via region links (search all edges)
        with store._lock:
            rows = store._conn.execute(
                """WITH RECURSIVE path AS (
                    SELECT edge_id, source_type, source_id,
                           target_type, target_id,
                           1 AS depth,
                           CAST(source_id AS TEXT) AS path_str
                    FROM atlas_edges
                    WHERE source_id = ?
                      AND (source_type = 'region' OR source_type = 'map')
                    UNION ALL
                    SELECT e.edge_id, e.source_type, e.source_id,
                           e.target_type, e.target_id,
                           p.depth + 1,
                           p.path_str || ',' || CAST(e.source_id AS TEXT)
                    FROM atlas_edges e
                    JOIN path p ON e.source_id = p.target_id
                    WHERE p.depth < ?
                      AND instr(',' || p.path_str || ',',
                                ',' || CAST(e.source_id AS TEXT) || ',') = 0
                )
                SELECT edge_id, source_type, source_id,
                       target_type, target_id, depth
                FROM path
                WHERE target_id = ?
                ORDER BY depth
                LIMIT 1""",
                (start_map_id, max_depth, end_map_id),
            ).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["node_type"] = d.pop("target_type", "region")
        d["node_id"] = d.pop("target_id", 0)
        results.append(d)
    return results


def map_stats(store, map_id: int) -> dict:
    """Get statistics for a map.

    Args:
        map_id: The map ID.

    Returns:
        Dict with ``region_count``, ``edge_count``, ``fact_count``.
    """
    with store._lock:
        region_count = store._conn.execute(
            "SELECT COUNT(*) FROM atlas_regions "
            "WHERE map_id = ? AND (deleted IS NULL OR deleted = 0)",
            (map_id,),
        ).fetchone()[0]
        edge_count = store._conn.execute(
            "SELECT COUNT(*) FROM atlas_edges WHERE map_id = ?",
            (map_id,),
        ).fetchone()[0]
        fact_count = store._conn.execute(
            "SELECT COUNT(*) FROM atlas_edges "
            "WHERE map_id = ? AND target_type = 'fact'",
            (map_id,),
        ).fetchone()[0]
    return {
        "map_id": map_id,
        "region_count": region_count,
        "edge_count": edge_count,
        "fact_count": fact_count,
    }
