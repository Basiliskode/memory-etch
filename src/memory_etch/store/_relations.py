"""Relation management — add, judge, query, navigate the relation graph.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
import sqlite3
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


def add_relation(
    store,
    fact_id_a: int,
    fact_id_b: int,
    relation_type: str = "related",
    confidence: float = 0.5,
    judged_by: str = "auto",
) -> bool:
    """Record a relation between two facts.

    Args:
        fact_id_a: First fact ID.
        fact_id_b: Second fact ID.
        relation_type: One of ``related``, ``compatible``, ``scoped``,
            ``conflicts_with``, ``supersedes``, ``not_conflict``.
        confidence: Confidence score for the relation (default: 0.5).
        judged_by: Who or what judged the relation (default: "auto").

    Returns:
        True if the relation was inserted, False if it already exists.
    """
    with store._lock:
        try:
            cur = store._conn.execute(
                """INSERT OR IGNORE INTO fact_relations
                   (fact_id_a, fact_id_b, relation_type, confidence, judged_by)
                   VALUES (?, ?, ?, ?, ?)""",
                (fact_id_a, fact_id_b, relation_type, confidence, judged_by),
            )
            store._conn.commit()
            if cur.rowcount > 0:
                store._log_event("relation_added",
                                metadata={"fact_id_a": fact_id_a, "fact_id_b": fact_id_b,
                                          "relation_type": relation_type, "confidence": confidence,
                                          "judged_by": judged_by})
            return True
        except sqlite3.IntegrityError:
            return False


def judge_relation(
    store,
    fact_id_a: int,
    fact_id_b: int,
    relation_type: str = "related",
    confidence: float = 0.5,
    judged_by: str = "auto",
) -> dict:
    """Alias for ``add_relation`` — returns enriched dict.

    If a relation already exists between the two facts, it is UPDATEd
    and ``updated`` is set to ``True``.

    Args:
        fact_id_a: First fact ID.
        fact_id_b: Second fact ID.
        relation_type: Relation type (default: "related").
        confidence: Confidence score (default: 0.5).
        judged_by: Who or what judged (default: "auto").

    Returns:
        Dict with keys ``relation_type``, ``confidence``, ``updated``,
        and ``relation_id``.

    Raises:
        ValueError: If *relation_type* is invalid.
        KeyError: If either fact does not exist.
    """
    if relation_type not in ("related", "compatible", "scoped", "conflicts_with", "supersedes", "not_conflict", "derived_from"):
        raise ValueError(f"Invalid relation_type: {relation_type}")
    # Verify facts exist
    for fid in (fact_id_a, fact_id_b):
        row = store._conn.execute(
            "SELECT 1 FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        if not row:
            raise KeyError(f"fact_id {fid} not found")

    with store._lock:
        # Check if relation already exists
        existing = store._conn.execute(
            "SELECT relation_id FROM fact_relations WHERE fact_id_a = ? AND fact_id_b = ?",
            (fact_id_a, fact_id_b),
        ).fetchone()

        if existing:
            # Update existing
            store._conn.execute(
                """UPDATE fact_relations SET relation_type = ?, confidence = ?, judged_by = ?
                   WHERE fact_id_a = ? AND fact_id_b = ?""",
                (relation_type, confidence, judged_by, fact_id_a, fact_id_b),
            )
            store._conn.commit()
            store._log_event("relation_added",
                            metadata={"fact_id_a": fact_id_a, "fact_id_b": fact_id_b,
                                      "relation_type": relation_type, "confidence": confidence,
                                      "judged_by": judged_by})
            return {
                "relation_type": relation_type,
                "confidence": confidence,
                "updated": True,
                "relation_id": existing["relation_id"],
            }
        else:
            # Insert new
            cur = store._conn.execute(
                """INSERT OR IGNORE INTO fact_relations
                   (fact_id_a, fact_id_b, relation_type, confidence, judged_by)
                   VALUES (?, ?, ?, ?, ?)""",
                (fact_id_a, fact_id_b, relation_type, confidence, judged_by),
            )
            store._conn.commit()
            if cur.rowcount > 0:
                store._log_event("relation_added",
                                metadata={"fact_id_a": fact_id_a, "fact_id_b": fact_id_b,
                                          "relation_type": relation_type, "confidence": confidence,
                                          "judged_by": judged_by})
            row = store._conn.execute(
                "SELECT relation_id FROM fact_relations WHERE fact_id_a = ? AND fact_id_b = ?",
                (fact_id_a, fact_id_b),
            ).fetchone()
            return {
                "relation_type": relation_type,
                "confidence": confidence,
                "updated": False,
                "relation_id": row["relation_id"] if row else 0,
            }


def get_relations(store, fact_id: int) -> list[dict]:
    """Get all relations for a fact.

    Args:
        fact_id: Fact ID to look up.

    Returns:
        List of relation dicts with the other fact as ``other_fact_id``.
    """
    with store._lock:
        rows = store._conn.execute(
            """SELECT r.relation_id,
                          CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END AS other_fact_id,
                          r.relation_type, r.confidence, r.judged_by, r.created_at
                   FROM fact_relations r
                   WHERE r.fact_id_a = ? OR r.fact_id_b = ?
                   ORDER BY r.created_at DESC""",
            (fact_id, fact_id, fact_id),
        ).fetchall()
    return [dict(r) for r in rows]


def get_contradictions(store, limit: int = 10) -> list[dict]:
    """Get known contradictions between facts.

    Args:
        limit: Max number of contradictions to return (default: 10).

    Returns:
        List of relation dicts with ``content_a`` and ``content_b``.
    """
    with store._lock:
        rows = store._conn.execute(
            """SELECT r.*, a.content as content_a, b.content as content_b
               FROM fact_relations r
               JOIN facts a ON a.fact_id = r.fact_id_a
               JOIN facts b ON b.fact_id = r.fact_id_b
               WHERE r.relation_type IN ('conflicts_with', 'supersedes')
               ORDER BY r.confidence DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_neighbors(
    store,
    fact_id: int,
    relation_type: str | None = None,
    min_confidence: float = 0.0,
    limit: int = 50,
) -> list[dict]:
    """Get direct connections to a fact through relations.

    Args:
        fact_id: Fact ID to look up.
        relation_type: Optional relation type filter (None = all types).
        min_confidence: Minimum confidence threshold (default: 0.0).
        limit: Max results to return (default: 50).

    Returns:
        List of neighbor dicts with ``other_fact_id``, ``content``,
        ``relation_type``, ``confidence``, and ``direction``
        (``"outgoing"`` when the given fact is ``fact_id_a``,
        ``"incoming"`` when it is ``fact_id_b``).
    """
    with store._lock:
        rows = store._conn.execute(
            """SELECT CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END AS other_fact_id,
                       f.content,
                       r.relation_type,
                       r.confidence,
                       CASE WHEN r.fact_id_a = ? THEN 'outgoing' ELSE 'incoming' END AS direction
                FROM fact_relations r
                JOIN facts f ON f.fact_id = CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END
                WHERE (r.fact_id_a = ? OR r.fact_id_b = ?)
                  AND r.confidence >= ?
                  AND (? IS NULL OR r.relation_type = ?)
                ORDER BY r.confidence DESC
                LIMIT ?""",
            (fact_id, fact_id, fact_id, fact_id, fact_id,
             min_confidence, relation_type, relation_type, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def find_path(
    store,
    start_id: int,
    end_id: int,
    max_depth: int = 5,
    relation_types: list[str] | None = None,
) -> list[dict] | None:
    """BFS shortest path between two facts.

    Uses breadth-first search to find the shortest sequence of relations
    connecting *start_id* to *end_id*.  Cycles are avoided via an in-memory
    visited set.

    Args:
        start_id: Starting fact ID.
        end_id: Target fact ID.
        max_depth: Maximum search depth (default: 5).  Search stops when
            this depth is reached without finding a path.
        relation_types: Optional list of relation types to traverse
            (e.g. ``["related", "compatible"]``).  ``None`` traverses all
            types.

    Returns:
        List of edges forming the path (ordered start → end), or ``None``
        if no path exists within ``max_depth``.  Each edge has ``from_id``,
        ``to_id``, ``from_content``, ``to_content``, ``relation_type``,
        ``confidence``, and ``depth``.
    """
    if start_id == end_id:
        return []

    # BFS: parent_map[child] = (parent, relation_type, confidence)
    parent_map: dict[int, tuple[int, str, float]] = {}
    visited: set[int] = {start_id}
    queue: deque[tuple[int, int]] = deque([(start_id, 0)])
    found = False

    while queue and not found:
        current_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        with store._lock:
            query = (
                """SELECT CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END AS other_fact_id,
                           r.relation_type, r.confidence
                    FROM fact_relations r
                    WHERE (r.fact_id_a = ? OR r.fact_id_b = ?)"""
            )
            params: list = [current_id, current_id, current_id]

            if relation_types:
                placeholders = ",".join("?" * len(relation_types))
                query += f" AND r.relation_type IN ({placeholders})"
                params.extend(relation_types)

            rows = store._conn.execute(query, params).fetchall()

        for row in rows:
            other_id = row["other_fact_id"]
            if other_id not in visited:
                visited.add(other_id)
                parent_map[other_id] = (
                    current_id,
                    row["relation_type"],
                    row["confidence"],
                )
                if other_id == end_id:
                    found = True
                    break
                queue.append((other_id, depth + 1))

    if not found:
        return None

    # Reconstruct path backward from end_id to start_id
    edges: list[dict] = []
    node = end_id
    while node != start_id:
        parent, rel_type, conf = parent_map[node]
        edges.append({
            "from_id": parent,
            "to_id": node,
            "relation_type": rel_type,
            "confidence": conf,
        })
        node = parent

    edges.reverse()

    # Assign depth (1-based)
    for i, edge in enumerate(edges):
        edge["depth"] = i + 1

    # Bulk-fetch content for all nodes in the path
    all_ids: set[int] = {start_id, end_id}
    for e in edges:
        all_ids.add(e["from_id"])
        all_ids.add(e["to_id"])

    with store._lock:
        id_placeholders = ",".join("?" * len(all_ids))
        content_rows = store._conn.execute(
            f"SELECT fact_id, content FROM facts WHERE fact_id IN ({id_placeholders})",
            list(all_ids),
        ).fetchall()
        content_map = {r["fact_id"]: r["content"] for r in content_rows}

    for e in edges:
        e["from_content"] = content_map.get(e["from_id"], "")
        e["to_content"] = content_map.get(e["to_id"], "")

    return edges


def get_ego_graph(
    store,
    fact_id: int,
    depth: int = 2,
    relation_types: list[str] | None = None,
    min_confidence: float = 0.0,
) -> dict:
    """Ego network: central fact + its neighborhood up to *depth*.

    Explores the relation graph outward from *fact_id*, collecting all
    reachable nodes within ``depth`` hops.  Returns the central node,
    all discovered nodes (deduplicated), and ALL edges among the node
    set (not just edges connected to ego).

    Args:
        fact_id: Central (ego) fact ID.
        depth: Neighborhood depth (default: 2).  0 means only the ego
            node is returned.
        relation_types: Optional list of relation types to include.
        min_confidence: Minimum confidence threshold (default: 0.0).

    Returns:
        Dict with keys:

        - ``ego`` — ``{fact_id, content, category, project}``
        - ``nodes`` — list of ``{fact_id, content, category, project,
          depth, relation_count}``
        - ``edges`` — list of ``{from_id, to_id, relation_type,
          confidence}``
    """
    # ---- BFS to discover reachable nodes ----
    node_set: set[int] = {fact_id}
    node_depth: dict[int, int] = {fact_id: 0}
    queue: deque[tuple[int, int]] = deque([(fact_id, 0)])

    while queue:
        current_id, current_depth = queue.popleft()
        if current_depth >= depth:
            continue

        with store._lock:
            query = (
                """SELECT CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END AS other_fact_id
                    FROM fact_relations r
                    WHERE (r.fact_id_a = ? OR r.fact_id_b = ?)
                      AND r.confidence >= ?"""
            )
            params: list = [current_id, current_id, current_id, min_confidence]

            if relation_types:
                placeholders = ",".join("?" * len(relation_types))
                query += f" AND r.relation_type IN ({placeholders})"
                params.extend(relation_types)

            rows = store._conn.execute(query, params).fetchall()

        for row in rows:
            other_id = row["other_fact_id"]
            if other_id not in node_set:
                node_set.add(other_id)
                node_depth[other_id] = current_depth + 1
                queue.append((other_id, current_depth + 1))

    # ---- Fetch node details ----
    with store._lock:
        id_placeholders = ",".join("?" * len(node_set))
        fact_rows = store._conn.execute(
            f"SELECT fact_id, content, category, project FROM facts WHERE fact_id IN ({id_placeholders})",
            list(node_set),
        ).fetchall()
        fact_map: dict[int, dict] = {r["fact_id"]: dict(r) for r in fact_rows}

    # ---- Count total relations per node (full graph) ----
    with store._lock:
        id_placeholders = ",".join("?" * len(node_set))
        count_rows = store._conn.execute(
            f"""SELECT nid AS fact_id, COUNT(*) AS relation_count
                FROM (
                    SELECT fact_id_a AS nid FROM fact_relations
                    UNION ALL
                    SELECT fact_id_b AS nid FROM fact_relations
                )
                WHERE nid IN ({id_placeholders})
                GROUP BY nid""",
            list(node_set),
        ).fetchall()
        relation_count: dict[int, int] = {r["fact_id"]: r["relation_count"] for r in count_rows}

    # ---- Build nodes list ----
    nodes = []
    for nid in sorted(node_set):
        f = fact_map.get(nid, {})
        nodes.append({
            "fact_id": nid,
            "content": f.get("content", ""),
            "category": f.get("category", ""),
            "project": f.get("project", ""),
            "depth": node_depth.get(nid, 0),
            "relation_count": relation_count.get(nid, 0),
        })

    # ---- Collect ALL edges among the node set ----
    with store._lock:
        id_placeholders = ",".join("?" * len(node_set))
        edge_rows = store._conn.execute(
            f"""SELECT r.fact_id_a AS from_id, r.fact_id_b AS to_id,
                       r.relation_type, r.confidence
                FROM fact_relations r
                WHERE r.fact_id_a IN ({id_placeholders})
                  AND r.fact_id_b IN ({id_placeholders})
                  AND r.confidence >= ?""",
            list(node_set) * 2 + [min_confidence],
        ).fetchall()

    edges = [dict(r) for r in edge_rows]

    # ---- Ego node ----
    ego_fact = fact_map.get(fact_id, {})
    ego = {
        "fact_id": fact_id,
        "content": ego_fact.get("content", ""),
        "category": ego_fact.get("category", ""),
        "project": ego_fact.get("project", ""),
    }

    return {
        "ego": ego,
        "nodes": nodes,
        "edges": edges,
    }


def get_subgraph(
    store,
    fact_ids: list[int],
    relation_types: list[str] | None = None,
) -> dict:
    """Return all nodes and edges among a given set of fact IDs.

    Only includes relations where BOTH ends are in *fact_ids*.

    Args:
        fact_ids: List of fact IDs to include in the subgraph.
        relation_types: Optional list of relation types to include.

    Returns:
        Dict with ``nodes`` (list of ``{fact_id, content, category,
        project}``) and ``edges`` (list of ``{from_id, to_id,
        relation_type, confidence}``).
    """
    if not fact_ids:
        return {"nodes": [], "edges": []}

    with store._lock:
        id_placeholders = ",".join("?" * len(fact_ids))

        node_rows = store._conn.execute(
            f"SELECT fact_id, content, category, project FROM facts WHERE fact_id IN ({id_placeholders})",
            list(fact_ids),
        ).fetchall()
        nodes = [dict(r) for r in node_rows]

        edge_query = (
            f"""SELECT r.fact_id_a AS from_id, r.fact_id_b AS to_id,
                       r.relation_type, r.confidence
                FROM fact_relations r
                WHERE r.fact_id_a IN ({id_placeholders})
                  AND r.fact_id_b IN ({id_placeholders})"""
        )
        params = list(fact_ids) * 2

        if relation_types:
            rt_placeholders = ",".join("?" * len(relation_types))
            edge_query += f" AND r.relation_type IN ({rt_placeholders})"
            params.extend(relation_types)

        edge_rows = store._conn.execute(edge_query, params).fetchall()
        edges = [dict(r) for r in edge_rows]

    return {"nodes": nodes, "edges": edges}


def get_graph_stats(store) -> dict:
    """Statistics about the entire relation graph.

    Returns:
        Dict with:

        - ``total_nodes`` — number of distinct facts with at least one
          relation
        - ``total_edges`` — number of entries in ``fact_relations``
        - ``relation_type_distribution`` — ``{"related": N, ...}``
        - ``most_connected`` — top 10 facts by relation count
        - ``average_degree`` — average relations per connected node
        - ``density`` — actual edges / possible edges (undirected)
    """
    with store._lock:
        # Nodes with at least one relation
        total_nodes_row = store._conn.execute(
            """SELECT COUNT(DISTINCT nid) AS cnt FROM (
                SELECT fact_id_a AS nid FROM fact_relations
                UNION
                SELECT fact_id_b AS nid FROM fact_relations
            )""",
        ).fetchone()
        total_nodes = total_nodes_row["cnt"] if total_nodes_row else 0

        # Total edges
        total_edges_row = store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM fact_relations",
        ).fetchone()
        total_edges = total_edges_row["cnt"] if total_edges_row else 0

        # Distribution by type
        dist_rows = store._conn.execute(
            "SELECT relation_type, COUNT(*) AS cnt FROM fact_relations GROUP BY relation_type",
        ).fetchall()
        relation_type_distribution = {r["relation_type"]: r["cnt"] for r in dist_rows}

        # Top 10 most connected nodes
        most_connected_rows = store._conn.execute(
            """SELECT nid AS fact_id, f.content, COUNT(*) AS relation_count
                FROM (
                    SELECT fact_id_a AS nid FROM fact_relations
                    UNION ALL
                    SELECT fact_id_b AS nid FROM fact_relations
                ) AS all_rels
                JOIN facts f ON f.fact_id = all_rels.nid
                GROUP BY nid
                ORDER BY relation_count DESC
                LIMIT 10""",
        ).fetchall()
        most_connected = [dict(r) for r in most_connected_rows]

        # Average degree (undirected: each edge contributes 2 to degree sum)
        average_degree = 0.0
        if total_nodes > 0:
            average_degree = (total_edges * 2) / total_nodes

        # Density (actual / possible undirected edges)
        density = 0.0
        if total_nodes > 1:
            possible_edges = total_nodes * (total_nodes - 1) / 2
            density = total_edges / possible_edges

    return {
        "total_nodes": total_nodes,
        "total_edges": total_edges,
        "relation_type_distribution": relation_type_distribution,
        "most_connected": most_connected,
        "average_degree": round(average_degree, 4),
        "density": round(density, 6),
    }
