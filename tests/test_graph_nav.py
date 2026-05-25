"""Tests for graph navigation methods on EtchStore.

Covers get_neighbors, find_path, get_ego_graph, get_subgraph,
and get_graph_stats.
"""

import pytest
from memento.store import EtchStore


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def store():
    """Fresh EtchStore in :memory: for each test."""
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


@pytest.fixture
def graph_store(store: EtchStore):
    """Pre-built graph for navigation tests.

    Structure:
        f1 --related--> f2 --compatible--> f3
        f1 --conflicts_with--> f4
        f2 --related--> f5
        f4 --supersedes--> f5 --scoped--> f6
        f1 --compatible--> f7 (low confidence)
    """
    f1 = store.add_fact("Alpha project uses PostgreSQL", project="alpha")
    f2 = store.add_fact("Beta project uses PostgreSQL", project="beta")
    f3 = store.add_fact("Gamma project uses MySQL", project="gamma")
    f4 = store.add_fact("Avoid SQLite in production", project="alpha")
    f5 = store.add_fact("Use connection pooling", project="general")
    f6 = store.add_fact("Pool size should be 20", project="general")
    f7 = store.add_fact("Low confidence suggestion", project="general")

    store.judge_relation(f1, f2, "related", confidence=0.9, judged_by="test")
    store.judge_relation(f2, f3, "compatible", confidence=0.8, judged_by="test")
    store.judge_relation(f1, f4, "conflicts_with", confidence=0.85, judged_by="test")
    store.judge_relation(f2, f5, "related", confidence=0.75, judged_by="test")
    store.judge_relation(f4, f5, "supersedes", confidence=0.7, judged_by="test")
    store.judge_relation(f5, f6, "scoped", confidence=0.6, judged_by="test")
    store.judge_relation(f1, f7, "compatible", confidence=0.3, judged_by="test")

    return store, (f1, f2, f3, f4, f5, f6, f7)


# =========================================================================
# TestGetNeighbors
# =========================================================================

class TestGetNeighbors:

    def test_basic_neighbors(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        nbrs = store.get_neighbors(f1)
        # f1 has relations to f2, f4, f7
        nbr_ids = {n["other_fact_id"] for n in nbrs}
        assert f2 in nbr_ids
        assert f4 in nbr_ids
        assert f7 in nbr_ids
        assert f3 not in nbr_ids  # not directly connected to f1
        assert len(nbrs) >= 3

    def test_neighbor_content(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        nbrs = store.get_neighbors(f1)
        nbr_map = {n["other_fact_id"]: n for n in nbrs}
        assert nbr_map[f2]["content"] == "Beta project uses PostgreSQL"
        assert nbr_map[f2]["relation_type"] == "related"
        assert nbr_map[f2]["confidence"] == 0.9

    def test_filter_by_type(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        related_nbrs = store.get_neighbors(f1, relation_type="related")
        assert len(related_nbrs) == 1
        assert related_nbrs[0]["other_fact_id"] == f2

        compatible_nbrs = store.get_neighbors(f1, relation_type="compatible")
        assert len(compatible_nbrs) == 1
        assert compatible_nbrs[0]["other_fact_id"] == f7

    def test_min_confidence(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        high_conf = store.get_neighbors(f1, min_confidence=0.8)
        high_ids = {n["other_fact_id"] for n in high_conf}
        assert f2 in high_ids   # confidence 0.9
        assert f4 in high_ids   # confidence 0.85
        assert f7 not in high_ids  # confidence 0.3

    def test_limit(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        limited = store.get_neighbors(f1, limit=2)
        assert len(limited) == 2

    def test_no_relations_returns_empty(self, store):
        fid = store.add_fact("Lonely fact")
        assert store.get_neighbors(fid) == []

    def test_direction_tracking(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # f1 is fact_id_a in relation with f2 (outgoing) and fact_id_b
        # in relation with f7 (incoming... wait)
        # Actually, in judge_relation(f1, f4), f1 is fact_id_a -> outgoing
        # In judge_relation(f1, f7), f1 is fact_id_a -> outgoing
        # All our judge_relation calls use f1 as fact_id_a
        nbrs = store.get_neighbors(f1)
        for n in nbrs:
            assert n["direction"] == "outgoing", (
                f"Expected outgoing for f1, got {n['direction']} "
                f"for fact {n['other_fact_id']}"
            )

        # For f2, look at incoming from f1 and outgoing to f3/f5
        nbrs_f2 = store.get_neighbors(f2)
        nbr_map = {n["other_fact_id"]: n for n in nbrs_f2}
        assert nbr_map[f1]["direction"] == "incoming"
        assert nbr_map[f3]["direction"] == "outgoing"
        assert nbr_map[f5]["direction"] == "outgoing"

    def test_non_existent_fact(self, store):
        """Non-existent fact returns empty list, no error."""
        assert store.get_neighbors(99999) == []

    def test_order_by_confidence_desc(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        nbrs = store.get_neighbors(f1)
        confs = [n["confidence"] for n in nbrs]
        assert confs == sorted(confs, reverse=True)


# =========================================================================
# TestFindPath
# =========================================================================

class TestFindPath:

    def test_direct_neighbor(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        path = store.find_path(f1, f2)
        assert path is not None
        assert len(path) == 1
        assert path[0]["from_id"] == f1
        assert path[0]["to_id"] == f2
        assert path[0]["relation_type"] == "related"
        assert path[0]["depth"] == 1
        assert "from_content" in path[0]
        assert "to_content" in path[0]

    def test_multi_hop_path(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # f1 -> f2 -> f3
        path = store.find_path(f1, f3)
        assert path is not None
        assert len(path) == 2
        assert path[0]["from_id"] == f1
        assert path[0]["to_id"] == f2
        assert path[1]["from_id"] == f2
        assert path[1]["to_id"] == f3
        assert path[0]["depth"] == 1
        assert path[1]["depth"] == 2
        assert path[0]["relation_type"] == "related"
        assert path[1]["relation_type"] == "compatible"

    def test_three_hop_path(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # f1 -> f2 -> f5 -> f6
        path = store.find_path(f1, f6)
        assert path is not None
        assert len(path) == 3
        assert path[0]["from_id"] == f1
        assert path[0]["to_id"] == f2
        assert path[1]["from_id"] == f2
        assert path[1]["to_id"] == f5
        assert path[2]["from_id"] == f5
        assert path[2]["to_id"] == f6

    def test_no_path_returns_none(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # f3 is connected, f7 has no path to f6... let's check
        # f1 -> f7 (compatible) but f7 has no outgoing to f6
        # There might be a path f1 -> f2 -> f5 -> f6... no that doesn't go through f7
        # Actually, f7 only connects to f1. Let's find a fact with no path.
        # New isolated fact:
        isolated = store.add_fact("Isolated fact")
        path = store.find_path(f1, isolated)
        assert path is None

    def test_max_depth_exceeded(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # f1 -> f2 -> f5 -> f6 requires depth 3
        path = store.find_path(f1, f6, max_depth=2)
        assert path is None

    def test_with_relation_types_filter(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # Only allow 'compatible' — f1 and f2 are 'related', so no path
        path = store.find_path(f1, f3, relation_types=["compatible"])
        assert path is None

        # Allow 'related' and 'compatible' — path exists
        path = store.find_path(f1, f3, relation_types=["related", "compatible"])
        assert path is not None
        assert len(path) == 2

    def test_self_loop_returns_empty(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        path = store.find_path(f1, f1)
        assert path == []

    def test_start_not_found(self, graph_store):
        store, _ = graph_store
        path = store.find_path(99999, 1)
        assert path is None

    def test_path_content_populated(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        path = store.find_path(f1, f3)
        assert path[0]["from_content"] == "Alpha project uses PostgreSQL"
        assert path[0]["to_content"] == "Beta project uses PostgreSQL"
        assert path[1]["from_content"] == "Beta project uses PostgreSQL"
        assert path[1]["to_content"] == "Gamma project uses MySQL"


# =========================================================================
# TestEgoGraph
# =========================================================================

class TestEgoGraph:

    def test_basic_ego_depth_0(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=0)
        assert eg["ego"]["fact_id"] == f1
        assert eg["ego"]["content"] == "Alpha project uses PostgreSQL"
        assert len(eg["nodes"]) == 1  # only ego
        assert eg["nodes"][0]["depth"] == 0
        assert eg["edges"] == []

    def test_depth_1_expansion(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=1)
        node_ids = {n["fact_id"] for n in eg["nodes"]}
        # f1, f2, f4, f7 (direct neighbors)
        assert f1 in node_ids
        assert f2 in node_ids
        assert f4 in node_ids
        assert f7 in node_ids
        assert f3 not in node_ids  # depth 2 from f1
        assert f5 not in node_ids  # depth 2 from f1

        # Verify depths
        depth_map = {n["fact_id"]: n["depth"] for n in eg["nodes"]}
        assert depth_map[f1] == 0
        assert depth_map[f2] == 1
        assert depth_map[f4] == 1
        assert depth_map[f7] == 1

    def test_depth_2_expansion_includes_further_nodes(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=2)
        node_ids = {n["fact_id"] for n in eg["nodes"]}
        assert f3 in node_ids  # f1 -> f2 -> f3
        assert f5 in node_ids  # f1 -> f2 -> f5 AND f1 -> f4 -> f5
        # f6 is depth 3 from f1 (f1->f2->f5->f6 or f1->f4->f5->f6)
        assert f6 not in node_ids

        # f3 should be depth 2: f1 -> f2 (1) -> f3 (2)
        assert f3 in node_ids
        depth_map = {n["fact_id"]: n["depth"] for n in eg["nodes"]}
        assert depth_map[f3] == 2

    def test_relation_count_populated(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=1)
        rc_map = {n["fact_id"]: n["relation_count"] for n in eg["nodes"]}
        # f1 has 3 relations (f2, f4, f7)
        assert rc_map[f1] >= 3
        # f2 has 3 relations (f1, f3, f5)
        assert rc_map[f2] >= 3
        # f4 has 2 relations (f1, f5)
        assert rc_map[f4] >= 2

    def test_nodes_deduplicated(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=2)
        node_ids = [n["fact_id"] for n in eg["nodes"]]
        assert len(node_ids) == len(set(node_ids)), "Nodes should be deduplicated"

    def test_edges_include_non_ego_connections(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=2)
        # Edges should include f2-f3 (compatible), f2-f5 (related), f4-f5 (supersedes)
        edge_pairs = {(e["from_id"], e["to_id"]) for e in eg["edges"]}
        assert (f2, f3) in edge_pairs or (f3, f2) in edge_pairs
        assert (f2, f5) in edge_pairs or (f5, f2) in edge_pairs
        assert (f4, f5) in edge_pairs or (f5, f4) in edge_pairs
        # f5->f6 should NOT be in edges (f6 not in depth 2 set)
        assert (f5, f6) not in edge_pairs
        assert (f6, f5) not in edge_pairs

    def test_ego_graph_with_filter(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=2, relation_types=["compatible"])
        # Only compatible edges: f1-f7, f2-f3
        # f1 -> f2 is "related", so f2 should NOT be reachable
        node_ids = {n["fact_id"] for n in eg["nodes"]}
        assert f1 in node_ids
        assert f2 not in node_ids, "f2 should not be reachable via compatible only"
        assert f7 in node_ids  # f1-f7 is compatible

    def test_ego_graph_min_confidence(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        eg = store.get_ego_graph(f1, depth=1, min_confidence=0.8)
        node_ids = {n["fact_id"] for n in eg["nodes"]}
        assert f2 in node_ids   # 0.9
        assert f4 in node_ids   # 0.85
        assert f7 not in node_ids  # 0.3

    def test_ego_graph_empty_for_isolated(self, store):
        fid = store.add_fact("Isolated node")
        eg = store.get_ego_graph(fid, depth=2)
        assert eg["ego"]["fact_id"] == fid
        assert len(eg["nodes"]) == 1
        assert eg["edges"] == []


# =========================================================================
# TestSubgraph
# =========================================================================

class TestSubgraph:

    def test_complete_subgraph(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        sg = store.get_subgraph([f1, f2, f4])
        node_ids = {n["fact_id"] for n in sg["nodes"]}
        assert node_ids == {f1, f2, f4}
        edge_pairs = {(e["from_id"], e["to_id"]) for e in sg["edges"]}
        assert (f1, f2) in edge_pairs
        assert (f1, f4) in edge_pairs
        # f2-f4 does NOT have a direct relation
        assert (f2, f4) not in edge_pairs
        assert (f4, f2) not in edge_pairs

    def test_empty_set(self, graph_store):
        store, _ = graph_store
        sg = store.get_subgraph([])
        assert sg == {"nodes": [], "edges": []}

    def test_nodes_without_edges(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # f3 and f6 have no direct relation... actually they do through f2/f5
        # Pick isolated fact:
        isolated = store.add_fact("Totally isolated")
        sg = store.get_subgraph([f1, isolated])
        node_ids = {n["fact_id"] for n in sg["nodes"]}
        assert f1 in node_ids
        assert isolated in node_ids
        assert sg["edges"] == []

    def test_partial_subgraph(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        # Subgraph of {f1, f2, f3} should have f1-f2 and f2-f3
        sg = store.get_subgraph([f1, f2, f3])
        edge_pairs = {(e["from_id"], e["to_id"]) for e in sg["edges"]}
        assert (f1, f2) in edge_pairs
        assert (f2, f3) in edge_pairs
        assert len(sg["edges"]) == 2

    def test_relation_type_filter(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        sg = store.get_subgraph([f1, f2, f3], relation_types=["compatible"])
        # f1-f2 is "related", not included
        edge_pairs = {(e["from_id"], e["to_id"]) for e in sg["edges"]}
        assert (f1, f2) not in edge_pairs
        assert (f2, f3) in edge_pairs  # compatible
        assert len(sg["edges"]) == 1

    def test_nodes_not_in_facts_table(self, graph_store):
        """Non-existent fact IDs still return nodes that do exist."""
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        sg = store.get_subgraph([f1, 99999])
        node_ids = {n["fact_id"] for n in sg["nodes"]}
        assert f1 in node_ids
        assert 99999 not in node_ids
        assert sg["edges"] == []


# =========================================================================
# TestGraphStats
# =========================================================================

class TestGraphStats:

    def test_empty_graph(self, store):
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["relation_type_distribution"] == {}
        assert stats["most_connected"] == []
        assert stats["average_degree"] == 0.0
        assert stats["density"] == 0.0

    def test_counts_match(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 7  # f1-f7 all connected
        assert stats["total_edges"] == 7

    def test_relation_type_distribution(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        stats = store.get_graph_stats()
        dist = stats["relation_type_distribution"]
        assert dist.get("related") == 2  # f1-f2, f2-f5
        assert dist.get("compatible") == 2  # f2-f3, f1-f7
        assert dist.get("conflicts_with") == 1  # f1-f4
        assert dist.get("supersedes") == 1  # f4-f5
        assert dist.get("scoped") == 1  # f5-f6

    def test_most_connected(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        stats = store.get_graph_stats()
        assert len(stats["most_connected"]) >= 1
        # f1 has 3 relations (f2, f4, f7)
        # f2 has 3 relations (f1, f3, f5)
        # f5 has 2 relations (f2, f4)... wait, f5 has f2 and f4 and f6 = 3
        # f1: f2, f4, f7 = 3
        # f2: f1, f3, f5 = 3
        # f5: f2, f4, f6 = 3
        top = stats["most_connected"][0]
        assert top["relation_count"] >= 3
        assert "content" in top

    def test_average_degree(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        stats = store.get_graph_stats()
        # 7 edges, 7 nodes -> degree_sum = 14, avg = 14/7 = 2.0
        assert stats["average_degree"] == pytest.approx(2.0, rel=0.01)

    def test_density(self, graph_store):
        store, (f1, f2, f3, f4, f5, f6, f7) = graph_store
        stats = store.get_graph_stats()
        # 7 nodes, 7 edges, possible edges = 7*6/2 = 21
        # density = 7/21 = 0.333333...
        assert stats["density"] == pytest.approx(7.0 / 21.0, rel=0.01)

    def test_stats_with_isolated_node(self, store):
        """Isolated nodes (no relations) don't count as graph nodes."""
        isolated = store.add_fact("Isolated fact")
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 0  # no relations
        assert stats["total_edges"] == 0

        # Add a relation involving that fact
        f2 = store.add_fact("Another fact")
        store.judge_relation(isolated, f2, "related", confidence=0.9, judged_by="test")
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1

    def test_stats_after_adding_relation(self, store):
        f_a = store.add_fact("Fact A")
        f_b = store.add_fact("Fact B")
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 0  # no relations yet

        store.judge_relation(f_a, f_b, "related", confidence=0.9, judged_by="test")
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["average_degree"] == pytest.approx(1.0, rel=0.01)
        assert stats["density"] == pytest.approx(1.0 / 1.0, rel=0.01)


# =========================================================================
# Edge Cases
# =========================================================================

class TestGraphEdgeCases:

    def test_get_neighbors_nonexistent_fact(self, store):
        assert store.get_neighbors(99999) == []

    def test_find_path_nonexistent_start(self, store):
        assert store.find_path(99999, 1) is None

    def test_get_ego_graph_nonexistent_fact(self, store):
        """Ego for a non-existent fact still returns ego node (entry point)."""
        eg = store.get_ego_graph(99999, depth=2)
        assert eg["ego"]["fact_id"] == 99999
        assert eg["ego"]["content"] == ""
        assert len(eg["nodes"]) == 1
        assert eg["nodes"][0]["fact_id"] == 99999
        assert eg["edges"] == []

    def test_graph_stats_empty(self, store):
        stats = store.get_graph_stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
