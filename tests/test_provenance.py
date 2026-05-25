"""Tests for provenance and derivation tracking on EtchStore.

Covers derived_from relation, get_provenance, get_derivation_tree,
and consolidation integration.
"""

import json
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
def store_with_derivations(store: EtchStore):
    """Pre-built derivation chain for testing.

    Structure:
        f1 (root) ──derived_from──> f2 ──derived_from──> f3
        f1 ──derived_from──> f4 (branch)
    """
    f1 = store.add_fact("Original research finding", project="provenance")
    f2 = store.add_fact("Expanded finding with details", project="provenance")
    f3 = store.add_fact("Final polished finding", project="provenance")
    f4 = store.add_fact("Alternative interpretation", project="provenance")

    store._add_derivation_link(f1, f2, judged_by="test")
    store._add_derivation_link(f2, f3, judged_by="test")
    store._add_derivation_link(f1, f4, judged_by="test")

    return store, (f1, f2, f3, f4)


# =========================================================================
# TestDerivedFromRelation
# =========================================================================

class TestDerivedFromRelation:

    def test_can_add_derived_from_relation(self, store):
        f1 = store.add_fact("Source fact")
        f2 = store.add_fact("Derived fact")
        result = store._add_derivation_link(f1, f2, judged_by="test")
        assert result is True

        # Verify it's stored as a relation
        relations = store.get_relations(f2)
        assert len(relations) >= 1
        rel_types = {r["relation_type"] for r in relations}
        assert "derived_from" in rel_types

    def test_derived_from_in_valid_relation_types(self, store):
        f1 = store.add_fact("Fact A")
        f2 = store.add_fact("Fact B")
        # Should not raise ValueError
        result = store.judge_relation(f1, f2, relation_type="derived_from", judged_by="test")
        assert result["relation_type"] == "derived_from"

    def test_derived_from_appears_in_get_relations(self, store):
        f1 = store.add_fact("Source")
        f2 = store.add_fact("Derived")
        store._add_derivation_link(f1, f2, judged_by="test")
        relations = store.get_relations(f1)
        rel_types = {r["relation_type"] for r in relations}
        assert "derived_from" in rel_types

    def test_derived_from_shown_in_graph_stats(self, store):
        f1 = store.add_fact("Source")
        f2 = store.add_fact("Derived")
        store._add_derivation_link(f1, f2, judged_by="test")
        stats = store.get_graph_stats()
        dist = stats["relation_type_distribution"]
        assert dist.get("derived_from") == 1

    def test_invalid_relation_type_still_raises(self, store):
        f1 = store.add_fact("Fact A")
        f2 = store.add_fact("Fact B")
        with pytest.raises(ValueError, match="Invalid relation_type"):
            store.judge_relation(f1, f2, relation_type="invalid_type")


# =========================================================================
# TestGetProvenance
# =========================================================================

class TestGetProvenance:

    def test_returns_fact_metadata_with_provenance_fields(self, store):
        fid = store.add_fact(
            "Test fact with provenance",
            project="test_project",
            source_harness="test_harness",
            source_agent="test_agent",
            source_kind="test_kind",
            scope="canonical",
            session_id="session_001",
        )
        prov = store.get_provenance(fid)
        assert prov["fact_id"] == fid
        assert prov["content"] == "Test fact with provenance"
        assert prov["provenance"]["source_harness"] == "test_harness"
        assert prov["provenance"]["source_agent"] == "test_agent"
        assert prov["provenance"]["source_kind"] == "test_kind"
        assert prov["provenance"]["scope"] == "canonical"
        assert prov["provenance"]["session_id"] == "session_001"
        assert prov["provenance"]["project"] == "test_project"

    def test_returns_empty_ancestors_when_no_derivation(self, store):
        fid = store.add_fact("Isolated fact")
        prov = store.get_provenance(fid)
        assert prov["derivation_ancestors"] == []

    def test_returns_ancestors_when_derivation_chain_exists(self, store_with_derivations):
        store, (f1, f2, f3, f4) = store_with_derivations
        # f3 was derived from f2, which was derived from f1
        prov = store.get_provenance(f3)
        ancestors = prov["derivation_ancestors"]
        # Should have f2 (depth 1) and f1 (depth 2)
        ancestor_ids = {a["fact_id"] for a in ancestors}
        assert f2 in ancestor_ids
        assert f1 in ancestor_ids
        # Verify depth ordering
        depth_map = {a["fact_id"]: a["depth"] for a in ancestors}
        assert depth_map[f2] == 1
        assert depth_map[f1] == 2

    def test_returns_event_log_entries(self, store):
        fid = store.add_fact("Event log test fact")
        # Do an update to generate an event
        store.update_fact(fid, content="Updated event log test fact")
        prov = store.get_provenance(fid)
        assert len(prov["event_log"]) >= 1
        event_types = {e["event_type"] for e in prov["event_log"]}
        assert "fact_added" in event_types

    def test_returns_session_info_when_set(self, store):
        store.start_session("prov_session_001", project="test_project")
        fid = store.add_fact(
            "Session fact",
            project="test_project",
            session_id="prov_session_001",
        )
        prov = store.get_provenance(fid)
        assert "session" in prov
        assert prov["session"]["session_id"] == "prov_session_001"
        # fact_count is a session-level counter updated separately; it may be 0 initially
        assert prov["session"]["started_at"] is not None

    def test_derived_facts_count(self, store_with_derivations):
        store, (f1, f2, f3, f4) = store_with_derivations
        # f1 has two derivations (f2, f4)
        prov = store.get_provenance(f1)
        assert prov["derived_facts_count"] == 2

        # f2 has one derivation (f3)
        prov = store.get_provenance(f2)
        assert prov["derived_facts_count"] == 1

        # f3 has no derivations
        prov = store.get_provenance(f3)
        assert prov["derived_facts_count"] == 0


# =========================================================================
# TestGetDerivationTree
# =========================================================================

class TestGetDerivationTree:

    def test_single_level_derivation(self, store):
        f1 = store.add_fact("Root fact")
        f2 = store.add_fact("Child fact")
        store._add_derivation_link(f1, f2, judged_by="test")
        tree = store.get_derivation_tree(f1)
        assert tree["root"]["fact_id"] == f1
        assert len(tree["derivations"]) == 1
        assert tree["derivations"][0]["fact_id"] == f2
        assert tree["derivations"][0]["depth"] == 1

    def test_multi_level_chain(self, store_with_derivations):
        store, (f1, f2, f3, f4) = store_with_derivations
        # Tree from f1 should show: f2 (depth 1, with child f3), f4 (depth 1)
        tree = store.get_derivation_tree(f1)
        assert tree["root"]["fact_id"] == f1
        assert len(tree["derivations"]) == 2

        # Find f2 and f4
        der_map = {d["fact_id"]: d for d in tree["derivations"]}
        assert f2 in der_map
        assert f4 in der_map
        assert der_map[f2]["depth"] == 1
        assert der_map[f4]["depth"] == 1

        # f2 should have f3 as a child
        assert len(der_map[f2]["children"]) == 1
        assert der_map[f2]["children"][0]["fact_id"] == f3
        assert der_map[f2]["children"][0]["depth"] == 2

    def test_max_depth_respected(self, store_with_derivations):
        store, (f1, f2, f3, f4) = store_with_derivations
        # max_depth=1 should only show f2 and f4 (no children of f2)
        tree = store.get_derivation_tree(f1, max_depth=1)
        der_map = {d["fact_id"]: d for d in tree["derivations"]}
        assert f2 in der_map
        assert f4 in der_map
        # f2's children should be empty because max_depth=1 stops at depth 1
        assert der_map[f2]["children"] == []

    def test_fact_with_no_derivations(self, store):
        fid = store.add_fact("Lonely fact")
        tree = store.get_derivation_tree(fid)
        assert tree["root"]["fact_id"] == fid
        assert tree["derivations"] == []

    def test_branching_derivations(self, store):
        f1 = store.add_fact("Root")
        f2 = store.add_fact("Branch A")
        f3 = store.add_fact("Branch B")
        f4 = store.add_fact("Branch C")
        store._add_derivation_link(f1, f2, judged_by="test")
        store._add_derivation_link(f1, f3, judged_by="test")
        store._add_derivation_link(f1, f4, judged_by="test")
        tree = store.get_derivation_tree(f1)
        assert len(tree["derivations"]) == 3
        der_ids = {d["fact_id"] for d in tree["derivations"]}
        assert der_ids == {f2, f3, f4}


# =========================================================================
# TestConsolidationIntegration
# =========================================================================

class TestConsolidationIntegration:

    def test_replace_path_auto_links_derived_from(self, store):
        """REPLACE consolidation should create a derived_from link."""
        existing_fid = store.add_fact(
            "Original content about topic",
            project="test_consolidation",
        )

        # Mock search and LLM decision to force REPLACE
        def mock_search(query, limit=3):
            return [{"fact_id": existing_fid, "content": "Original content about topic"}]

        def mock_llm(new_content, existing):
            return {"action": "REPLACE", "reason": "replacing"}

        result = store.add_fact_with_consolidation(
            "New improved content about topic",
            project="test_consolidation",
            search_fn=mock_search,
            llm_decide_fn=mock_llm,
        )
        assert result["action"] == "merged", f"Expected merged, got {result}"
        new_fid = result["fact_id"]
        assert new_fid is not None

        # Verify derived_from link exists
        relations = store.get_relations(new_fid)
        rel_types = {r["relation_type"] for r in relations}
        assert "derived_from" in rel_types, (
            f"Expected derived_from relation on new fact {new_fid}, "
            f"got relations: {relations}"
        )

    def test_derived_from_relation_exists_after_consolidation_replace(self, store):
        """After REPLACE, the new fact should show the old one as ancestor."""
        existing_fid = store.add_fact(
            "Old version about topic",
            project="test_consolidation_v2",
        )

        def mock_search(query, limit=3):
            return [{"fact_id": existing_fid, "content": "Old version about topic"}]

        def mock_llm(new_content, existing):
            return {"action": "REPLACE", "reason": "new version"}

        result = store.add_fact_with_consolidation(
            "New version about topic that replaces",
            project="test_consolidation_v2",
            search_fn=mock_search,
            llm_decide_fn=mock_llm,
        )
        assert result["action"] == "merged", f"Expected merged, got {result}"
        new_fid = result["fact_id"]

        # Check provenance of new fact
        prov = store.get_provenance(new_fid)
        ancestor_ids = {a["fact_id"] for a in prov["derivation_ancestors"]}
        assert existing_fid in ancestor_ids, (
            f"Expected existing_fid {existing_fid} in derivation_ancestors "
            f"of new_fid {new_fid}, got {ancestor_ids}"
        )

    def test_merge_path_does_not_add_derived_from(self, store):
        """MERGE consolidation should NOT create a derived_from link."""
        existing_fid = store.add_fact(
            "Original content about merge topic",
            project="test_merge",
        )

        def mock_search(query, limit=3):
            return [{"fact_id": existing_fid, "content": "Original content about merge topic"}]

        def mock_llm(new_content, existing):
            return {
                "action": "MERGE",
                "merged_content": "Merged original and new about merge topic",
                "reason": "merging",
            }

        result = store.add_fact_with_consolidation(
            "New content about merge topic",
            project="test_merge",
            search_fn=mock_search,
            llm_decide_fn=mock_llm,
        )
        assert result["action"] == "merged", f"Expected merged, got {result}"
        merged_fid = result["fact_id"]

        # The MERGE path updates in place, so merged_fid == existing_fid
        relations = store.get_relations(merged_fid)
        rel_types = {r["relation_type"] for r in relations}
        assert "derived_from" not in rel_types, (
            "MERGE should not add derived_from relation"
        )
