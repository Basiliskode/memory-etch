"""Tests for EtchStore.query() — structured Query DSL.

Covers all query_dict keys, edge cases, and combinations.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from memory_etch import EtchStore


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def empty_store(tmp_path):
    """Fresh store with no facts."""
    store = EtchStore(str(tmp_path / "empty.db"))
    yield store
    store.close()


@pytest.fixture
def store_with_facts(tmp_path):
    """Store with diverse facts for structured query tests."""
    store = EtchStore(str(tmp_path / "facts.db"))
    # Register all fact type schemas first
    for ft in ("definition", "preference", "note", "reminder", "cache"):
        store.register_schema(ft)
    # ---- tech / canonical facts ----
    store.add_fact(
        "Python is a programming language",
        category="tech", tags="python,lang",
        trust_score=0.9, importance=0.8,
        project="docs", fact_type="definition",
        what="Python language", why="It is interpreted",
        where_text="docs.python.org", learned="dynamic typing",
    )
    store.add_fact(
        "FastAPI is a web framework",
        category="tech", tags="python,web",
        trust_score=0.7, importance=0.6,
        project="docs", fact_type="definition",
    )
    store.add_fact(
        "SQLite is a database engine",
        category="tech", tags="sqlite,db",
        trust_score=0.5, importance=0.4,
        project="docs", fact_type="definition",
    )
    store.add_fact(
        "User prefers dark mode",
        category="user_pref", tags="ui,theme",
        trust_score=0.4, importance=0.9,
        project="app", fact_type="preference",
        what="User preferences",
    )
    # ---- inbox / personal facts ----
    store.add_fact(
        "Inbox note about project setup",
        category="note", tags="setup",
        trust_score=0.3, importance=0.2,
        project="app", fact_type="note", scope="inbox",
    )
    store.add_fact(
        "Personal reminder to update deps",
        category="note", tags="deps,maintenance",
        trust_score=0.6, importance=0.3,
        project="app", fact_type="reminder", scope="personal",
        why="Keep dependencies current",
        learned="Use uv for speed",
    )
    # ---- ephemeral ----
    store.add_fact(
        "Temp cache entry",
        category="cache", tags="temp",
        trust_score=0.1, importance=0.1,
        project="app", fact_type="cache", scope="ephemeral",
    )
    # ---- different project ----
    store.add_fact(
        "React is a UI library",
        category="tech", tags="react,web",
        trust_score=0.8, importance=0.7,
        project="frontend", fact_type="definition",
        why="Component-based architecture",
    )
    yield store
    store.close()


# =============================================================================
# TestBasicQuery
# =============================================================================

class TestBasicQuery:
    """No filters — default behavior."""

    def test_empty_query_returns_all(self, store_with_facts):
        result = store_with_facts.query({})
        assert len(result["results"]) == 8
        assert result["total"] == 8

    def test_empty_store_returns_empty(self, empty_store):
        result = empty_store.query({})
        assert result["results"] == []
        assert result["total"] == 0

    def test_limit_respected(self, store_with_facts):
        result = store_with_facts.query({"limit": 3})
        assert len(result["results"]) == 3
        assert result["total"] == 8

    def test_offset_pagination(self, store_with_facts):
        first = store_with_facts.query({"limit": 2, "offset": 0})
        second = store_with_facts.query({"limit": 2, "offset": 2})
        assert len(first["results"]) == 2
        assert len(second["results"]) == 2
        # Different facts when offset changes
        first_ids = {r["fact_id"] for r in first["results"]}
        second_ids = {r["fact_id"] for r in second["results"]}
        assert first_ids.isdisjoint(second_ids)

    def test_total_count(self, store_with_facts):
        result = store_with_facts.query({"limit": 3, "offset": 0})
        assert result["total"] == 8  # without limit

    def test_limit_clamped_to_max(self, store_with_facts):
        """limit > 200 is silently clamped to 200."""
        result = store_with_facts.query({"limit": 999})
        assert len(result["results"]) == 8  # only 8 facts exist
        assert result["total"] == 8

    def test_negative_limit(self, store_with_facts):
        result = store_with_facts.query({"limit": -1})
        assert result["results"] == []
        assert result["total"] == 8


# =============================================================================
# TestSearchQuery
# =============================================================================

class TestSearchQuery:
    """FTS5 full-text search integration."""

    def test_search_filter(self, store_with_facts):
        result = store_with_facts.query({"search": "programming"})
        assert len(result["results"]) == 1
        assert result["results"][0]["fact_id"] == 1  # "Python is a programming language"

    def test_search_no_match(self, store_with_facts):
        result = store_with_facts.query({"search": "zzzznonexistent"})
        assert result["results"] == []
        assert result["total"] == 0

    def test_search_fact_type_combined(self, store_with_facts):
        """search + fact_type combined with AND."""
        result = store_with_facts.query({
            "search": "framework",
            "fact_type": "definition",
        })
        # "FastAPI is a web framework" is fact_type=definition
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert r["fact_type"] == "definition"

    def test_search_project_combined(self, store_with_facts):
        """search + project combined with AND."""
        result = store_with_facts.query({
            "search": "programming",
            "project": "docs",
        })
        assert len(result["results"]) == 1
        assert "Python" in result["results"][0]["content"]

    def test_search_fact_type_no_match(self, store_with_facts):
        """search + fact_type with no matching combo."""
        result = store_with_facts.query({
            "search": "programming",
            "fact_type": "preference",
        })
        assert result["results"] == []
        assert result["total"] == 0


# =============================================================================
# TestStructuredQuery
# =============================================================================

class TestStructuredQuery:
    """fact_type, category, project, scope, tags filters."""

    def test_type_filter(self, store_with_facts):
        result = store_with_facts.query({"fact_type": "definition"})
        assert len(result["results"]) == 4
        for r in result["results"]:
            assert r["fact_type"] == "definition"

    def test_type_filter_no_match(self, store_with_facts):
        result = store_with_facts.query({"fact_type": "nonexistent"})
        assert result["results"] == []

    def test_category_filter(self, store_with_facts):
        result = store_with_facts.query({"category": "tech"})
        assert len(result["results"]) == 4
        for r in result["results"]:
            assert r["category"] == "tech"

    def test_project_filter(self, store_with_facts):
        result = store_with_facts.query({"project": "frontend"})
        assert len(result["results"]) == 1
        assert result["results"][0]["project"] == "frontend"

    def test_scope_filter(self, store_with_facts):
        result = store_with_facts.query({"scope": "inbox"})
        assert len(result["results"]) == 1
        assert result["results"][0]["scope"] == "inbox"

    def test_scope_filter_canonical(self, store_with_facts):
        """canonical is the default scope for most facts."""
        result = store_with_facts.query({"scope": "canonical"})
        assert len(result["results"]) == 5  # 5 canonical facts in fixture

    def test_tags_filter(self, store_with_facts):
        result = store_with_facts.query({"tags": "python"})
        assert len(result["results"]) >= 2  # Python, FastAPI

    def test_tags_filter_case_sensitive(self, store_with_facts):
        """LIKE is case-insensitive in default SQLite collation."""
        result = store_with_facts.query({"tags": "PYTHON"})
        assert len(result["results"]) >= 2

    def test_tags_filter_no_match(self, store_with_facts):
        result = store_with_facts.query({"tags": "imaginary"})
        assert result["results"] == []

    def test_combined_filters(self, store_with_facts):
        """category + project + fact_type + tags combined."""
        result = store_with_facts.query({
            "category": "tech",
            "project": "docs",
            "fact_type": "definition",
            "tags": "python",
        })
        # Only "Python is a..." and "FastAPI is a..." match
        assert len(result["results"]) == 2
        for r in result["results"]:
            assert r["category"] == "tech"
            assert r["project"] == "docs"
            assert r["fact_type"] == "definition"
            assert "python" in r["tags"]


# =============================================================================
# TestTrustImportanceQuery
# =============================================================================

class TestTrustImportanceQuery:
    """Numeric range filters on trust_score and importance."""

    def test_min_trust(self, store_with_facts):
        result = store_with_facts.query({"min_trust": 0.8})
        for r in result["results"]:
            assert r["trust_score"] >= 0.8
        # Fact 1 (0.9) and Fact 8 (0.8) and maybe Fact ... let's see
        assert len(result["results"]) >= 2

    def test_max_trust(self, store_with_facts):
        result = store_with_facts.query({"max_trust": 0.3})
        for r in result["results"]:
            assert r["trust_score"] <= 0.3
        assert len(result["results"]) >= 1  # ephemeral 0.1, inbox 0.3

    def test_trust_range(self, store_with_facts):
        result = store_with_facts.query({"min_trust": 0.3, "max_trust": 0.6})
        for r in result["results"]:
            assert 0.3 <= r["trust_score"] <= 0.6

    def test_min_importance(self, store_with_facts):
        result = store_with_facts.query({"min_importance": 0.8})
        for r in result["results"]:
            assert r["importance"] >= 0.8
        # Python (0.8), User prefers dark mode (0.9)
        assert len(result["results"]) >= 2

    def test_max_importance(self, store_with_facts):
        result = store_with_facts.query({"max_importance": 0.2})
        for r in result["results"]:
            assert r["importance"] <= 0.2

    def test_importance_range(self, store_with_facts):
        result = store_with_facts.query({
            "min_importance": 0.3,
            "max_importance": 0.7,
        })
        for r in result["results"]:
            assert 0.3 <= r["importance"] <= 0.7


# =============================================================================
# TestStructuredFieldsQuery
# =============================================================================

class TestStructuredFieldsQuery:
    """has_what, has_why, has_where, has_learned filters."""

    def test_has_what(self, store_with_facts):
        result = store_with_facts.query({"has_what": True})
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert r["what"] is not None and r["what"] != ""

    def test_has_why(self, store_with_facts):
        result = store_with_facts.query({"has_why": True})
        assert len(result["results"]) >= 2  # Personal reminder, React
        for r in result["results"]:
            assert r["why"] is not None and r["why"] != ""

    def test_has_learned(self, store_with_facts):
        result = store_with_facts.query({"has_learned": True})
        assert len(result["results"]) >= 1  # Python fact
        for r in result["results"]:
            assert r["learned"] is not None and r["learned"] != ""

    def test_combined_has_filters(self, store_with_facts):
        """AND of has_what + has_why."""
        result = store_with_facts.query({"has_what": True, "has_why": True})
        # Python fact has both what and why
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert r["what"] is not None and r["what"] != ""
            assert r["why"] is not None and r["why"] != ""


# =============================================================================
# TestTimeQuery
# =============================================================================

class TestTimeQuery:
    """created_after / created_before datetime filters."""

    def test_created_after(self, store_with_facts):
        """Created after the first fact (fact_id 1 is earliest)."""
        # Get the first fact's created_at
        first = store_with_facts.get_fact(1)
        assert first is not None
        # Use a time far in the past to ensure we get all facts
        result = store_with_facts.query({
            "created_after": "2020-01-01T00:00:00",
        })
        # All 8 facts are created now (well after 2020)
        assert len(result["results"]) == 8

    def test_created_before(self, store_with_facts):
        """Created before a very recent timestamp should return all."""
        result = store_with_facts.query({
            "created_before": "2030-01-01T00:00:00",
        })
        assert len(result["results"]) == 8

    def test_created_range(self, store_with_facts):
        """Both after and before — should return all facts."""
        result = store_with_facts.query({
            "created_after": "2020-01-01T00:00:00",
            "created_before": "2030-01-01T00:00:00",
        })
        assert len(result["results"]) == 8

    def test_created_after_future(self, store_with_facts):
        """Future date returns no results."""
        result = store_with_facts.query({
            "created_after": "2099-01-01T00:00:00",
        })
        assert result["results"] == []
        assert result["total"] == 0

    def test_created_before_past(self, store_with_facts):
        """Very old date returns no results."""
        result = store_with_facts.query({
            "created_before": "2000-01-01T00:00:00",
        })
        assert result["results"] == []
        assert result["total"] == 0


# =============================================================================
# TestRelationQuery
# =============================================================================

class TestRelationQuery:
    """related_to filter with relation graph."""

    def test_related_to(self, store_with_facts):
        """Facts connected via relation."""
        store_with_facts.add_relation(1, 2, "compatible")
        store_with_facts.add_relation(1, 3, "related")
        result = store_with_facts.query({"related_to": 1})
        assert len(result["results"]) == 2
        found_ids = {r["fact_id"] for r in result["results"]}
        assert found_ids == {2, 3}
        assert 1 not in found_ids  # self excluded

    def test_related_to_no_relations(self, store_with_facts):
        """Isolated fact returns empty."""
        result = store_with_facts.query({"related_to": 5})
        assert result["results"] == []
        assert result["total"] == 0

    def test_related_to_with_type(self, store_with_facts):
        """Filter by relation_type."""
        store_with_facts.add_relation(1, 2, "compatible")
        store_with_facts.add_relation(1, 3, "related")
        result = store_with_facts.query({
            "related_to": 1,
            "relation_type": "compatible",
        })
        assert len(result["results"]) == 1
        assert result["results"][0]["fact_id"] == 2

    def test_related_to_with_direction_outgoing(self, store_with_facts):
        """Only outgoing relations where fact_id_a = related_to."""
        store_with_facts.add_relation(1, 2, "compatible")
        store_with_facts.add_relation(3, 1, "derived_from")  # incoming for 1
        result = store_with_facts.query({
            "related_to": 1,
            "relation_direction": "outgoing",
        })
        assert len(result["results"]) == 1
        assert result["results"][0]["fact_id"] == 2

    def test_related_to_with_direction_incoming(self, store_with_facts):
        """Only incoming relations where fact_id_b = related_to."""
        store_with_facts.add_relation(1, 2, "compatible")  # outgoing for 1
        store_with_facts.add_relation(3, 1, "derived_from")  # incoming for 1
        result = store_with_facts.query({
            "related_to": 1,
            "relation_direction": "incoming",
        })
        assert len(result["results"]) == 1
        assert result["results"][0]["fact_id"] == 3

    def test_related_to_self_excluded(self, store_with_facts):
        """The related_to fact itself is excluded from results."""
        store_with_facts.add_relation(1, 2, "compatible")
        result = store_with_facts.query({"related_to": 1})
        assert 1 not in {r["fact_id"] for r in result["results"]}


# =============================================================================
# TestOrderBy
# =============================================================================

class TestOrderBy:
    """Sorting behavior."""

    def test_order_by_trust_asc(self, store_with_facts):
        """Ordered by trust_score ascending."""
        result = store_with_facts.query({"order_by": "trust_score", "order_dir": "asc"})
        scores = [r["trust_score"] for r in result["results"]]
        assert scores == sorted(scores)

    def test_order_by_importance_desc(self, store_with_facts):
        """Ordered by importance descending (default)."""
        result = store_with_facts.query({"order_by": "importance", "order_dir": "desc"})
        scores = [r["importance"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_order_by_created_at_asc(self, store_with_facts):
        """Oldest first — default created_at asc ordering."""
        result = store_with_facts.query({
            "order_by": "created_at",
            "order_dir": "asc",
        })
        assert len(result["results"]) == 8  # no crash, returns all

    def test_order_by_invalid_falls_back_to_created_at(self, store_with_facts):
        """Invalid order_by silently falls back to created_at."""
        result = store_with_facts.query({"order_by": "invalid_field"})
        assert len(result["results"]) == 8  # no crash

    def test_order_dir_asc(self, store_with_facts):
        result = store_with_facts.query({"order_dir": "asc"})
        assert len(result["results"]) == 8


# =============================================================================
# TestQueryReturnFormat
# =============================================================================

class TestQueryReturnFormat:
    """Verifies the return dict structure."""

    def test_returns_dict_with_results_total_query(self, store_with_facts):
        result = store_with_facts.query({})
        assert isinstance(result, dict)
        assert "results" in result
        assert "total" in result
        assert "query" in result
        assert isinstance(result["results"], list)
        assert isinstance(result["total"], int)
        assert isinstance(result["query"], str)

    def test_query_summary_string(self, store_with_facts):
        """Query string includes the applied filters."""
        result = store_with_facts.query({
            "search": "Python",
            "fact_type": "definition",
            "project": "docs",
        })
        qs = result["query"]
        assert "search=Python" in qs
        assert "fact_type=definition" in qs
        assert "project=docs" in qs

    def test_query_summary_empty(self, store_with_facts):
        """Empty query_dict produces empty query string."""
        result = store_with_facts.query({})
        assert result["query"] == ""

    def test_query_summary_skips_defaults(self, store_with_facts):
        """Default relation_direction (any) not included."""
        result = store_with_facts.query({
            "related_to": 1,
        })
        qs = result["query"]
        assert "related_to=1" in qs
        assert "relation_direction" not in qs

    def test_total_ignores_limit(self, store_with_facts):
        """total reflects count WITHOUT limit/offset."""
        result = store_with_facts.query({"limit": 1, "offset": 0})
        assert len(result["results"]) == 1
        assert result["total"] == 8

    def test_total_ignores_offset(self, store_with_facts):
        result = store_with_facts.query({"limit": 1, "offset": 5})
        assert len(result["results"]) == 1
        assert result["total"] == 8

    def test_result_fields(self, store_with_facts):
        """Each result has the expected fields."""
        result = store_with_facts.query({"limit": 1})
        r = result["results"][0]
        expected_keys = {
            "fact_id", "content", "category", "tags", "trust_score",
            "importance", "project", "scope", "fact_type",
            "created_at", "updated_at", "what", "why", "where_text", "learned",
        }
        assert set(r.keys()) == expected_keys


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:
    """Boundary and edge-case scenarios."""

    def test_soft_deleted_excluded(self, tmp_path):
        """Soft-deleted facts are excluded by default."""
        store = EtchStore(str(tmp_path / "deleted.db"))
        store.add_fact("Keep me")
        fid = store.add_fact("Delete me")
        store.soft_delete_fact(fid, reason="test")
        result = store.query({})
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Keep me"
        store.close()

    def test_invalid_order_dir_falls_back_to_desc(self, store_with_facts):
        """Invalid order_dir defaults to desc — no crash, returns all."""
        result = store_with_facts.query({"order_dir": "invalid"})
        assert len(result["results"]) == 8

    def test_all_filters_combined(self, store_with_facts):
        """Many filters combined should still work."""
        result = store_with_facts.query({
            "search": "programming",
            "fact_type": "definition",
            "project": "docs",
            "category": "tech",
            "scope": "canonical",
            "min_trust": 0.5,
            "max_trust": 1.0,
            "min_importance": 0.0,
            "max_importance": 1.0,
            "has_why": True,
            "created_after": "2020-01-01T00:00:00",
            "created_before": "2030-01-01T00:00:00",
        })
        assert len(result["results"]) >= 1
        for r in result["results"]:
            assert r["fact_type"] == "definition"
            assert r["category"] == "tech"
            assert r["project"] == "docs"
            assert r["scope"] == "canonical"
            assert r["trust_score"] >= 0.5

    def test_query_does_not_mutate_store(self, store_with_facts):
        """Running query() has no side effects on the store."""
        before = store_with_facts.query({})
        after = store_with_facts.query({})
        assert before == after
