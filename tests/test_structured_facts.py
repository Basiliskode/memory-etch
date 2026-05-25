"""Tests for structured fact fields (what, why, where_text, learned)."""

import pytest
from memento import EtchStore


@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    yield s
    s.close()


class TestStructuredFactFields:
    def test_add_fact_with_structured_fields(self, store):
        """add_fact accepts optional what, why, where_text, learned."""
        fid = store.add_fact(
            "Added dark mode support",
            category="tech",
            what="Dark mode implementation",
            why="User requested better UX for night usage",
            where_text="src/components/ThemeToggle.tsx",
            learned="CSS variables make theme switching trivial",
        )
        assert fid > 0
        fact = store.get_fact(fid)
        assert fact["what"] == "Dark mode implementation"
        assert fact["why"] == "User requested better UX for night usage"
        assert fact["where_text"] == "src/components/ThemeToggle.tsx"
        assert fact["learned"] == "CSS variables make theme switching trivial"

    def test_add_fact_without_structured_fields_defaults_empty(self, store):
        """add_fact without structured fields stores them as empty strings."""
        fid = store.add_fact("Plain fact", category="general")
        fact = store.get_fact(fid)
        assert fact["what"] == ""
        assert fact["why"] == ""
        assert fact["where_text"] == ""
        assert fact["learned"] == ""

    def test_add_fact_partial_structured_fields(self, store):
        """Only provided structured fields are stored; others default to empty."""
        fid = store.add_fact("Partial fact", what="Only what")
        fact = store.get_fact(fid)
        assert fact["what"] == "Only what"
        assert fact["why"] == ""
        assert fact["where_text"] == ""
        assert fact["learned"] == ""

    def test_topic_upsert_preserves_structured_fields(self, store):
        """Topic upsert updates structured fields too."""
        fid1 = store.add_fact(
            "Original about project X",
            project="test",
            topic_key="topic:projx",
            what="Original what",
            why="Original why",
        )
        fid2 = store.add_fact(
            "Updated about project X",
            project="test",
            topic_key="topic:projx",
            what="Updated what",
            learned="New learning",
        )
        assert fid2 == fid1  # upsert
        fact = store.get_fact(fid2)
        assert fact["content"] == "Updated about project X"
        assert fact["what"] == "Updated what"
        assert fact["why"] == "Original why"  # preserved from original
        assert fact["learned"] == "New learning"


class TestSearchByMetadata:
    def test_search_by_what(self, store):
        store.add_fact("Something about auth", what="Login flow")
        store.add_fact("Something about CSS", what="Style variables")
        results = store.search_by_metadata(what="Login")
        assert len(results) == 1
        assert results[0]["what"] == "Login flow"

    def test_search_by_why(self, store):
        store.add_fact("Added feature", why="Performance issue")
        store.add_fact("Fixed bug", why="User complaint")
        results = store.search_by_metadata(why="Performance")
        assert len(results) == 1

    def test_search_by_where_text(self, store):
        store.add_fact("Changed stuff", where_text="src/core/main.py")
        results = store.search_by_metadata(where_text="main.py")
        assert len(results) == 1

    def test_search_by_learned(self, store):
        store.add_fact("Something", learned="Caching is hard")
        results = store.search_by_metadata(learned="Caching")
        assert len(results) == 1

    def test_search_multiple_fields_and(self, store):
        store.add_fact("Auth rewrite", what="Auth", why="Security issue", where_text="auth.py")
        store.add_fact("UI update", what="UI", why="Design refresh")
        results = store.search_by_metadata(what="Auth", why="Security")
        assert len(results) == 1

    def test_search_by_metadata_respects_limit(self, store):
        for i in range(5):
            store.add_fact(f"Fact {i}", what="Same what")
        results = store.search_by_metadata(what="Same", limit=3)
        assert len(results) == 3

    def test_search_by_metadata_empty_results(self, store):
        store.add_fact("Only fact", what="Missing")
        results = store.search_by_metadata(what="Nonexistent")
        assert len(results) == 0

    def test_search_by_metadata_all_none_returns_all(self, store):
        store.add_fact("Fact A")
        store.add_fact("Fact B")
        results = store.search_by_metadata(limit=10)
        assert len(results) >= 2

    def test_search_by_metadata_partial_match(self, store):
        store.add_fact("Fact", what="Database connection pool")
        results = store.search_by_metadata(what="connection")
        assert len(results) == 1

    def test_structured_fields_in_search_facts(self, store):
        """search_facts does NOT include structured fields by default (backward compat)."""
        store.add_fact("Some content", what="A what")
        results = store.search_facts("Some content")
        assert len(results) >= 1
        # search_facts returns what's in the SELECT, which doesn't include what/why/where_text/learned
        assert "what" not in results[0]
