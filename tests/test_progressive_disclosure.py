"""Tests for progressive disclosure — summary in search, full in get_fact."""

from pathlib import Path

import pytest

from memory_etch.store import EtchStore


@pytest.fixture
def db_path(tmp_path):
    return Path(tmp_path) / "test_progressive.db"


@pytest.fixture
def store(db_path):
    s = EtchStore(str(db_path), auto_migrate=True)
    # Add a fact with content longer than 200 chars
    s.add_fact(
        "Python is a high-level, general-purpose programming language. "
        "Its design philosophy emphasizes code readability with the use of "
        "significant indentation. Python is dynamically typed and "
        "garbage-collected. It supports multiple programming paradigms, "
        "including structured, object-oriented, and functional programming. "
        "It is often described as a batteries-included language.",
        category="tech",
        tags="python",
        session_id="s1",
    )
    s.add_fact(
        "Short fact",
        category="general",
        session_id="s1",
    )
    yield s
    s.close()


class TestSearchReturnsSummary:
    """search_facts results include a summary field."""

    def test_search_contains_summary_key(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        assert "summary" in results[0]

    def test_summary_is_first_200_chars_of_content(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        full_content = results[0]["content"]
        expected_summary = full_content[:200]
        assert results[0]["summary"] == expected_summary

    def test_short_content_summary_equals_content(self, store):
        results = store.search_facts("Short fact")
        assert len(results) >= 1
        assert results[0]["summary"] == results[0]["content"]

    def test_search_still_returns_full_content(self, store):
        """Backward compat: content field is still present and full."""
        results = store.search_facts("Python")
        assert len(results) >= 1
        assert len(results[0]["content"]) > 200
        assert results[0]["content"].startswith("Python is a high-level")


class TestSearchViaSearchExpanded:
    """search_expanded also includes summary."""

    def test_search_expanded_contains_summary(self, store):
        from memory_etch.retrieval import EtchRetriever
        retriever = EtchRetriever(store)
        results = retriever.search_expanded("Python")
        assert len(results) >= 1
        assert "summary" in results[0]
        assert results[0]["summary"] == results[0]["content"][:200]


class TestGetFactFull:
    """get_fact_full is an alias for get_fact."""

    def test_get_fact_full_returns_same_as_get_fact(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        fid = results[0]["fact_id"]
        fact1 = store.get_fact(fid)
        fact2 = store.get_fact_full(fid)
        assert fact1 == fact2

    def test_get_fact_full_returns_full_content(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        fid = results[0]["fact_id"]
        fact = store.get_fact_full(fid)
        assert fact is not None
        assert len(fact["content"]) > 200

    def test_get_fact_full_nonexistent(self, store):
        fact = store.get_fact_full(99999)
        assert fact is None


class TestSearchHasRelevantFields:
    """search returns id, project, type fields."""

    def test_search_result_has_id_field(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        assert "fact_id" in results[0]

    def test_search_result_has_score_or_trust(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        assert "score" in results[0] or "trust_score" in results[0]

    def test_search_result_has_project(self, store):
        results = store.search_facts("Python")
        assert len(results) >= 1
        # project may be empty string
        assert "project" in results[0]
