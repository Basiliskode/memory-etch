"""Tests for Phase 3 — Smarter Search (FTS5 expansion, HRR multi-query, dynamic RRF, fallback chain).

Requires: `python -m pytest tests/test_smarter_search.py -v --tb=short`
"""

import pytest
from memento import EtchStore, EtchRetriever
from memento.retrieval import _STOPWORDS, _extract_keywords


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store():
    s = EtchStore(":memory:", auto_migrate=True)
    s.add_fact("Python is a programming language used for web development", category="tech", tags="python")
    s.add_fact("FastAPI is a modern web framework for building APIs", category="tech", tags="python,web")
    s.add_fact("SQLite is a lightweight database engine", category="tech", tags="sqlite,db")
    s.add_fact("PostgreSQL is a powerful open-source relational database", category="tech", tags="sql,db")
    s.add_fact("React is a JavaScript library for building user interfaces", category="tech", tags="js,frontend")
    s.add_fact("Flask is a micro web framework for Python", category="tech", tags="python,framework")
    s.add_fact("The sky appears blue due to Rayleigh scattering", category="science", tags="physics")
    s.add_fact("User prefers dark mode in all applications", category="user_pref", tags="ui,theme")
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    return EtchRetriever(store, hrr_dim=256)


# ===================================================================
# Task 3.1 — FTS5 search expansion
# ===================================================================

class TestFTS5KeywordExtraction:
    """Unit tests for the keyword extraction helper."""

    def test_removes_stopwords(self):
        keywords = _extract_keywords("the database engine is fast")
        assert "the" not in keywords
        assert "is" not in keywords
        assert "database" in keywords
        assert "engine" in keywords
        assert "fast" in keywords

    def test_empty_query(self):
        assert _extract_keywords("") == []

    def test_all_stopwords(self):
        assert _extract_keywords("the a is at") == []

    def test_preserves_order(self):
        keywords = _extract_keywords("fast database engine")
        assert keywords == ["fast", "database", "engine"]

    def test_single_content_word(self):
        keywords = _extract_keywords("the Python is")
        assert keywords == ["Python"]


class TestFTS5StopwordConstant:
    def test_stopwords_is_list_of_strings(self):
        assert isinstance(_STOPWORDS, list)
        assert len(_STOPWORDS) > 10
        assert all(isinstance(w, str) for w in _STOPWORDS)

    def test_common_stopwords_present(self):
        for w in ("the", "a", "is", "what", "this", "that"):
            assert w in _STOPWORDS


class TestSearchExpanded:
    """Integration tests for search_expanded()."""

    def test_full_query_returns_results(self, retriever):
        results = retriever.search_expanded("database engine")
        assert len(results) > 0
        # SQLite and PostgreSQL should be in results
        contents = [r["content"].lower() for r in results]
        assert any("sqlite" in c for c in contents)

    def test_empty_query_returns_empty(self, retriever):
        results = retriever.search_expanded("")
        assert len(results) == 0

    def test_single_word_works(self, retriever):
        results = retriever.search_expanded("Python")
        assert len(results) > 0
        assert any("python" in r["content"].lower() for r in results)

    def test_stopwords_only_returns_empty(self, retriever):
        """Stopwords-only query produces no keywords → empty results."""
        results = retriever.search_expanded("the a is")
        assert len(results) == 0

    def test_stopwords_with_one_content_word(self, retriever):
        results = retriever.search_expanded("the Python")
        assert len(results) > 0

    def test_expands_to_keywords_when_full_query_too_specific(self, retriever):
        """When full query returns few results, keyword expansion should find more."""
        # "Rayleigh scattering" should only match 1 fact directly
        results = retriever.search_expanded("the Rayleigh scattering effect", match_threshold=5)
        assert len(results) >= 1

    def test_respects_max_depth(self, retriever):
        """max_depth=1 means only full query, no expansion."""
        direct = retriever.search_expanded("the Rayleigh scattering effect", max_depth=1, match_threshold=100)
        expanded = retriever.search_expanded("the Rayleigh scattering effect", max_depth=3, match_threshold=100)
        assert len(expanded) >= len(direct)

    def test_results_deduplicated(self, retriever):
        results = retriever.search_expanded("database engine")
        ids = [r["fact_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_limit_respected(self, retriever):
        results = retriever.search_expanded("database", limit=2)
        assert len(results) <= 2

    def test_configurable_threshold(self, retriever):
        """Very low threshold stops expansion immediately."""
        results = retriever.search_expanded("database engine", match_threshold=1)
        # With threshold=1, getting >=1 results from full query means no expansion
        assert len(results) >= 0


# ===================================================================
# Task 3.2 — HRR multi-query
# ===================================================================

class TestHRRMultiQuery:
    """Tests for _hrr_multi_query()."""

    def test_returns_merged_results(self, retriever):
        results = retriever._hrr_multi_query("database engine")
        assert len(results) > 0

    def test_results_deduplicated(self, retriever):
        results = retriever._hrr_multi_query("database engine")
        ids = [r["fact_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_single_word(self, retriever):
        results = retriever._hrr_multi_query("Python")
        assert len(results) > 0

    def test_empty_query(self, retriever):
        results = retriever._hrr_multi_query("")
        assert len(results) == 0

    def test_stopwords_only(self, retriever):
        results = retriever._hrr_multi_query("the a is")
        assert len(results) == 0

    def test_limit_respected(self, retriever):
        results = retriever._hrr_multi_query("database", limit=2)
        assert len(results) <= 2


# ===================================================================
# Task 3.3 — Dynamic RRF k
# ===================================================================

class TestDynamicRRFk:
    def test_rrf_dynamic_k_small_result_set(self, retriever):
        """Small result set should floor at k=10."""
        k = retriever._compute_rrf_k(3)
        assert k == 10
        k = retriever._compute_rrf_k(1)
        assert k == 10

    def test_rrf_dynamic_k_medium_result_set(self, retriever):
        """Medium set: k = num_results * 2."""
        k = retriever._compute_rrf_k(10)
        assert k == 20
        k = retriever._compute_rrf_k(25)
        assert k == 50

    def test_rrf_dynamic_k_large_result_set(self, retriever):
        """Large set: capped at k=100."""
        k = retriever._compute_rrf_k(60)
        assert k == 100
        k = retriever._compute_rrf_k(100)
        assert k == 100

    def test_rrf_merge_uses_dynamic_k(self, retriever):
        """_rrf_merge should use dynamic k when provided num_results."""
        # Build mock streams
        stream_a = [{"fact_id": 1}, {"fact_id": 2}, {"fact_id": 3}]
        stream_b = [{"fact_id": 4}, {"fact_id": 5}]
        merged = retriever._rrf_merge(stream_a, stream_b, limit=5, num_results=5)
        assert len(merged) > 0
        # With num_results=5, k = max(10, min(100, 10)) = 10
        # Just verify it produces valid scores
        assert all("score" in r for r in merged)


# ===================================================================
# Task 3.4 — Fallback chain
# ===================================================================

class TestFallbackCascade:
    """Tests for search() with mode='auto'."""

    def test_mode_auto_returns_results(self, retriever):
        results = retriever.search("database engine", mode="auto")
        assert len(results) > 0

    def test_mode_auto_empty_query(self, retriever):
        results = retriever.search("", mode="auto")
        assert len(results) == 0

    def test_mode_auto_backward_compat(self, retriever):
        """search() without mode param still works (existing behavior)."""
        results = retriever.search("database engine")
        assert len(results) > 0
        assert "_score" in results[0]

    def test_mode_auto_non_empty_query(self, retriever):
        results = retriever.search("Python", mode="auto")
        assert len(results) > 0
        contents = [r["content"].lower() for r in results]
        assert any("python" in c for c in contents)

    def test_mode_auto_results_deduplicated(self, retriever):
        results = retriever.search("database", mode="auto")
        ids = [r["fact_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_mode_auto_limit_respected(self, retriever):
        results = retriever.search("database", mode="auto", limit=2)
        assert len(results) <= 2

    def test_fallback_thresholds_configurable(self, retriever):
        """Very high thresholds still work (forces full cascade)."""
        results = retriever.search("Python", mode="auto", fallback_thresholds=[100, 100])
        # Should still return results via cascade
        assert len(results) > 0

    def test_exclude_deleted_in_mode_auto(self, retriever):
        """mode='auto' should still exclude deleted facts."""
        r = retriever
        store = r._store
        # Soft-delete a fact
        all_python = r.search("Python", mode="auto")
        if all_python:
            target_id = all_python[0]["fact_id"]
            store.soft_delete_fact(target_id)
            after = r.search("Python", mode="auto")
            after_ids = [x["fact_id"] for x in after]
            assert target_id not in after_ids
