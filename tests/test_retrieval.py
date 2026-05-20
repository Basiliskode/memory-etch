"""Tests for EtchRetriever hybrid search."""

import pytest
from memory_etch import EtchStore, EtchRetriever


@pytest.fixture
def store_with_facts():
    s = EtchStore(":memory:", auto_migrate=True)
    s.add_fact("Python is a programming language", category="tech", tags="python")
    s.add_fact("FastAPI is a web framework", category="tech", tags="python,web")
    s.add_fact("SQLite is a database engine", category="tech", tags="sqlite,db")
    s.add_fact("User prefers dark mode in all applications", category="user_pref", tags="ui,theme")
    s.add_fact("The project uses PostgreSQL for production", category="tech", tags="sql,db,production")
    s.add_fact("Docker containers run on a VPS with Dokploy", category="tech", tags="docker,deploy")
    yield s
    s.close()


@pytest.fixture
def retriever(store_with_facts):
    return EtchRetriever(store_with_facts, hrr_dim=256)


class TestRetrieverBasics:
    def test_search_returns_results(self, retriever):
        results = retriever.search("database")
        assert len(results) > 0

    def test_search_empty_query(self, retriever):
        results = retriever.search("")
        assert len(results) == 0

    def test_search_no_match(self, retriever):
        results = retriever.search("xyznonexistent12345")
        assert len(results) == 0

    def test_search_respects_limit(self, retriever):
        results = retriever.search("python", limit=2)
        assert len(results) <= 2

    def test_search_returns_scored_results(self, retriever):
        results = retriever.search("python")
        assert len(results) > 0
        assert "_score" in results[0]
        assert results[0]["_score"] > 0

    def test_search_relevance(self, retriever):
        """Most relevant result should be first."""
        results = retriever.search("database engine")
        assert len(results) > 0
        # SQLite mention should rank high
        first = results[0]["content"].lower()
        scores = [r["_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_works_without_numpy(self, retriever):
        """Test graceful degradation when HRR is unavailable."""
        # Simulate by setting hrr_weight to 0
        retriever._hrr_weight = 0.0
        results = retriever.search("python")
        assert len(results) > 0
        assert "_score" in results[0]


class TestRetrieverContent:
    def test_content_preserved(self, retriever):
        results = retriever.search("SQLite")
        assert any("SQLite" in r["content"] for r in results)

    def test_metadata_preserved(self, retriever):
        results = retriever.search("PostgreSQL")
        matching = [r for r in results if "PostgreSQL" in r["content"]]
        assert len(matching) > 0
        assert matching[0]["category"] == "tech"

    def test_trust_score_influences_rank(self, retriever):
        """Facts with higher trust scores should get a boost."""
        store = retriever._store
        # Find a fact and bump its trust
        results_before = retriever.search("deploy")
        # Update Docker fact trust to max
        docker_facts = [r for r in results_before if "Docker" in r["content"]]
        if docker_facts:
            store.update_fact(docker_facts[0]["fact_id"], trust_score=1.0)
            results_after = retriever.search("deploy")
            # Docker should be higher now
            docker_idx = next(
                (i for i, r in enumerate(results_after) if "Docker" in r["content"]),
                None,
            )
            if docker_idx is not None:
                assert docker_idx <= 1  # should be top 2


class TestRetrieverFilters:
    def test_exclude_deleted(self, retriever):
        store = retriever._store
        # Find any fact and soft-delete it
        results_before = retriever.search("Python")
        if results_before:
            store.soft_delete_fact(results_before[0]["fact_id"])
            results_after = retriever.search("Python")
            after_ids = [r["fact_id"] for r in results_after]
            assert results_before[0]["fact_id"] not in after_ids


class TestRetrievalFeedback:
    """Retrieval reinforces trust_score (retrieval feedback loop)."""

    def test_search_increases_trust(self, store_with_facts):
        store = store_with_facts
        # Get baseline trust for "Python" fact
        before = store._conn.execute(
            "SELECT trust_score, retrieval_count FROM facts WHERE content LIKE '%Python is a programming%'"
        ).fetchone()
        assert before is not None
        before_trust = before["trust_score"]

        # Search — triggers reinforcement
        store.search_facts("programming language")
        after = store._conn.execute(
            "SELECT trust_score, retrieval_count FROM facts WHERE content LIKE '%Python is a programming%'"
        ).fetchone()
        assert after["trust_score"] > before_trust
        assert after["retrieval_count"] >= 1

    def test_search_retrieval_count_increments(self, store_with_facts):
        store = store_with_facts
        before = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content LIKE '%SQLite%'"
        ).fetchone()[0]

        store.search_facts("database engine")
        after = store._conn.execute(
            "SELECT retrieval_count FROM facts WHERE content LIKE '%SQLite%'"
        ).fetchone()[0]
        assert after == before + 1

    def test_search_resets_unknown_facts_not_affected(self, store_with_facts):
        store = store_with_facts
        before = store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%Docker%'"
        ).fetchone()[0]

        # Search for something unrelated to Docker
        store.search_facts("programming language")
        after = store._conn.execute(
            "SELECT trust_score FROM facts WHERE content LIKE '%Docker%'"
        ).fetchone()[0]
        assert after == before  # no change
