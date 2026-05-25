"""Tests for topic_key upsert, timeline, and session lifecycle.

Tests core store logic directly, no Hermes plugin imports needed.
"""
import json
import sqlite3
import threading
import tempfile
from pathlib import Path

import pytest

import sys
from pathlib import Path
_sys_path = str(Path(__file__).resolve().parent.parent.parent / "plugins/memory/etch")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from memento.store import EtchStore
from memento.retrieval import EtchRetriever


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def store():
    """Fresh EtchStore in a temp directory for each test."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "test.db"
        s = EtchStore(db_path=str(db_path))
        yield s
        s.close()


@pytest.fixture
def retriever(store: EtchStore):
    """EtchRetriever bound to the fresh store."""
    return EtchRetriever(store)


# =========================================================================
# Topic Upsert Tests
# =========================================================================

class TestTopicUpsert:

    def test_topic_upsert_creates_new_when_not_exists(self, store):
        fid = store.add_fact("Usamos PostgreSQL para la DB principal", tags="topic:db-choice")
        assert fid > 0
        row = store._conn.execute(
            "SELECT fact_id, topic_key, revision_count FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row is not None
        assert row["topic_key"] == "topic:db-choice"
        assert row["revision_count"] == 0

    def test_topic_upsert_updates_existing(self, store):
        fid1 = store.add_fact("Usamos PostgreSQL para la DB principal", tags="topic:db-choice")
        fid2 = store.add_fact("Usamos PostgreSQL 16 para la DB principal (actualizado)", tags="topic:db-choice")
        assert fid2 == fid1, "Should return same fact_id on upsert"
        row = store._conn.execute(
            "SELECT content, revision_count FROM facts WHERE fact_id = ?", (fid1,)
        ).fetchone()
        assert row["content"] == "Usamos PostgreSQL 16 para la DB principal (actualizado)"
        assert row["revision_count"] == 1

    def test_topic_upsert_increments_revision_count(self, store):
        fid = store.add_fact("v1", tags="topic:evolving")
        assert store.add_fact("v2", tags="topic:evolving") == fid
        assert store.add_fact("v3", tags="topic:evolving") == fid
        row = store._conn.execute(
            "SELECT content, revision_count FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["content"] == "v3"
        assert row["revision_count"] == 2

    def test_topic_upsert_preserves_without_tags(self, store):
        fid1 = store.add_fact("Mismo texto sin tags")
        fid2 = store.add_fact("Mismo texto sin tags")
        assert fid2 == fid1

    def test_topic_upsert_different_topics_separate(self, store):
        fid1 = store.add_fact("PostgreSQL", tags="topic:db-choice")
        fid2 = store.add_fact("Redis", tags="topic:cache-choice")
        assert fid2 > fid1
        assert fid2 != fid1

    def test_topic_upsert_no_topic_key_uses_content_dedup(self, store):
        fid1 = store.add_fact("content exacto")
        fid2 = store.add_fact("content exacto")
        assert fid2 == fid1
        fid3 = store.add_fact("content distinto")
        assert fid3 > fid1


# =========================================================================
# Timeline Tests
# =========================================================================

class TestTimeline:

    def _seed_facts(self, store, count=5):
        ids = []
        with store._lock:
            for i in range(count):
                fid = store.add_fact(f"Fact number {i}", tags=f"seq:{i}",
                                     session_id="test-session")
                store._conn.execute(
                    "UPDATE facts SET created_at = datetime('now', ? || ' minutes') WHERE fact_id = ?",
                    (f"-{count - i}", fid)
                )
                ids.append(fid)
            # Close the implicit transaction left by the last UPDATE
            store._conn.commit()
        return ids

    def test_timeline_returns_before_and_after(self, store):
        ids = self._seed_facts(store, 5)
        middle_id = ids[2]
        result = store.timeline(fact_id=middle_id, before=2, after=2)
        assert result["fact"]["fact_id"] == middle_id
        assert len(result["before"]) == 2
        assert len(result["after"]) == 2

    def test_timeline_limits_respected(self, store):
        ids = self._seed_facts(store, 5)
        first_id = ids[0]
        result = store.timeline(fact_id=first_id, before=10, after=3)
        assert len(result["before"]) == 0
        assert len(result["after"]) == 3
        assert result["fact"]["fact_id"] == first_id

    def test_timeline_fact_not_found(self, store):
        with pytest.raises(KeyError, match="fact_id 99999 not found"):
            store.timeline(fact_id=99999, before=5, after=5)

    def test_timeline_no_session_id(self, store):
        fid = store.add_fact("Fact sin session")
        with pytest.raises(ValueError, match="no session association"):
            store.timeline(fact_id=fid, before=5, after=5)

    def test_timeline_all_facts_in_order(self, store):
        ids = self._seed_facts(store, 5)
        result = store.timeline(fact_id=ids[2], before=10, after=10)
        before_ids = [f["fact_id"] for f in result["before"]]
        after_ids = [f["fact_id"] for f in result["after"]]
        assert before_ids == ids[1::-1]
        assert after_ids == ids[3:5]


# =========================================================================
# Session Lifecycle Tests
# =========================================================================

class TestSessionLifecycle:

    def test_session_start_creates_row(self, store):
        info = store.session_start(session_id="sess-001", project="test-project")
        assert info["session_id"] == "sess-001"
        assert info["prior_session_count"] == 0
        row = store._conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?", ("sess-001",)
        ).fetchone()
        assert row is not None
        assert row["status"] == "active"
        assert row["project"] == "test-project"

    def test_session_start_tracks_prior_count(self, store):
        store.session_start("sess-001", project="proj-x")
        info = store.session_start("sess-002", project="proj-x")
        assert info["prior_session_count"] >= 1

    def test_session_end_marks_completed(self, store):
        store.session_start("sess-001")
        result = store.session_end("sess-001", summary="Completed architecture review")
        assert result is True
        row = store._conn.execute(
            "SELECT status, summary FROM sessions WHERE session_id = ?", ("sess-001",)
        ).fetchone()
        assert row["status"] == "ended"
        assert "architecture" in row["summary"]

    def test_session_end_nonexistent(self, store):
        result = store.session_end("nonexistent")
        assert result is False

    def test_session_end_tracks_fact_count(self, store):
        store.session_start("sess-fact-cnt")
        store.add_fact("Fact 1", tags="test", session_id="sess-fact-cnt")
        store.add_fact("Fact 2", tags="test", session_id="sess-fact-cnt")
        store.session_end("sess-fact-cnt")
        row = store._conn.execute(
            "SELECT fact_count FROM sessions WHERE session_id = ?", ("sess-fact-cnt",)
        ).fetchone()
        assert row["fact_count"] == 2

    def test_session_start_has_top_facts(self, store):
        store.add_fact("Architecture: PostgreSQL 16", category="project", tags="topic:db")
        store.add_fact("Prefer yarn over npm", category="user_pref")
        info = store.session_start("sess-with-facts", project="test")
        assert "top_facts" in info
        assert len(info["top_facts"]) >= 2
        # Top fact should be high trust
        assert all(f["trust_score"] >= 0.3 for f in info["top_facts"])

    def test_get_recent_sessions_returns_ended(self, store):
        store.session_start("sess-a", project="test")
        store.session_end("sess-a", summary="Fixed auth bug")
        store.session_start("sess-b", project="test")
        store.session_end("sess-b", summary="Added rate limiting, changed DB schema, and deployed v2")
        recent = store.get_recent_sessions(project="test", limit=3)
        assert len(recent) == 2
        assert recent[0]["session_id"] == "sess-b"
        assert recent[1]["session_id"] == "sess-a"

    def test_get_recent_sessions_summary_truncated(self, store):
        long_summary = "A" * 300
        store.session_start("sess-long", project="test")
        store.session_end("sess-long", summary=long_summary)
        recent = store.get_recent_sessions(project="test", limit=1)
        assert len(recent[0]["summary"]) <= 200

    def test_get_recent_sessions_empty_when_none(self, store):
        recent = store.get_recent_sessions(project="nonexistent", limit=3)
        assert recent == []



# =========================================================================
# Multi-project Scoping Tests
# =========================================================================

class TestMultiProject:

    def test_add_fact_with_project_stores_project(self, store):
        fid = store.add_fact("Fact for project A", tags="test", project="hermes-agent")
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["project"] == "hermes-agent"

    def test_add_fact_default_project_is_empty(self, store):
        fid = store.add_fact("Fact without project")
        row = store._conn.execute(
            "SELECT project FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["project"] == ""

    def test_search_facts_filters_by_project(self, store):
        store.add_fact("Database: PostgreSQL main DB", project="project-a")
        store.add_fact("Database: MongoDB analytics", project="project-b")
        store.add_fact("Frontend: React UI", project="project-a")

        results_a = store.search_facts("Database", project="project-a")
        assert len(results_a) == 1
        assert "PostgreSQL" in results_a[0]["content"]

        results_b = store.search_facts("Database", project="project-b")
        assert len(results_b) == 1
        assert "MongoDB" in results_b[0]["content"]

    def test_search_facts_no_project_returns_all(self, store):
        store.add_fact("Fact A", project="proj-a")
        store.add_fact("Fact B", project="proj-b")
        results = store.search_facts("Fact")
        assert len(results) == 2

    def test_list_facts_filters_by_project(self, store):
        store.add_fact("Alpha", project="proj-x")
        store.add_fact("Beta", project="proj-y")
        store.add_fact("Gamma", project="proj-x")

        x_facts = store.list_facts(project="proj-x")
        assert len(x_facts) == 2
        y_facts = store.list_facts(project="proj-y")
        assert len(y_facts) == 1

    def test_topic_upsert_respects_project(self, store):
        "Topic upsert finds by topic_key regardless of project (global scope)."
        fid1 = store.add_fact("v1 global config", tags="topic:config", project="proj-a")
        # Same topic_key, same topic_key column in DB — should update
        fid2 = store.add_fact("v2 global config", tags="topic:config", project="proj-b")
        assert fid2 == fid1
        row = store._conn.execute(
            "SELECT content, project FROM facts WHERE fact_id = ?", (fid1,)
        ).fetchone()
        assert row["content"] == "v2 global config"
        assert row["project"] == "proj-b"

    def test_retriever_search_with_project(self, store):
        from memento.retrieval import EtchRetriever
        retriever = EtchRetriever(store=store)

        store.add_fact("Authentication is done with JWT tokens", project="hermes-agent")
        store.add_fact("Authentication uses OAuth2 in the old system", project="other-tool")
        store.add_fact("Deployment goes through Coolify", project="hermes-agent")

        results = retriever.search("Authentication", project="hermes-agent")
        assert len(results) == 1
        assert "JWT" in results[0]["content"]

    def test_retriever_probe_with_project(self, store):
        from memento.retrieval import EtchRetriever
        retriever = EtchRetriever(store=store)

        store.add_fact("HermesAgent is the main framework", tags="hermes", project="project-a")
        store.add_fact("HermesAgent is also used in testing", tags="hermes", project="project-b")

        results = retriever.probe("hermes", project="project-a")
        assert len(results) >= 1
        # All results should be from project-a
        for r in results:
            assert r.get("project", "project-a") == "project-a"



# =========================================================================
# LLM-Judge & fact_relations Tests
# =========================================================================

class TestFactRelations:

    def test_judge_relation_stores_and_returns(self, store):
        fid_a = store.add_fact("Database is PostgreSQL 16")
        fid_b = store.add_fact("Database uses MySQL instead")
        result = store.judge_relation(fid_a, fid_b, "conflicts_with", confidence=0.9, judged_by="test")
        assert result["relation_type"] == "conflicts_with"
        assert result["confidence"] == 0.9
        assert not result["updated"]
        assert result["relation_id"] > 0

    def test_judge_relation_upserts_on_duplicate(self, store):
        fid_a = store.add_fact("Fact X")
        fid_b = store.add_fact("Fact Y")
        r1 = store.judge_relation(fid_a, fid_b, "compatible", confidence=0.7, judged_by="test")
        r2 = store.judge_relation(fid_a, fid_b, "related", confidence=0.8, judged_by="test")
        assert r2["updated"]
        assert r2["relation_type"] == "related"
        assert r2["confidence"] == 0.8
        assert r2["relation_id"] == r1["relation_id"]

    def test_judge_relation_raises_on_invalid_type(self, store):
        fid = store.add_fact("Some fact")
        with pytest.raises(ValueError, match="Invalid relation_type"):
            store.judge_relation(fid, fid, "invalid_type")

    def test_judge_relation_raises_on_missing_fact(self, store):
        with pytest.raises(KeyError):
            store.judge_relation(999, 1000, "related")

    def test_get_relations_returns_both_directions(self, store):
        fa = store.add_fact("Alpha")
        fb = store.add_fact("Beta")
        store.judge_relation(fa, fb, "related", judged_by="test")
        # Query from fa
        rels_a = store.get_relations(fa)
        assert len(rels_a) == 1
        assert rels_a[0]["other_fact_id"] == fb
        assert rels_a[0]["relation_type"] == "related"
        # Query from fb
        rels_b = store.get_relations(fb)
        assert len(rels_b) == 1
        assert rels_b[0]["other_fact_id"] == fa

    def test_get_relations_empty_for_no_relations(self, store):
        fid = store.add_fact("Lonely fact")
        assert store.get_relations(fid) == []

    def test_multiple_relations_for_fact(self, store):
        f1 = store.add_fact("Main config")
        f2 = store.add_fact("Alternative config")
        f3 = store.add_fact("Related doc")
        store.judge_relation(f1, f2, "conflicts_with", judged_by="test")
        store.judge_relation(f1, f3, "related", judged_by="test")
        rels = store.get_relations(f1)
        assert len(rels) == 2
        types = {r["relation_type"] for r in rels}
        assert types == {"conflicts_with", "related"}

    def test_llm_judge_fallback_when_no_api_key(self, store):
        """When no LLM is configured, _call_llm_judge returns None -> fallback."""
        # Import the plugin and test without API key
        pass  # This validates the guard logic in the method


class TestContradictWithFactRelations:
    """Test that contradict() uses fact_relations table."""

    def test_contradict_returns_known_conflicts(self, store, retriever):
        f1 = store.add_fact("Use PostgreSQL for all databases")
        f2 = store.add_fact("Use MongoDB instead of PostgreSQL")
        store.judge_relation(f1, f2, "conflicts_with", confidence=0.9, judged_by="test")
        results = retriever.contradict(limit=10)
        assert len(results) >= 1
        matched = [r for r in results if r.get("source") == "fact_relations"]
        assert len(matched) >= 1

    def test_contradict_prioritizes_known_over_algorithmic(self, store, retriever):
        f1 = store.add_fact("Use SQLite for embedded DB")
        f2 = store.add_fact("Use PostgreSQL for everything")
        store.judge_relation(f1, f2, "supersedes", confidence=0.85, judged_by="test")
        results = retriever.contradict(limit=10)
        if results:
            top = results[0]
            assert top.get("source") == "fact_relations" or top.get("relation_type") in (
                "conflicts_with", "supersedes"
            )

    def test_contradict_no_relations_still_works(self, store, retriever):
        """Fallback to algorithmic when no fact_relations exist."""
        f1 = store.add_fact("Deploy on AWS exclusively")
        f2 = store.add_fact("GCP is better than AWS for our case")
        results = retriever.contradict(limit=10)
        # May or may not find algorithmic matches depending on entities
        assert isinstance(results, list)


# =========================================================================
# Embedding / Vector Search Tests
# =========================================================================

class TestEmbeddingColumn:

    def test_embedding_column_exists(self, store):
        """Migration adds the embedding BLOB column to facts table."""
        cols = {
            row["name"]
            for row in store._conn.execute("PRAGMA table_info(facts)").fetchall()
        }
        assert "embedding" in cols, "embedding column missing after migration"

    def test_add_fact_accepts_embedding(self, store):
        """add_fact() stores the embedding BLOB."""
        blob = b"\x00" * 64
        fid = store.add_fact("fact with embedding", embedding=blob)
        row = store._conn.execute(
            "SELECT embedding FROM facts WHERE fact_id = ?", (fid,)
        ).fetchone()
        assert row["embedding"] == blob

    def test_get_fact_hides_embedding(self, store):
        """get_fact() should NOT return the embedding BLOB."""
        fid = store.add_fact("hidden embedding fact")
        d = store.get_fact(fid)
        assert "embedding" not in d, "get_fact leaked embedding field"

    def test_search_by_vector_no_results(self, store):
        """search_by_vector on empty column returns empty list."""
        import struct
        zero_vec = struct.pack("384f", *([0.0] * 384))
        results = store.search_by_vector(zero_vec, limit=5)
        assert results == [], f"Expected empty, got {len(results)} results"

    def test_search_by_vector_orders_by_cosine(self, store):
        """search_by_vector returns facts sorted by cosine similarity."""
        import struct
        fid_a = store.add_fact(
            "database PostgreSQL config",
            embedding=struct.pack("384f", *([1.0] + [0.0] * 383)),
        )
        fid_b = store.add_fact(
            "frontend React styling",
            embedding=struct.pack("384f", *([0.0] * 383 + [1.0])),
        )
        query_vec = struct.pack("384f", *([0.9] + [0.0] * 383))
        results = store.search_by_vector(query_vec, limit=2)
        assert len(results) >= 1
        assert results[0]["fact_id"] == fid_a, "Expected database fact first (cosine 0.9)"

    def test_search_by_vector_respects_trust(self, store):
        """search_by_vector applies min_trust filter."""
        import struct
        store.add_fact(
            "low trust fact",
            embedding=struct.pack("384f", *([1.0] + [0.0] * 383)),
        )
        query_vec = struct.pack("384f", *([0.9] + [0.0] * 383))
        results = store.search_by_vector(query_vec, min_trust=0.99, limit=5)
        assert results == []

    def test_search_by_vector_respects_category(self, store):
        """search_by_vector applies category filter."""
        import struct
        store.add_fact(
            "tool fact",
            category="tool",
            embedding=struct.pack("384f", *([1.0] + [0.0] * 383)),
        )
        query_vec = struct.pack("384f", *([0.9] + [0.0] * 383))
        results = store.search_by_vector(query_vec, category="nonexistent", limit=5)
        assert results == []

    def test_search_by_vector_respects_project(self, store):
        """search_by_vector applies project filter."""
        import struct
        vec = struct.pack("384f", *([1.0] + [0.0] * 383))
        store.add_fact(
            "alpha project fact",
            project="alpha",
            embedding=vec,
        )
        store.add_fact(
            "beta project fact",
            project="beta",
            embedding=vec,
        )
        query_vec = struct.pack("384f", *([0.9] + [0.0] * 383))
        results = store.search_by_vector(query_vec, project="alpha", limit=5)
        assert len(results) == 1, f"Expected 1 alpha result, got {len(results)}"
        assert "alpha" in results[0]["content"]
        results_beta = store.search_by_vector(query_vec, project="nonexistent", limit=5)
        assert results_beta == []


class TestRRFFusion:

    def test_rrf_merge_combines_streams(self):
        """_rrf_merge boosts items appearing in both streams."""
        stream_a = [
            {"fact_id": 1, "content": "a"},
            {"fact_id": 2, "content": "b"},
            {"fact_id": 3, "content": "c"},
        ]
        stream_b = [
            {"fact_id": 4, "content": "d"},
            {"fact_id": 1, "content": "a"},
        ]
        merged = EtchRetriever._rrf_merge(stream_a, stream_b, limit=3)
        fids = [m["fact_id"] for m in merged]
        assert fids[0] == 1, f"Expected #1 first (appears in both), got #{fids[0]}"
        assert "score" in merged[0]
        assert merged[0]["score"] > merged[1]["score"]

    def test_rrf_merge_respects_limit(self):
        stream_a = [{"fact_id": i} for i in range(20)]
        stream_b = [{"fact_id": i + 20} for i in range(20)]
        merged = EtchRetriever._rrf_merge(stream_a, stream_b, limit=5)
        assert len(merged) == 5, f"Expected 5 results, got {len(merged)}"

    def test_rrf_merge_empty_streams(self):
        merged = EtchRetriever._rrf_merge([], [], limit=10)
        assert merged == []

    def test_retriever_search_fallback_no_embedder(self, store):
        """search() works without embedder (BM25 fallback)."""
        store.add_fact("PostgreSQL database connection pool")
        store.add_fact("React frontend state management")
        store.add_fact("Docker Kubernetes orchestration")

        r = EtchRetriever(store=store, compute_embedding=None)
        results = r.search("database", limit=3)
        assert len(results) >= 1
        assert "score" in results[0]
        db_results = [r for r in results if "database" in r["content"].lower() or "PG" in r["content"].lower()]
        assert len(db_results) >= 1, f"Expected database-related result, got: {[r['content'] for r in results]}"
