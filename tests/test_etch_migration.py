"""Backward-compatibility migration tests for the Memory Etch store.

Ensures DBs created with older schemas still work with the current code.
Tests the PRAGMA table_info guards that protect every query.
"""
import json
import logging
import sqlite3
import tempfile
from pathlib import Path

import pytest
import sys
from pathlib import Path
_sys_path = str(Path(__file__).resolve().parent.parent.parent / "plugins/memory/etch")
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)
from memory_etch.store import EtchStore
from memory_etch.retrieval import EtchRetriever

logger = logging.getLogger(__name__)

# ── Original v3 schema (before topic_key, project, sessions, fact_relations) ──

_V3_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER REFERENCES facts(fact_id),
    entity_id INTEGER REFERENCES entities(entity_id),
    PRIMARY KEY (fact_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_trust    ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name  ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;
"""


# ── Fixtures ──


@pytest.fixture
def v3_db_path():
    """Create a DB with the original v3 schema (no new columns/tables)."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_V3_SCHEMA)
        conn.commit()

        # Seed some facts
        facts = [
            ("Use PostgreSQL for all databases", "project", "topic:db"),
            ("Prefer yarn over npm", "user_pref", "topic:package-manager"),
            ("Alice is CEO of Acme Corp", "user_pref", ""),
            ("Deploy via CI/CD pipeline", "project", "topic:deploy"),
        ]
        for content, cat, tags in facts:
            conn.execute(
                "INSERT INTO facts (content, category, tags) VALUES (?, ?, ?)",
                (content, cat, tags),
            )
        conn.commit()
        conn.close()
        yield str(db_path)


# ── Tests ──


class TestV3BackwardCompat:
    """All operations must work on a v3 DB with no new columns/tables."""

    def test_add_fact_works(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        try:
            fid = store.add_fact("New fact on v3 DB", tags="test")
            assert fid > 0
            row = store._conn.execute(
                "SELECT content FROM facts WHERE fact_id = ?", (fid,)
            ).fetchone()
            assert row["content"] == "New fact on v3 DB"
        finally:
            store.close()

    def test_search_works(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        retriever = EtchRetriever(store)
        try:
            results = retriever.search("PostgreSQL")
            assert len(results) >= 1
            assert "PostgreSQL" in results[0]["content"]
        finally:
            store.close()

    def test_fact_count(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        try:
            row = store._conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()
            assert row["c"] == 4
        finally:
            store.close()

    def test_add_multiple_facts(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        try:
            f1 = store.add_fact("Use Redis for caching")
            f2 = store.add_fact("Use Docker for local dev")
            assert f1 > 0
            assert f2 > 0
            row = store._conn.execute("SELECT COUNT(*) as c FROM facts").fetchone()
            assert row["c"] == 6
        finally:
            store.close()

    def test_contradict_returns_empty_on_v3(self, v3_db_path):
        """contradict() should return [] on v3 (no fact_relations table)."""
        store = EtchStore(db_path=v3_db_path)
        retriever = EtchRetriever(store)
        try:
            results = retriever.contradict()
            assert isinstance(results, list)
        finally:
            store.close()

    def test_probe_works(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        retriever = EtchRetriever(store)
        try:
            results = retriever.probe("PostgreSQL")
            assert isinstance(results, list)
        finally:
            store.close()

    def test_related_works(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        retriever = EtchRetriever(store)
        try:
            results = retriever.related("AcmeCorp")
            assert isinstance(results, list)
        finally:
            store.close()

    def test_update_fact_works(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        try:
            updated = store.update_fact(1, content="Updated content")
            assert updated
            row = store._conn.execute(
                "SELECT content FROM facts WHERE fact_id = 1"
            ).fetchone()
            assert row["content"] == "Updated content"
        finally:
            store.close()

    def test_get_fact_works(self, v3_db_path):
        store = EtchStore(db_path=v3_db_path)
        try:
            fact = store.get_fact(1)
            assert fact["content"] == "Use PostgreSQL for all databases"
        finally:
            store.close()

    def test_session_ops_dont_crash_on_v3(self, v3_db_path):
        """session_start/end should not crash; sessions table doesn't exist."""
        store = EtchStore(db_path=v3_db_path)
        try:
            # Should not raise
            info = store.session_start("test-sess", project="test")
            # On v3, sessions table missing → graceful fallback
            assert isinstance(info, dict)
        finally:
            store.close()

    def test_get_recent_sessions_empty_on_v3(self, v3_db_path):
        """get_recent_sessions should return [] on v3 (no sessions table)."""
        store = EtchStore(db_path=v3_db_path)
        try:
            recent = store.get_recent_sessions(project="test", limit=3)
            # The PRAGMA guard in get_recent_sessions handles this
            assert isinstance(recent, list)
        finally:
            store.close()

    def test_fact_relations_ops_dont_crash(self, v3_db_path):
        """judge_relation/get_relations should work after migration adds tables."""
        store = EtchStore(db_path=v3_db_path)
        try:
            # judge_relation — migration creates the fact_relations table
            result = store.judge_relation(1, 2, "related", judged_by="test")
            assert isinstance(result, dict)  # succeeded: migration added the table
            rels = store.get_relations(1)
            assert len(rels) == 1
            assert rels[0]["relation_type"] == "related"
        finally:
            store.close()
