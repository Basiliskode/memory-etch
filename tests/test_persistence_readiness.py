import json
import sqlite3

import pytest

from memento import hrr
from memento.etch import EtchMemoryProvider
from memento.store import EtchStore


def test_export_import_roundtrips_public_fact_fields(tmp_path):
    source = EtchStore(":memory:", auto_migrate=True)
    target = EtchStore(":memory:", auto_migrate=True)
    export_path = tmp_path / "memory.json"
    try:
        source.register_schema("decision", description="Decision facts")
        target.register_schema("decision", description="Decision facts")
        source.add_fact(
            "Use scoped exports for production readiness",
            what="Export public fields",
            why="External backups must be lossless",
            where_text="store export/import",
            learned="Roundtrip tests prevent silent data loss",
            scope="personal",
            fact_type="decision",
        )

        source.export_memory(str(export_path))
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        exported_fact = exported["facts"][0]

        assert exported_fact["what"] == "Export public fields"
        assert exported_fact["why"] == "External backups must be lossless"
        assert exported_fact["where"] == "store export/import"
        assert exported_fact["learned"] == "Roundtrip tests prevent silent data loss"
        assert exported_fact["scope"] == "personal"
        assert exported_fact["fact_type"] == "decision"

        target.import_memory(str(export_path))
        imported = target.get_fact(target.list_facts(scope="personal")[0]["fact_id"])

        assert imported["what"] == "Export public fields"
        assert imported["why"] == "External backups must be lossless"
        assert imported["where_text"] == "store export/import"
        assert imported["learned"] == "Roundtrip tests prevent silent data loss"
        assert imported["scope"] == "personal"
        assert imported["fact_type"] == "decision"
    finally:
        source.close()
        target.close()


def test_dedup_is_scoped_by_project_and_scope():
    store = EtchStore(":memory:", auto_migrate=True)
    try:
        first = store.add_fact("Same content", project="alpha", scope="canonical")
        same_scope = store.add_fact("Same content", project="alpha", scope="canonical")
        other_project = store.add_fact("Same content", project="beta", scope="canonical")
        other_scope = store.add_fact("Same content", project="alpha", scope="personal")

        assert same_scope == first
        assert other_project != first
        assert other_scope != first
        assert len(store.list_facts(project="alpha", scope="canonical")) == 1
        assert len(store.list_facts(project="alpha", scope="personal")) == 1
        assert len(store.list_facts(project="beta", scope="canonical")) == 1
    finally:
        store.close()


def test_legacy_migration_removes_content_unique_and_rebuilds_fts(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE facts (
            fact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL UNIQUE,
            category TEXT DEFAULT 'general',
            tags TEXT DEFAULT '',
            trust_score REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector BLOB
        );
        CREATE VIRTUAL TABLE facts_fts
            USING fts5(content, tags, content=facts, content_rowid=fact_id);
        INSERT INTO facts (content, category, tags)
            VALUES ('Legacy searchable needle', 'general', 'legacy');
        """
    )
    conn.commit()
    conn.close()

    store = EtchStore(str(db_path), auto_migrate=True)
    try:
        table_sql = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts'"
        ).fetchone()["sql"]
        assert "UNIQUE" not in table_sql.upper()
        assert store.search_facts("needle")

        first = store.add_fact("Duplicate after migration", project="alpha")
        second = store.add_fact("Duplicate after migration", project="beta")
        assert second != first
    finally:
        store.close()


@pytest.mark.skipif(not hrr.HAS_NUMPY, reason="HRR vectors require numpy")
def test_close_flushes_pending_hrr_vectors_before_closing(tmp_path):
    store = EtchStore(str(tmp_path / "hrr_close.db"), auto_migrate=True)
    store._hrr_flush_stop.set()
    store._signal_flush()
    if store._hrr_flush_thread and store._hrr_flush_thread.is_alive():
        store._hrr_flush_thread.join()
    fid = store.add_fact("Close must persist pending HRR vectors")

    assert store._pending_hrr
    store.close()

    conn = sqlite3.connect(tmp_path / "hrr_close.db")
    try:
        row = conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?",
            (fid,),
        ).fetchone()
        assert row is not None
        assert row[0] is not None
    finally:
        conn.close()


def test_store_rejects_empty_db_path():
    with pytest.raises(ValueError, match="db_path"):
        EtchStore("", auto_migrate=True)


def test_provider_default_db_path_is_explicit_safe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    provider = EtchMemoryProvider({"auto_extract_llm": False})
    try:
        provider.initialize("safe-default")
        assert provider._store is not None
        assert provider._store._db_path == "memento_safe-default.db"
        assert (tmp_path / "memento_safe-default.db").exists()
    finally:
        provider.shutdown()


def test_return_metadata_conflict_detection_uses_locked_connection(tmp_path):
    store = EtchStore(str(tmp_path / "conflicts.db"), auto_migrate=True)
    try:
        first = store.add_fact("alpha beta gamma", project="locks")
        result = store.add_fact(
            "alpha beta delta",
            project="locks",
            return_metadata=True,
        )
        assert result["id"] != first
        assert isinstance(result["conflicts_with"], list)
    finally:
        store.close()
