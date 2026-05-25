"""Schema DDL, schema creation, schema migrations, and FTS5 sanitisation.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

# Valid scopes for Hive Memory governance — duplicates the global in store.py
# so that this module is fully self-contained.
VALID_SCOPES: set[str] = {"canonical", "inbox", "personal", "ephemeral"}


def _sanitize_fts5(query: str) -> str:
    """Strip FTS5-unsafe characters from a natural-language query.

    FTS5 treats certain characters as query operators (``?``, ``'``, ``!``,
    etc.) which cause syntax errors or silently produce no matches when
    used in ``MATCH`` expressions.  This replacement is lossy — it drops
    the character — but keeps the rest of the query usable.
    """
    cleaned = re.sub(r"""[?!'".;:\-+=~`@#$%^&*()\[\]{}|,<>]""", " ", query)
    return " ".join(cleaned.split())


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    hrr_vector      BLOB,
    embedding       BLOB,
    reinforcement_count INTEGER DEFAULT 0,
    consolidated    INTEGER DEFAULT 0,
    importance      REAL DEFAULT 0.5,
    session_id      TEXT DEFAULT '',
    topic_key       TEXT DEFAULT '',
    revision_count  INTEGER DEFAULT 0,
    project         TEXT DEFAULT '',
    deleted         INTEGER DEFAULT 0,
    deleted_reason  TEXT DEFAULT '',
    replaced_by     INTEGER DEFAULT NULL
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

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    project         TEXT DEFAULT '',
    status          TEXT DEFAULT 'active',
    fact_count      INTEGER DEFAULT 0,
    summary         TEXT DEFAULT '',
    metadata        TEXT DEFAULT '{}',
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at        TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fact_relations (
    relation_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id_a      INTEGER NOT NULL REFERENCES facts(fact_id),
    fact_id_b      INTEGER NOT NULL REFERENCES facts(fact_id),
    relation_type  TEXT NOT NULL
                   CHECK(relation_type IN ('related', 'compatible', 'scoped', 'conflicts_with', 'supersedes', 'not_conflict', 'derived_from')),
    confidence     REAL DEFAULT 0.5,
    judged_by      TEXT DEFAULT 'auto',
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(fact_id_a, fact_id_b)
);

CREATE TABLE IF NOT EXISTS extractions (
    extraction_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT DEFAULT '',
    facts_found     INTEGER DEFAULT 0,
    facts_extracted INTEGER DEFAULT 0,
    facts_added     INTEGER DEFAULT 0,
    dedup_skipped   INTEGER DEFAULT 0,
    model_used      TEXT DEFAULT '',
    duration_ms     INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS turn_buffer (
    turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT DEFAULT '',
    role        TEXT DEFAULT '',
    content     TEXT DEFAULT '',
    meaningful  INTEGER DEFAULT 0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS failed_buffers (
    failed_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT DEFAULT '',
    turn_count  INTEGER DEFAULT 0,
    error       TEXT DEFAULT '',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS event_log (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    fact_id     INTEGER,
    project     TEXT DEFAULT '',
    metadata    TEXT DEFAULT '{}',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS workspaces (
    workspace_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT DEFAULT '',
    tags TEXT DEFAULT '[]',
    settings TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    fact_count INTEGER DEFAULT 0,
    last_active TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    deleted INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    description     TEXT DEFAULT '',
    tags            TEXT DEFAULT '[]',
    project         TEXT DEFAULT '',
    data            TEXT NOT NULL,
    state_hash      TEXT DEFAULT '',
    fact_count      INTEGER DEFAULT 0,
    session_count   INTEGER DEFAULT 0,
    workspace_count INTEGER DEFAULT 0,
    relation_count  INTEGER DEFAULT 0,
    turn_count      INTEGER DEFAULT 0,
    event_count     INTEGER DEFAULT 0,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_schemas (
    fact_type TEXT PRIMARY KEY,
    description TEXT DEFAULT '',
    required_fields TEXT DEFAULT '[]',
    optional_fields TEXT DEFAULT '[]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _ensure_schema(store) -> None:
    """Create all tables if they don't exist."""
    store._conn.executescript(_SCHEMA)
    store._conn.commit()


def _recreate_fts(store) -> None:
    """Recreate FTS triggers and rebuild the external-content index."""
    store._conn.executescript("""
        DROP TRIGGER IF EXISTS facts_ai;
        DROP TRIGGER IF EXISTS facts_ad;
        DROP TRIGGER IF EXISTS facts_au;
        DROP TABLE IF EXISTS facts_fts;

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

        CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content, tags ON facts BEGIN
            INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                VALUES ('delete', old.fact_id, old.content, old.tags);
            INSERT INTO facts_fts(rowid, content, tags)
                VALUES (new.fact_id, new.content, new.tags);
        END;
    """)
    store._conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")


def _remove_legacy_content_unique(store) -> None:
    """Drop the legacy UNIQUE(content) constraint by rebuilding facts."""
    row = store._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='facts'"
    ).fetchone()
    sql = row["sql"] if row else ""
    if "UNIQUE" not in sql.upper():
        return

    logger.info("Migrating schema: removing legacy UNIQUE(content) constraint")
    store._conn.execute("PRAGMA foreign_keys=OFF")
    store._conn.executescript("""
        DROP TRIGGER IF EXISTS facts_ai;
        DROP TRIGGER IF EXISTS facts_ad;
        DROP TRIGGER IF EXISTS facts_au;
        DROP TABLE IF EXISTS facts_fts;

        CREATE TABLE facts_new (
            fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
            content         TEXT NOT NULL,
            category        TEXT DEFAULT 'general',
            tags            TEXT DEFAULT '',
            trust_score     REAL DEFAULT 0.5,
            retrieval_count INTEGER DEFAULT 0,
            helpful_count   INTEGER DEFAULT 0,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hrr_vector      BLOB,
            embedding       BLOB,
            reinforcement_count INTEGER DEFAULT 0,
            consolidated    INTEGER DEFAULT 0,
            importance      REAL DEFAULT 0.5,
            session_id      TEXT DEFAULT '',
            topic_key       TEXT DEFAULT '',
            revision_count  INTEGER DEFAULT 0,
            project         TEXT DEFAULT '',
            deleted         INTEGER DEFAULT 0,
            deleted_reason  TEXT DEFAULT '',
            replaced_by     INTEGER DEFAULT NULL,
            what            TEXT DEFAULT '',
            why             TEXT DEFAULT '',
            where_text      TEXT DEFAULT '',
            learned         TEXT DEFAULT '',
            content_hash    TEXT DEFAULT '',
            duplicate_count INTEGER DEFAULT 0,
            last_retrieved_at TIMESTAMP,
            source_harness  TEXT DEFAULT '',
            source_agent    TEXT DEFAULT '',
            source_kind     TEXT DEFAULT '',
            scope           TEXT DEFAULT 'canonical',
            fact_type       TEXT DEFAULT ''
        );

        INSERT INTO facts_new (
            fact_id, content, category, tags, trust_score, retrieval_count,
            helpful_count, created_at, updated_at, hrr_vector, embedding,
            reinforcement_count, consolidated, importance, session_id,
            topic_key, revision_count, project, deleted, deleted_reason,
            replaced_by, what, why, where_text, learned, content_hash,
            duplicate_count, last_retrieved_at, source_harness, source_agent,
            source_kind, scope, fact_type
        )
        SELECT
            fact_id, content, category, tags, trust_score, retrieval_count,
            helpful_count, created_at, updated_at, hrr_vector, embedding,
            reinforcement_count, consolidated, importance, session_id,
            topic_key, revision_count, project, deleted, deleted_reason,
            replaced_by, what, why, where_text, learned, content_hash,
            duplicate_count, last_retrieved_at, source_harness, source_agent,
            source_kind, scope, fact_type
        FROM facts;

        DROP TABLE facts;
        ALTER TABLE facts_new RENAME TO facts;
    """)
    store._conn.execute("PRAGMA foreign_keys=ON")
    _recreate_fts(store)
    # Recreate indexes lost during table rebuild
    store._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_trust ON facts(trust_score DESC)"
    )
    store._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category)"
    )
    store._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_content_hash "
        "ON facts(content_hash, project)"
    )
    store._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_last_retrieved "
        "ON facts(last_retrieved_at)"
    )


def _refresh_content_hashes(store) -> None:
    """Recompute content hashes with project and scope as dedup boundaries."""
    rows = store._conn.execute(
        "SELECT fact_id, content, project, scope FROM facts"
    ).fetchall()
    for row in rows:
        content_hash = hashlib.sha256(
            row["content"].encode()
            + str(row["project"] or "").encode()
            + str(row["scope"] or "canonical").encode()
        ).hexdigest()
        store._conn.execute(
            "UPDATE facts SET content_hash = ? WHERE fact_id = ?",
            (content_hash, row["fact_id"]),
        )


def _migrate_schema(store) -> None:
    """Backward-compatible schema migrations."""
    cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(facts)").fetchall()}

    for col, type_def in [
        ("session_id", "TEXT DEFAULT ''"),
        ("topic_key", "TEXT DEFAULT ''"),
        ("revision_count", "INTEGER DEFAULT 0"),
        ("project", "TEXT DEFAULT ''"),
        ("embedding", "BLOB"),
        ("reinforcement_count", "INTEGER DEFAULT 0"),
        ("consolidated", "INTEGER DEFAULT 0"),
        ("importance", "REAL DEFAULT 0.5"),
        ("deleted", "INTEGER DEFAULT 0"),
        ("deleted_reason", "TEXT DEFAULT ''"),
        ("replaced_by", "INTEGER DEFAULT NULL"),
        ("what", "TEXT DEFAULT ''"),
        ("why", "TEXT DEFAULT ''"),
        ("where_text", "TEXT DEFAULT ''"),
        ("learned", "TEXT DEFAULT ''"),
    ]:
        if col not in cols:
            logger.info("Migrating schema: adding column %s", col)
            store._conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {type_def}")

    # Sessions table
    tables = {r["name"] for r in store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "sessions" not in tables:
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                project TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                fact_count INTEGER DEFAULT 0,
                summary TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP
            );
        """)

    if "turn_buffer" not in tables:
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS turn_buffer (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT DEFAULT '',
                role TEXT DEFAULT '',
                content TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # turn_buffer: meaningful column
    if "turn_buffer" in tables:
        turn_cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(turn_buffer)").fetchall()}
        if "meaningful" not in turn_cols:
            logger.info("Migrating schema: adding column meaningful to turn_buffer")
            store._conn.execute("ALTER TABLE turn_buffer ADD COLUMN meaningful INTEGER DEFAULT 0")

    # extractions: facts_extracted column
    if "extractions" in tables:
        ext_cols = {r["name"] for r in store._conn.execute("PRAGMA table_info(extractions)").fetchall()}
        if "facts_extracted" not in ext_cols:
            logger.info("Migrating schema: adding column facts_extracted to extractions")
            store._conn.execute("ALTER TABLE extractions ADD COLUMN facts_extracted INTEGER DEFAULT 0")

    # failed_buffers table
    if "failed_buffers" not in tables:
        logger.info("Migrating schema: creating failed_buffers table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS failed_buffers (
                failed_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT DEFAULT '',
                turn_count  INTEGER DEFAULT 0,
                error       TEXT DEFAULT '',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # event_log table
    if "event_log" not in tables:
        logger.info("Migrating schema: creating event_log table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS event_log (
                event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT NOT NULL,
                fact_id     INTEGER,
                project     TEXT DEFAULT '',
                metadata    TEXT DEFAULT '{}',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # workspaces table
    if "workspaces" not in tables:
        logger.info("Migrating schema: creating workspaces table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT DEFAULT '',
                tags TEXT DEFAULT '[]',
                settings TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                fact_count INTEGER DEFAULT 0,
                last_active TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                deleted INTEGER DEFAULT 0
            );
        """)

    # snapshots table
    if "snapshots" not in tables:
        logger.info("Migrating schema: creating snapshots table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                name            TEXT NOT NULL UNIQUE,
                description     TEXT DEFAULT '',
                tags            TEXT DEFAULT '[]',
                project         TEXT DEFAULT '',
                data            TEXT NOT NULL,
                state_hash      TEXT DEFAULT '',
                fact_count      INTEGER DEFAULT 0,
                session_count   INTEGER DEFAULT 0,
                workspace_count INTEGER DEFAULT 0,
                relation_count  INTEGER DEFAULT 0,
                turn_count      INTEGER DEFAULT 0,
                event_count     INTEGER DEFAULT 0,
                created_at      TEXT DEFAULT (datetime('now'))
            );
        """)

    if "fact_relations" not in tables:
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS fact_relations (
                relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id_a INTEGER NOT NULL REFERENCES facts(fact_id),
                fact_id_b INTEGER NOT NULL REFERENCES facts(fact_id),
                relation_type TEXT NOT NULL
                    CHECK(relation_type IN ('related','compatible','scoped','conflicts_with','supersedes','not_conflict')),
                confidence REAL DEFAULT 0.5,
                judged_by TEXT DEFAULT 'auto',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(fact_id_a, fact_id_b)
            );
        """)

    # content_hash and duplicate_count for lifetime dedup
    for col, type_def in [
        ("content_hash", "TEXT DEFAULT ''"),
        ("duplicate_count", "INTEGER DEFAULT 0"),
    ]:
        if col not in cols:
            logger.info("Migrating schema: adding column %s", col)
            store._conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {type_def}")

    # Index for O(1) content_hash lookups
    store._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_content_hash "
        "ON facts(content_hash, project)"
    )

    # last_retrieved_at for eviction tracking
    if "last_retrieved_at" not in cols:
        logger.info("Migrating schema: adding column last_retrieved_at")
        store._conn.execute(
            "ALTER TABLE facts ADD COLUMN last_retrieved_at TIMESTAMP"
        )

    # Hive Memory v1 provenance and scope columns
    for col, type_def in [
        ("source_harness", "TEXT DEFAULT ''"),
        ("source_agent", "TEXT DEFAULT ''"),
        ("source_kind", "TEXT DEFAULT ''"),
        ("scope", "TEXT DEFAULT 'canonical'"),
    ]:
        if col not in cols:
            logger.info("Migrating schema: adding column %s", col)
            store._conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {type_def}")

    # Typed facts: fact_type column
    if "fact_type" not in cols:
        logger.info("Migrating schema: adding column fact_type to facts")
        store._conn.execute("ALTER TABLE facts ADD COLUMN fact_type TEXT DEFAULT ''")

    # fact_schemas table
    if "fact_schemas" not in tables:
        logger.info("Migrating schema: creating fact_schemas table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS fact_schemas (
                fact_type TEXT PRIMARY KEY,
                description TEXT DEFAULT '',
                required_fields TEXT DEFAULT '[]',
                optional_fields TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # Index for eviction queries
    store._conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_facts_last_retrieved "
        "ON facts(last_retrieved_at)"
    )

    _remove_legacy_content_unique(store)
    _refresh_content_hashes(store)
    _recreate_fts(store)

    # fact_relations: add 'derived_from' to CHECK constraint
    if "fact_relations" in tables:
        row = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='fact_relations'"
        ).fetchone()
        if row and "derived_from" not in row["sql"]:
            logger.info("Migrating schema: adding derived_from to fact_relations CHECK")
            store._conn.execute("PRAGMA foreign_keys=OFF")
            store._conn.execute("""
                CREATE TABLE fact_relations_new (
                    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact_id_a INTEGER NOT NULL REFERENCES facts(fact_id),
                    fact_id_b INTEGER NOT NULL REFERENCES facts(fact_id),
                    relation_type TEXT NOT NULL
                        CHECK(relation_type IN ('related','compatible','scoped','conflicts_with','supersedes','not_conflict','derived_from')),
                    confidence REAL DEFAULT 0.5,
                    judged_by TEXT DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(fact_id_a, fact_id_b)
                )
            """)
            store._conn.execute(
                "INSERT INTO fact_relations_new SELECT * FROM fact_relations"
            )
            store._conn.execute("DROP TABLE fact_relations")
            store._conn.execute("ALTER TABLE fact_relations_new RENAME TO fact_relations")
            store._conn.execute("PRAGMA foreign_keys=ON")

    # ------------------------------------------------------------------
    # Distributed Sync tables
    # ------------------------------------------------------------------

    # store_meta: key-value for EtchStore instance metadata (node_id, schema_version)
    if "store_meta" not in tables:
        logger.info("Migrating schema: creating store_meta table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS store_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    # sync_peers: registry of known sync peers
    if "sync_peers" not in tables:
        logger.info("Migrating schema: creating sync_peers table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sync_peers (
                peer_id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT UNIQUE NOT NULL,
                kind              TEXT DEFAULT 'file',
                address           TEXT DEFAULT '',
                peer_node_id      TEXT DEFAULT '',
                last_sync_cursor  INTEGER DEFAULT 0,
                last_sync_at      TEXT,
                created_at        TEXT DEFAULT (datetime('now'))
            );
        """)

    # sync_conflicts: tracks facts with content_hash collisions between instances
    if "sync_conflicts" not in tables:
        logger.info("Migrating schema: creating sync_conflicts table")
        store._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sync_conflicts (
                conflict_id       INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash      TEXT NOT NULL,
                local_fact_id     INTEGER,
                local_content     TEXT DEFAULT '',
                local_metadata    TEXT DEFAULT '{}',
                remote_data       TEXT NOT NULL,
                status            TEXT DEFAULT 'unresolved',
                created_at        TEXT DEFAULT (datetime('now')),
                resolved_at       TEXT
            );
        """)

    store._conn.commit()
