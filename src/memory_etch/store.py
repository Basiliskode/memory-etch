"""SQLite-backed fact store with trust scoring, HRR vectors, and consolidation.

This is the core storage layer of Memory Etch. It manages:

- Facts with trust scores, importance, and revision tracking
- Entity extraction and N:M relationships
- FTS5 full-text search with auto-sync triggers
- Session tracking
- Fact relations (compatible, conflicts_with, supersedes, etc.)
- HRR vector encoding for semantic similarity
- Soft delete with audit trail
- Active consolidation (LLM-decide on collision)
"""

import hashlib
import json
import logging
import re
import sqlite3
import struct
import threading
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional, Callable

from . import hrr
from .embedding import EmbeddingProvider, NoopProvider
from .project import detect_project

logger = logging.getLogger(__name__)


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
    content         TEXT NOT NULL UNIQUE,
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

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
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
                   CHECK(relation_type IN ('related', 'compatible', 'scoped', 'conflicts_with', 'supersedes', 'not_conflict')),
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
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EtchStore:
    """SQLite-backed fact store.

    Thread-safe via RLock. Handles schema creation, migration, CRUD,
    FTS5 sync, HRR encoding, soft delete, and consolidation.

    Args:
        db_path: Path to the SQLite database file.
        hrr_dim: Dimension for HRR vectors (default: 256).
        auto_migrate: Whether to run schema migrations on init.
    """

    def __init__(
        self,
        db_path: str,
        hrr_dim: int = 256,
        auto_migrate: bool = True,
        embedding_provider: Optional[EmbeddingProvider] = None,
        project: Optional[str] = None,
    ) -> None:
        """Initialize the EtchStore.

        Creates or opens the SQLite database, runs schema migrations, and
        starts the background HRR flush thread when NumPy is available.

        Args:
            db_path: Path to the SQLite database file.
            hrr_dim: Dimension for HRR vectors (default: 256).
            auto_migrate: Whether to run schema creation and migration
                on initialization (default: True).
            embedding_provider: Optional EmbeddingProvider for semantic
                search. If None, uses NoopProvider (no-op, no deps).
            project: Optional project name. ``"auto"`` calls
                ``detect_project()`` on cwd to auto-detect.

        Raises:
            sqlite3.Error: If the database cannot be opened or created.
        """
        self._db_path = db_path
        self._hrr_dim = hrr_dim
        self._lock = threading.RLock()
        self._embedding_provider = embedding_provider or NoopProvider()
        self._project = self._resolve_project(project)

        # HRR async flush
        self._pending_hrr: list[tuple[int, str]] = []  # (fact_id, content)
        self._hrr_ready = threading.Event()
        self._hrr_flush_signal = threading.Event()
        self._hrr_flush_stop = threading.Event()
        self._hrr_flush_thread: Optional[threading.Thread] = None

        # HRR vector cache (fact_id → np.ndarray, LRU max 500)
        self._hrr_vector_cache: dict[int, "np.ndarray"] = {}
        self._hrr_cache_max = 500

        # Connect
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA cache_size=-262144")  # 256 MB
        self._conn.execute("PRAGMA mmap_size=1073741824")  # 1 GB
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")

        if auto_migrate:
            self._ensure_schema()
            self._migrate_schema()
            self._start_hrr_flush()

    @staticmethod
    def _resolve_project(project: Optional[str]) -> Optional[str]:
        """Resolve the ``project`` parameter.

        If ``project`` is the literal string ``"auto"``, calls
        ``detect_project()`` on the current working directory.
        Otherwise returns the value as-is.
        """
        if project == "auto":
            return detect_project()
        return project

    # ------------------------------------------------------------------
    # Schema & migrations
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Backward-compatible schema migrations."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(facts)").fetchall()}

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
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {type_def}")

        # Sessions table
        tables = {r["name"] for r in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "sessions" not in tables:
            self._conn.executescript("""
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
            self._conn.executescript("""
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
            turn_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(turn_buffer)").fetchall()}
            if "meaningful" not in turn_cols:
                logger.info("Migrating schema: adding column meaningful to turn_buffer")
                self._conn.execute("ALTER TABLE turn_buffer ADD COLUMN meaningful INTEGER DEFAULT 0")

        # extractions: facts_extracted column
        if "extractions" in tables:
            ext_cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(extractions)").fetchall()}
            if "facts_extracted" not in ext_cols:
                logger.info("Migrating schema: adding column facts_extracted to extractions")
                self._conn.execute("ALTER TABLE extractions ADD COLUMN facts_extracted INTEGER DEFAULT 0")

        # failed_buffers table
        if "failed_buffers" not in tables:
            logger.info("Migrating schema: creating failed_buffers table")
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS failed_buffers (
                    failed_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT DEFAULT '',
                    turn_count  INTEGER DEFAULT 0,
                    error       TEXT DEFAULT '',
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

        if "fact_relations" not in tables:
            self._conn.executescript("""
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

        # content_hash and duplicate_count for 60s rolling-window dedup
        for col, type_def in [
            ("content_hash", "TEXT DEFAULT ''"),
            ("duplicate_count", "INTEGER DEFAULT 0"),
        ]:
            if col not in cols:
                logger.info("Migrating schema: adding column %s", col)
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {col} {type_def}")

        # Index for O(1) content_hash lookups
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_content_hash "
            "ON facts(content_hash, project)"
        )

        # last_retrieved_at for eviction tracking
        if "last_retrieved_at" not in cols:
            logger.info("Migrating schema: adding column last_retrieved_at")
            self._conn.execute(
                "ALTER TABLE facts ADD COLUMN last_retrieved_at TIMESTAMP"
            )

        # Index for eviction queries
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_last_retrieved "
            "ON facts(last_retrieved_at)"
        )

        self._conn.commit()

    # ------------------------------------------------------------------
    # HRR flush thread
    # ------------------------------------------------------------------

    def _start_hrr_flush(self) -> None:
        if not hrr.HAS_NUMPY:
            logger.info("NumPy not available — HRR vectors disabled")
            return

        def _flush_worker():
            while not self._hrr_flush_stop.is_set():
                self._hrr_flush_signal.wait(timeout=5)
                self._hrr_flush_signal.clear()
                if self._hrr_flush_stop.is_set():
                    break
                self._flush_pending_hrr_batch()

        self._hrr_flush_thread = threading.Thread(target=_flush_worker, daemon=True)
        self._hrr_flush_thread.start()

    def _signal_flush(self) -> None:
        self._hrr_flush_signal.set()

    def _flush_pending_hrr_batch(self) -> None:
        """Snapshot pending list and encode under lock."""
        with self._lock:
            batch = self._pending_hrr.copy()
            self._pending_hrr.clear()
            if not batch:
                return

        try:
            dim = self._get_effective_hrr_dim()
            for fact_id, content in batch:
                vec = hrr.encode_text(content, dim)
                blob = hrr.phases_to_bytes(vec)
                with self._lock:
                    self._conn.execute(
                        "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                        (blob, fact_id),
                    )
                    self._invalidate_hrr_cache(fact_id)
            with self._lock:
                self._conn.commit()
        except Exception:
            logger.exception("HRR flush failed")
            # Re-queue on failure
            with self._lock:
                self._pending_hrr.extend(batch)

    def _get_effective_hrr_dim(self) -> int:
        """Detect HRR dim from existing vectors, or return default."""
        with self._lock:
            row = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE hrr_vector IS NOT NULL LIMIT 1"
            ).fetchone()
        if row and row["hrr_vector"]:
            try:
                vec = hrr.bytes_to_phases(row["hrr_vector"])
                return len(vec)
            except Exception:
                pass
        return self._hrr_dim

    def get_effective_hrr_dim(self) -> int:
        """Return the HRR dimension currently used by this store.

        Existing databases may contain HRR vectors created with a dimension
        different from the constructor default. Retrieval code should call this
        method instead of assuming its own default dimension.
        """
        return self._get_effective_hrr_dim()

    def compute_hrr_batch(self) -> None:
        """Flush pending HRR vectors synchronously.

        Public compatibility wrapper for integrations that need to force HRR
        computation before searching or shutting down.
        """
        self._flush_pending_hrr_batch()

    # ------------------------------------------------------------------
    # HRR cache
    # ------------------------------------------------------------------

    def _get_hrr_cached(self, fact_id: int):
        """Get cached HRR vector, or decode from DB."""
        if fact_id in self._hrr_vector_cache:
            return self._hrr_vector_cache[fact_id]
        with self._lock:
            row = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
        if not row or not row["hrr_vector"]:
            return None
        vec = hrr.bytes_to_phases(row["hrr_vector"])
        if len(self._hrr_vector_cache) < self._hrr_cache_max:
            self._hrr_vector_cache[fact_id] = vec
        return vec

    def _invalidate_hrr_cache(self, fact_id: int) -> None:
        self._hrr_vector_cache.pop(fact_id, None)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _maybe_store_embedding(self, fact_id: int, content: str) -> None:
        """Compute and store an embedding for a fact if the provider is active.

        Skips if the provider is NoopProvider (not configured).
        If the provider raises, the fact simply has embedding=NULL.
        """
        if isinstance(self._embedding_provider, NoopProvider):
            return
        try:
            vec = self._embedding_provider.embed([content])
            if vec and vec[0]:
                import struct
                blob = struct.pack(f"{len(vec[0])}f", *vec[0])
                self._conn.execute(
                    "UPDATE facts SET embedding=? WHERE fact_id=?",
                    (blob, fact_id),
                )
                self._conn.commit()
        except Exception:
            logger.exception("Embedding computation failed for fact %d", fact_id)

    def _search_by_embedding(self, query_emb: list[float], k: int) -> list[int]:
        """Search facts by embedding vector similarity (dot product).

        Loads stored BLOBs as float32 ndarray, L2-normalizes, computes
        dot product with query vector, returns top-k fact IDs.

        Returns empty list if numpy is unavailable or no embeddings exist.
        """
        try:
            import numpy as np  # type: ignore[import-untyped]
        except ImportError:
            return []

        with self._lock:
            rows = self._conn.execute(
                "SELECT fact_id, embedding FROM facts "
                "WHERE embedding IS NOT NULL AND (deleted IS NULL OR deleted = 0)"
            ).fetchall()

        if not rows:
            return []

        n_dim = len(query_emb)
        embs = []
        ids = []
        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            try:
                vec = np.frombuffer(blob, dtype=np.float32)
                if len(vec) != n_dim:
                    continue  # skip mismatched dimensions
                embs.append(vec)
                ids.append(r["fact_id"])
            except Exception:
                continue

        if not embs:
            return []

        # Stack into (N, dim) matrix
        matrix = np.stack(embs, axis=0)
        # L2-normalize (already normalized for fastembed, but be safe)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms

        # Query vector
        q = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Dot product (cosine similarity for unit vectors)
        scores = matrix @ q

        # Sort by score descending, get top-k
        top_indices = np.argsort(scores)[::-1][:k]
        return [ids[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Fact CRUD
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        trust_score: Optional[float] = None,
        importance: Optional[float] = None,
        project: str = "",
        session_id: str = "",
        topic_key: str = "",
        entities: Optional[list[str]] = None,
        embedding: Optional[bytes] = None,
        what: Optional[str] = None,
        why: Optional[str] = None,
        where_text: Optional[str] = None,
        learned: Optional[str] = None,
        return_metadata: bool = False,
    ) -> int | dict:
        """Insert a new fact.

        When tags contain ``topic:<name>``, the topic_key is auto-extracted
        and an existing fact with the same key is UPDATEd (topic upsert).

        Content hash dedup: if the same ``content + project`` is added within
        a 60-second window, ``duplicate_count`` is incremented and the existing
        ``fact_id`` is returned (no new row created).

        Args:
            content: Fact text content.
            category: Fact category (e.g. "general", "project", "user_pref").
            tags: Comma-separated tags. ``topic:<name>`` triggers topic upsert.
            trust_score: Initial trust score (default: 0.5).
            importance: Fact importance (default: 0.5).
            project: Optional project namespace.
            session_id: Optional session identifier.
            topic_key: Optional topic key for upsert behavior.
            entities: Optional list of entity names to associate.
            embedding: Optional pre-computed embedding bytes.
            what: Optional structured "what" field.
            why: Optional structured "why" field.
            where_text: Optional structured "where" field (``where`` is a
                SQL reserved word, so we use ``where_text``).
            learned: Optional structured "learned" field.
            return_metadata: If True, returns a dict with ``id``, ``status``,
                and optional ``conflicts_with``. If False (default), returns
                the ``fact_id`` as an int (backward compat).

        Returns:
            The ``fact_id`` (int) by default, or a dict with metadata when
            ``return_metadata=True``.

        Raises:
            sqlite3.Error: On database-level errors.
        """
        if trust_score is None:
            trust_score = 0.5
        if importance is None:
            importance = 0.5

        # SHA-256 content hash for 60s rolling-window dedup
        content_hash = hashlib.sha256(
            content.encode() + str(project or "").encode()
        ).hexdigest()

        # Structured field values.
        # For INSERT we default to empty string; for UPDATE we pass None
        # so that COALESCE preserves the existing value.
        what_val: Optional[str] = what
        why_val: Optional[str] = why
        where_val: Optional[str] = where_text
        learned_val: Optional[str] = learned

        # Auto-extract topic_key from tags if not explicitly provided
        if not topic_key:
            m = re.search(r"(?:^|,)topic:([^,]+)", tags)
            if m:
                topic_key = "topic:" + m.group(1).strip()

        with self._lock:
            # ---- Content hash dedup (60s rolling window) ----
            dedup_row = self._conn.execute(
                """SELECT fact_id, duplicate_count FROM facts
                   WHERE content_hash = ? AND project IS ?
                   AND created_at > datetime('now', '-60 seconds')
                   AND (deleted IS NULL OR deleted = 0)""",
                (content_hash, project),
            ).fetchone()
            if dedup_row:
                dedup_id = dedup_row["fact_id"]
                self._conn.execute(
                    """UPDATE facts SET duplicate_count = duplicate_count + 1,
                       updated_at = CURRENT_TIMESTAMP WHERE fact_id = ?""",
                    (dedup_id,),
                )
                self._conn.commit()
                self._invalidate_hrr_cache(dedup_id)
                if return_metadata:
                    return {"id": dedup_id, "status": "dedup"}
                return dedup_id

            # ---- Topic upsert ----
            if topic_key:
                existing = self._conn.execute(
                    "SELECT fact_id, content, revision_count FROM facts "
                    "WHERE topic_key = ? AND (deleted IS NULL OR deleted = 0) LIMIT 1",
                    (topic_key,),
                ).fetchone()
                if existing:
                    eid = existing["fact_id"]
                    self._conn.execute(
                        """UPDATE facts SET content = ?, updated_at = CURRENT_TIMESTAMP,
                           revision_count = revision_count + 1, category = ?, tags = ?,
                           trust_score = ?, importance = ?, project = ?, session_id = ?,
                           embedding = COALESCE(?, embedding),
                           what = COALESCE(?, what), why = COALESCE(?, why),
                           where_text = COALESCE(?, where_text),
                           learned = COALESCE(?, learned),
                           content_hash = ?
                        WHERE fact_id = ?""",
                        (content, category, tags, trust_score, importance,
                         project, session_id, embedding,
                         what_val, why_val, where_val, learned_val,
                         content_hash, eid),
                    )
                    self._conn.commit()
                    self._invalidate_hrr_cache(eid)
                    if hrr.HAS_NUMPY:
                        self._pending_hrr.append((eid, content))
                        self._signal_flush()
                    # Compute embedding if provider is active (skip if pre-supplied)
                    if embedding is None:
                        self._maybe_store_embedding(eid, content)
                    if return_metadata:
                        conflicts = self._detect_conflicts(
                            content=content, fact_id=eid,
                            project=project, topic_key=topic_key,
                        )
                        return {
                            "id": eid, "status": "updated",
                            "conflicts_with": conflicts,
                        }
                    return eid

            # ---- Normal INSERT ----
            try:
                self._conn.execute(
                    """INSERT OR IGNORE INTO facts
                       (content, category, tags, trust_score, importance,
                        project, session_id, topic_key, embedding,
                        what, why, where_text, learned, content_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (content, category, tags, trust_score, importance,
                     project, session_id, topic_key, embedding,
                     what_val or "", why_val or "", where_val or "",
                     learned_val or "", content_hash),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                fact_id = row["fact_id"] if row else 0
            except sqlite3.IntegrityError:
                logger.warning("Duplicate fact (content collision): %s", content[:60])
                row = self._conn.execute(
                    "SELECT fact_id FROM facts WHERE content = ?", (content,)
                ).fetchone()
                rid = row["fact_id"] if row else 0
                if return_metadata:
                    return {"id": rid, "status": "dedup"}
                return rid

            if fact_id and hrr.HAS_NUMPY:
                self._pending_hrr.append((fact_id, content))
                self._signal_flush()

            if entities:
                for entity_name in entities:
                    self._ensure_entity(fact_id, entity_name)

            # Compute embedding if provider is active and not pre-supplied
            if fact_id and embedding is None:
                self._maybe_store_embedding(fact_id, content)

        # ---- Conflict surfacing (outside lock) ----
        if return_metadata and fact_id:
            conflicts = self._detect_conflicts(
                content=content,
                fact_id=fact_id,
                project=project,
                topic_key=topic_key,
            )
            return {"id": fact_id, "status": "created", "conflicts_with": conflicts}

        return fact_id

    def _detect_conflicts(
        self,
        content: str,
        fact_id: int,
        project: str,
        topic_key: str,
        limit: int = 5,
    ) -> list[dict]:
        """Search for existing facts that conflict with a newly added fact.

        Uses FTS5 with an OR-based query (non-trivial content tokens) and
        topic_key matching to detect conflicts.

        Args:
            content: The content of the newly added fact.
            fact_id: The ID of the newly added fact (excluded from results).
            project: Project namespace to scope the search.
            topic_key: Topic key of the new fact (for topic-based matching).
            limit: Max conflict candidates to return.

        Returns:
            List of dicts with keys ``id``, ``content``, ``score``.
        """
        # Build an OR-based FTS5 query from non-trivial content words.
        # FTS5 default MATCH requires ALL terms (AND), which is too strict
        # for conflict detection — we want any content overlap.
        words = content.split()
        # Filter out very short tokens (likely stop words / noise)
        sig_words = [w for w in words if len(w) >= 3]
        if not sig_words:
            return []

        or_query = " OR ".join(_sanitize_fts5(w) for w in sig_words)
        if not or_query.strip():
            return []

        try:
            params: list[str | int] = [or_query, fact_id]
            project_filter = "AND f.project IS ?"
            params.append(project)
            rows = self._conn.execute(
                f"""SELECT f.fact_id, f.content, f.topic_key, fts.rank
                    FROM facts f
                    JOIN facts_fts fts ON fts.rowid = f.fact_id
                    WHERE facts_fts MATCH ?
                    AND f.fact_id != ?
                    AND (f.deleted IS NULL OR f.deleted = 0)
                    {project_filter}
                    ORDER BY fts.rank
                    LIMIT ?""",
                params + [limit],
            ).fetchall()

            conflicts: list[dict] = []
            for row in rows:
                # FTS5 rank in SELECT returns -BM25.
                # rank = 0 → perfect match (all query terms found)
                # rank < 0 → partial match (some terms matched)
                rank = row["rank"]
                # Any non-zero rank means FTS5 detected overlap.
                is_similar = rank < 0
                same_topic = (
                    topic_key and topic_key == row.get("topic_key", "") and topic_key
                )
                if is_similar or same_topic:
                    score = -rank if rank < 0 else 0.0
                    conflicts.append({
                        "id": row["fact_id"],
                        "content": row["content"],
                        "score": round(score, 4),
                    })
            return conflicts
        except Exception:
            logger.exception("Conflict detection failed for fact %d", fact_id)
            return []

    def add_fact_with_consolidation(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        trust_score: Optional[float] = None,
        importance: Optional[float] = None,
        project: str = "",
        session_id: str = "",
        topic_key: str = "",
        entities: Optional[list[str]] = None,
        search_fn: Optional[Callable] = None,
        llm_decide_fn: Optional[Callable] = None,
    ) -> dict:
        """Add a fact with active consolidation — merges or deletes old facts on collision.

        Returns {"action": "added"|"merged"|"skipped"|"error", "fact_id": int, "detail": str}.
        """
        # Fast path: no consolidation needed
        if not search_fn or not llm_decide_fn:
            fid = self.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
            return {"action": "added" if fid else "error", "fact_id": fid, "detail": ""}

        # Search for similar existing facts
        try:
            results = search_fn(query=content, limit=3)
        except TypeError:
            # search_fn doesn't support all kwargs
            results = search_fn(content, limit=3) if callable(search_fn) else []

        if not results:
            fid = self.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
            return {"action": "added", "fact_id": fid, "detail": "no collision"}

        # Check Jaccard similarity for fast-path collision detection
        tokens_new = set(content.lower().split())
        best_sim = 0.0
        best_result = None
        for r in results:
            r_content = r.get("content", r.get("text", str(r)))
            tokens_existing = set(r_content.lower().split())
            if not tokens_new or not tokens_existing:
                continue
            jac = len(tokens_new & tokens_existing) / len(tokens_new | tokens_existing)
            if jac > best_sim:
                best_sim = jac
                best_result = r

        # Jaccard < 0.4 → no significant overlap → ADD directly
        if best_sim < 0.4:
            fid = self.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
            return {"action": "added", "fact_id": fid, "detail": f"jaccard={best_sim:.2f} < 0.4"}

        # Jaccard >= 0.4 → let LLM decide
        try:
            decision = llm_decide_fn(new_content=content, existing=best_result)
        except Exception as exc:
            logger.warning("LLM consolidation failed (%s), falling back to ADD", exc)
            fid = self.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
            return {"action": "added", "fact_id": fid, "detail": f"llm_fallback: {exc}"}

        action = (decision or {}).get("action", "ADD")
        existing_fid = None
        if best_result:
            existing_fid = best_result.get("fact_id") or best_result.get("id")

        if action == "SKIP":
            return {"action": "skipped", "fact_id": existing_fid, "detail": decision.get("reason", "llm decided to skip")}

        if action == "UPDATE" or action == "MERGE":
            if existing_fid:
                merged = decision.get("merged_content", content)
                with self._lock:
                    self._conn.execute(
                        "UPDATE facts SET content=?, updated_at=CURRENT_TIMESTAMP, revision_count=revision_count+1 WHERE fact_id=?",
                        (merged, existing_fid),
                    )
                    self._conn.commit()
                self._invalidate_hrr_cache(existing_fid)
                return {"action": "merged", "fact_id": existing_fid, "detail": f"updated #{existing_fid}"}

        if action == "REPLACE":
            if existing_fid:
                self.soft_delete_fact(existing_fid, reason="replaced")
            fid = self.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
            return {"action": "merged", "fact_id": fid, "detail": f"replaced #{existing_fid}"}

        # Default: ADD
        fid = self.add_fact(content, category, tags, trust_score, importance, project, session_id, topic_key, entities)
        return {"action": "added", "fact_id": fid, "detail": f"action={action} defaulted to ADD"}

    def soft_delete_fact(self, fact_id: int, reason: str = "") -> bool:
        """Soft-delete a fact.

        The fact remains in the database but is excluded from searches
        by default. A deleted reason is recorded for audit.

        Args:
            fact_id: ID of the fact to soft-delete.
            reason: Optional reason for deletion.

        Returns:
            True if a fact was soft-deleted, False if already deleted or not found.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE facts SET deleted=1, deleted_reason=? WHERE fact_id=? AND (deleted IS NULL OR deleted=0)",
                (reason, fact_id),
            )
            self._conn.commit()
            self._invalidate_hrr_cache(fact_id)
        return cur.rowcount > 0

    def restore_fact(self, fact_id: int) -> bool:
        """Restore a previously soft-deleted or archived fact.

        Sets ``deleted = 0`` and clears ``deleted_reason``, making the fact
        visible in searches again. No-op if the fact is already active.

        Args:
            fact_id: ID of the fact to restore.

        Returns:
            True if a fact was restored, False if not found or already active.
        """
        with self._lock:
            cur = self._conn.execute(
                "UPDATE facts SET deleted=0, deleted_reason='' "
                "WHERE fact_id=? AND deleted=1",
                (fact_id,),
            )
            self._conn.commit()
            self._invalidate_hrr_cache(fact_id)
        return cur.rowcount > 0

    def purge_facts(self, dry_run: bool = True) -> dict:
        """Purge low-value facts: >90d old, low trust <0.3, low importance <0.5.

        Returns stats about what would be / was deleted.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        with self._lock:
            candidates = self._conn.execute(
                """SELECT fact_id, content, trust_score, importance, created_at FROM facts
                   WHERE (deleted IS NULL OR deleted = 0)
                     AND created_at < ?
                     AND trust_score < 0.3
                     AND importance < 0.5""",
                (cutoff,),
            ).fetchall()

            if dry_run:
                return {
                    "action": "dry_run",
                    "candidates": len(candidates),
                    "detail": [dict(r) for r in candidates[:10]],
                }

            count = 0
            for row in candidates:
                self.soft_delete_fact(row["fact_id"], reason="auto_purge")
                count += 1
            self._conn.commit()
            return {"action": "purged", "count": count}

    def evict_stale(
        self,
        min_trust: float = 0.1,
        max_days: int = 30,
    ) -> int:
        """Soft-delete stale facts with low trust and old retrieval age.

        Evicts facts where:
        - ``trust_score < min_trust`` AND
        - ``last_retrieved_at`` is more than ``max_days`` ago AND
        - fact is not already deleted

        Also evicts facts that were never retrieved (``last_retrieved_at IS NULL``)
        if created more than 7 days ago.

        Args:
            min_trust: Minimum trust score threshold (default: 0.1).
            max_days: Maximum days since last retrieval (default: 30).

        Returns:
            Number of facts soft-deleted.
        """
        with self._lock:
            # Condition 1: retrieved facts that are stale
            cursor1 = self._conn.execute(
                """UPDATE facts SET
                       deleted = 1,
                       deleted_reason = 'eviction: trust=' || ROUND(trust_score, 3)
                           || ' last_retrieved=' || COALESCE(last_retrieved_at, 'never'),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE (deleted IS NULL OR deleted = 0)
                     AND trust_score < ?
                     AND last_retrieved_at IS NOT NULL
                     AND julianday('now') - julianday(last_retrieved_at) > ?""",
                (min_trust, max_days),
            )
            count1 = cursor1.rowcount

            # Condition 2: never-retrieved facts older than 7 days
            cursor2 = self._conn.execute(
                """UPDATE facts SET
                       deleted = 1,
                       deleted_reason = 'eviction: never retrieved, trust='
                           || ROUND(trust_score, 3),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE (deleted IS NULL OR deleted = 0)
                     AND trust_score < ?
                     AND last_retrieved_at IS NULL
                     AND created_at < datetime('now', '-7 days')""",
                (min_trust,),
            )
            count2 = cursor2.rowcount

            self._conn.commit()

        total = count1 + count2
        if total:
            logger.info("Evicted %d stale facts (retrieved=%d, never_retrieved=%d)",
                        total, count1, count2)
        return total

    def _reinforce_facts(self, fact_ids: list[int]) -> None:
        """Boost trust_score and increment retrieval_count for retrieved facts.

        Each retrieval gives a small trust boost (0.01), capping at 1.0.
        Called internally after search to implement the retrieval feedback loop.
        """
        if not fact_ids:
            return
        placeholders = ",".join("?" for _ in fact_ids)
        self._conn.execute(
            f"""UPDATE facts SET
                    retrieval_count = retrieval_count + 1,
                    trust_score = MIN(1.0, ROUND(trust_score + 0.01, 4))
                WHERE fact_id IN ({placeholders})""",
            fact_ids,
        )
        self._conn.commit()

    def search_facts(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
    ) -> list[dict]:
        """Full-text search via FTS5.

        """
        with self._lock:
            try:
                sql = """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                                f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count,
                                f.importance, f.session_id
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?
                         ORDER BY fts.rank
                         LIMIT ?"""
                params: list = [query, limit]
                if exclude_deleted:
                    sql = sql.replace("WHERE", "WHERE (f.deleted IS NULL OR f.deleted = 0) AND")
                rows = self._conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                return []

    def search_by_metadata(
        self,
        what: Optional[str] = None,
        why: Optional[str] = None,
        where_text: Optional[str] = None,
        learned: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search facts by structured metadata fields.

        Builds SQL WHERE clauses for each non-None field using
        ``LIKE '%value%'`` for partial matching. All non-None fields
        are combined with AND.

        Args:
            what: Filter by ``what`` field (partial match).
            why: Filter by ``why`` field (partial match).
            where_text: Filter by ``where_text`` field (partial match).
            learned: Filter by ``learned`` field (partial match).
            limit: Max results (default: 10).

        Returns:
            List of fact dicts matching all provided filters.
        """
        conditions: list[str] = ["(f.deleted IS NULL OR f.deleted = 0)"]
        params: list = []

        field_map = {
            "what": what,
            "why": why,
            "where_text": where_text,
            "learned": learned,
        }

        for col, val in field_map.items():
            if val is not None:
                conditions.append(f"f.{col} LIKE ?")
                params.append(f"%{val}%")

        with self._lock:
            w = " AND ".join(conditions)
            rows = self._conn.execute(
                f"SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score, "
                f"f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count, "
                f"f.importance, f.session_id, f.what, f.why, f.where_text, f.learned "
                f"FROM facts f WHERE {w} ORDER BY f.trust_score DESC LIMIT ?",
                params + [limit],
            ).fetchall()

        return [dict(r) for r in rows]

    def search_by_vector(
        self,
        query_vector: bytes,
        limit: int = 10,
        min_trust: float = 0.0,
        category: str = "",
        project: str = "",
    ) -> list[dict]:
        """Search facts by embedding vector (cosine similarity).

        SQL pre-filter narrows candidates; Python ``struct.unpack`` decodes
        float32 arrays and computes cosine similarity in a loop.

        Retrieved facts get a small trust boost (retrieval feedback loop).

        Returns list of fact dicts sorted by cosine similarity descending.
        """
        with self._lock:
            conditions = ["(f.deleted IS NULL OR f.deleted = 0)", "f.embedding IS NOT NULL"]
            params: list = []
            if min_trust > 0:
                conditions.append("f.trust_score >= ?")
                params.append(min_trust)
            if category:
                conditions.append("f.category = ?")
                params.append(category)
            if project:
                conditions.append("f.project = ?")
                params.append(project)

            w = " AND ".join(conditions)
            rows = self._conn.execute(
                f"SELECT fact_id, content, embedding, trust_score, category, project "
                f"FROM facts f WHERE {w}",
                params,
            ).fetchall()

        n_floats = len(query_vector) // 4
        try:
            q = struct.unpack(f"{n_floats}f", query_vector)
        except struct.error:
            return []

        norm_q = sum(a * a for a in q) ** 0.5
        if norm_q == 0:
            return []

        scored: list[tuple[float, dict]] = []
        for r in rows:
            blob = r["embedding"]
            if not blob:
                continue
            try:
                v = struct.unpack(f"{len(blob) // 4}f", blob)
            except struct.error:
                continue
            dot = sum(a * b for a, b in zip(q, v))
            norm_v = sum(b * b for b in v) ** 0.5
            sim = dot / (norm_q * norm_v) if norm_v > 0 else 0.0
            d = dict(r)
            d.pop("embedding", None)
            scored.append((sim, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [d for _, d in scored[:limit]]
        # Reinforce retrieved facts
        with self._lock:
            self._reinforce_facts([r["fact_id"] for r in results])
        return results

    def update_fact(self, fact_id: int, **kwargs) -> bool:
        """Update fact fields.

        Allowed keys: ``content``, ``category``, ``tags``, ``trust_score``,
        ``importance``, ``project``.

        Args:
            fact_id: ID of the fact to update.
            **kwargs: Field-value pairs to update.

        Returns:
            True if the fact was updated, False if no valid fields given.
        """
        allowed = {"content", "category", "tags", "trust_score", "importance", "project"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k}=?" for k in updates)
        vals = list(updates.values()) + [fact_id]
        with self._lock:
            self._conn.execute(f"UPDATE facts SET {set_clause} WHERE fact_id=?", vals)
            self._conn.commit()
            self._invalidate_hrr_cache(fact_id)
        return True

    def remove_fact(self, fact_id: int) -> bool:
        """Permanently delete a fact from the database.

        This operation cannot be undone. Consider ``soft_delete_fact``
        for reversible deletion.

        Args:
            fact_id: ID of the fact to permanently delete.

        Returns:
            True if the fact was deleted.
        """
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE fact_id=?", (fact_id,))
            self._conn.commit()
            self._invalidate_hrr_cache(fact_id)
        return True

    def get_fact(self, fact_id: int) -> Optional[dict]:
        """Get a single fact by ID.

        Args:
            fact_id: ID of the fact to retrieve.

        Returns:
            Fact dict (excluding ``hrr_vector`` and ``embedding`` blobs),
            or None if not found.
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("hrr_vector", None)
        d.pop("embedding", None)
        return d

    def get_fact_full(self, fact_id: int) -> Optional[dict]:
        """Alias for ``get_fact`` — returns full fact content with all fields.

        Args:
            fact_id: ID of the fact to retrieve.

        Returns:
            Full fact dict (excluding large blobs), or None if not found.
        """
        return self.get_fact(fact_id)

    def list_facts(
        self,
        category: str = "",
        project: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """List facts with optional filters. Returns a flat list of fact dicts."""
        with self._lock:
            where = ["(deleted IS NULL OR deleted = 0)"]
            params: list = []
            if category:
                where.append("category = ?")
                params.append(category)
            if project:
                where.append("project = ?")
                params.append(project)
            w = " AND ".join(where)

            rows = self._conn.execute(
                f"SELECT fact_id, content, category, tags, trust_score, project, "
                f"created_at, updated_at, topic_key, revision_count, importance, session_id "
                f"FROM facts WHERE {w} ORDER BY trust_score DESC LIMIT ? OFFSET ?",
                params + [limit, offset],
            ).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Entities
    # ------------------------------------------------------------------

    def _ensure_entity(self, fact_id: int, name: str, entity_type: str = "unknown") -> int:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO entities (name, entity_type) VALUES (?, ?)",
                (name.lower(), entity_type),
            )
            row = self._conn.execute(
                "SELECT entity_id FROM entities WHERE name = ?", (name.lower(),)
            ).fetchone()
            if row:
                eid = row["entity_id"]
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, eid),
                )
                return eid
        return 0

    def get_entities(self, fact_id: int) -> list[dict]:
        """Get entities associated with a fact.

        Args:
            fact_id: ID of the fact.

        Returns:
            List of entity dicts with keys ``entity_id``, ``name``, ``entity_type``.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT e.entity_id, e.name, e.entity_type
                   FROM entities e
                   JOIN fact_entities fe ON fe.entity_id = e.entity_id
                   WHERE fe.fact_id = ?""",
                (fact_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def start_session(self, session_id: str, project: str = "", metadata: Optional[dict] = None) -> bool:
        """Start a new session.

        Args:
            session_id: Unique session identifier.
            project: Optional project namespace.
            metadata: Optional dict stored as JSON.

        Returns:
            True on success.
        """
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO sessions (session_id, project, status, metadata)
                   VALUES (?, ?, 'active', ?)""",
                (session_id, project, json.dumps(metadata or {})),
            )
            self._conn.commit()
        return True

    def end_session(self, session_id: str, summary: str = "") -> bool:
        """End an active session.

        Args:
            session_id: Session to end.
            summary: Optional summary of the session.

        Returns:
            True if the session was found and ended, False otherwise.
        """
        with self._lock:
            # Count facts for this session
            fact_count = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE session_id = ? AND (deleted IS NULL OR deleted = 0)",
                (session_id,),
            ).fetchone()[0]
            c = self._conn.execute(
                "UPDATE sessions SET status='ended', ended_at=CURRENT_TIMESTAMP, summary=?, fact_count=? WHERE session_id=?",
                (summary, fact_count, session_id),
            )
            self._conn.commit()
        return c.rowcount > 0

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session details by ID.

        Args:
            session_id: Session identifier.

        Returns:
            Session dict with all columns, or None if not found.
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def generate_session_summary(self, session_id: str) -> dict:
        """Generate a structured summary of a session from its facts.

        Best-effort aggregation: missing sections return empty defaults.

        Args:
            session_id: The session identifier to summarize.

        Returns:
            Dict with keys ``goal`` (str), ``discoveries`` (list[str]),
            ``accomplished`` (list[str]), ``next_steps`` (str).
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT content, category FROM facts "
                "WHERE session_id = ? AND (deleted IS NULL OR deleted = 0)",
                (session_id,),
            ).fetchall()

        goal: str = ""
        discoveries: list[str] = []
        accomplished: list[str] = []
        next_steps: str = ""

        for row in rows:
            content = row["content"]
            category = row["category"]

            # All session facts are "accomplished"
            accomplished.append(content)

            # Goal detection — fact starting with "## Goal"
            if not goal and content.lstrip().upper().startswith("## GOAL"):
                goal = content

            # Next steps detection — fact starting with "## Next Steps" or "Next Steps:"
            if content.lstrip().upper().startswith("## NEXT STEPS"):
                next_steps = content
            elif "Next Steps:" in content or "Next steps:" in content:
                if not next_steps:
                    next_steps = content

            # Discoveries — facts with category discovery or bugfix
            if category in ("discovery", "bugfix"):
                discoveries.append(content)

        return {
            "goal": goal,
            "discoveries": discoveries,
            "accomplished": accomplished,
            "next_steps": next_steps,
        }

    # ------------------------------------------------------------------
    # Relations
    # ------------------------------------------------------------------

    def add_relation(
        self,
        fact_id_a: int,
        fact_id_b: int,
        relation_type: str = "related",
        confidence: float = 0.5,
        judged_by: str = "auto",
    ) -> bool:
        """Record a relation between two facts.

        Args:
            fact_id_a: First fact ID.
            fact_id_b: Second fact ID.
            relation_type: One of ``related``, ``compatible``, ``scoped``,
                ``conflicts_with``, ``supersedes``, ``not_conflict``.
            confidence: Confidence score for the relation (default: 0.5).
            judged_by: Who or what judged the relation (default: "auto").

        Returns:
            True if the relation was inserted, False if it already exists.
        """
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT OR IGNORE INTO fact_relations
                       (fact_id_a, fact_id_b, relation_type, confidence, judged_by)
                       VALUES (?, ?, ?, ?, ?)""",
                    (fact_id_a, fact_id_b, relation_type, confidence, judged_by),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_relations(self, fact_id: int) -> list[dict]:
        """Get all relations for a fact.

        Args:
            fact_id: Fact ID to look up.

        Returns:
            List of relation dicts with the other fact as ``other_fact_id``.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.relation_id,
                              CASE WHEN r.fact_id_a = ? THEN r.fact_id_b ELSE r.fact_id_a END AS other_fact_id,
                              r.relation_type, r.confidence, r.judged_by, r.created_at
                       FROM fact_relations r
                       WHERE r.fact_id_a = ? OR r.fact_id_b = ?
                       ORDER BY r.created_at DESC""",
                (fact_id, fact_id, fact_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_contradictions(self, limit: int = 10) -> list[dict]:
        """Get known contradictions between facts.

        Args:
            limit: Max number of contradictions to return (default: 10).

        Returns:
            List of relation dicts with ``content_a`` and ``content_b``.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.*, a.content as content_a, b.content as content_b
                   FROM fact_relations r
                   JOIN facts a ON a.fact_id = r.fact_id_a
                   JOIN facts b ON b.fact_id = r.fact_id_b
                   WHERE r.relation_type IN ('conflicts_with', 'supersedes')
                   ORDER BY r.confidence DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def get_timeline(self, fact_id: int, before: int = 5, after: int = 5) -> dict:
        """Get chronological context around a fact.

        Args:
            fact_id: Anchor fact ID.
            before: Number of preceding facts to include (default: 5).
            after: Number of subsequent facts to include (default: 5).

        Returns:
            Dict with keys ``fact``, ``before``, ``after``.
        """
        with self._lock:
            anchor = self._conn.execute(
                "SELECT fact_id, content, session_id FROM facts WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if not anchor:
                return {"fact": None, "before": [], "after": []}

            session_id = anchor["session_id"] or ""
            b4: list = []
            aft: list = []

            if session_id:
                try:
                    b4 = self._conn.execute(
                        "SELECT fact_id, content, category, tags, trust_score, created_at FROM facts "
                        "WHERE fact_id < ? AND session_id = ? ORDER BY fact_id DESC LIMIT ?",
                        (fact_id, session_id, before),
                    ).fetchall()
                    aft = self._conn.execute(
                        "SELECT fact_id, content, category, tags, trust_score, created_at FROM facts "
                        "WHERE fact_id > ? AND session_id = ? ORDER BY fact_id ASC LIMIT ?",
                        (fact_id, session_id, after),
                    ).fetchall()
                except Exception:
                    pass

        return {
            "fact": dict(anchor),
            "before": [dict(r) for r in b4],
            "after": [dict(r) for r in aft],
        }

    # ------------------------------------------------------------------
    # Backward-compatible API aliases
    # ------------------------------------------------------------------

    def session_start(self, session_id: str, project: str = "", metadata: Optional[dict] = None) -> dict:
        """Alias for ``start_session`` — returns enriched dict."""
        prior = 0
        if project:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE project = ?", (project,)
            ).fetchone()
            prior = row[0] if row else 0
        self.start_session(session_id, project, metadata)
        # Include facts matching project OR global facts (no project)
        if project:
            top_rows = self._conn.execute(
                "SELECT fact_id, content, trust_score FROM facts "
                "WHERE (deleted IS NULL OR deleted = 0) AND (project = ? OR project = '') "
                "ORDER BY trust_score DESC LIMIT 5",
                (project,),
            ).fetchall()
        else:
            top_rows = self._conn.execute(
                "SELECT fact_id, content, trust_score FROM facts "
                "WHERE (deleted IS NULL OR deleted = 0) "
                "ORDER BY trust_score DESC LIMIT 5",
            ).fetchall()
        return {
            "session_id": session_id,
            "prior_session_count": prior,
            "top_facts": [dict(r) for r in top_rows],
        }

    def session_end(self, session_id: str, summary: str = "") -> bool:
        """Alias for ``end_session``."""
        return self.end_session(session_id, summary)

    def timeline(self, fact_id: int, before: int = 5, after: int = 5) -> dict:
        """Alias for ``get_timeline`` with error-raising behavior."""
        result = self.get_timeline(fact_id, before, after)
        if not result["fact"]:
            raise KeyError(f"fact_id {fact_id} not found")
        if not result["fact"].get("session_id"):
            raise ValueError("no session association")
        return result

    def judge_relation(
        self,
        fact_id_a: int,
        fact_id_b: int,
        relation_type: str = "related",
        confidence: float = 0.5,
        judged_by: str = "auto",
    ) -> dict:
        """Alias for ``add_relation`` — returns enriched dict.

        If a relation already exists between the two facts, it is UPDATEd
        and ``updated`` is set to ``True``.
        """
        if relation_type not in ("related", "compatible", "scoped", "conflicts_with", "supersedes", "not_conflict"):
            raise ValueError(f"Invalid relation_type: {relation_type}")
        # Verify facts exist
        for fid in (fact_id_a, fact_id_b):
            row = self._conn.execute(
                "SELECT 1 FROM facts WHERE fact_id = ?", (fid,)
            ).fetchone()
            if not row:
                raise KeyError(f"fact_id {fid} not found")

        with self._lock:
            # Check if relation already exists
            existing = self._conn.execute(
                "SELECT relation_id FROM fact_relations WHERE fact_id_a = ? AND fact_id_b = ?",
                (fact_id_a, fact_id_b),
            ).fetchone()

            if existing:
                # Update existing
                self._conn.execute(
                    """UPDATE fact_relations SET relation_type = ?, confidence = ?, judged_by = ?
                       WHERE fact_id_a = ? AND fact_id_b = ?""",
                    (relation_type, confidence, judged_by, fact_id_a, fact_id_b),
                )
                self._conn.commit()
                return {
                    "relation_type": relation_type,
                    "confidence": confidence,
                    "updated": True,
                    "relation_id": existing["relation_id"],
                }
            else:
                # Insert new
                self._conn.execute(
                    """INSERT OR IGNORE INTO fact_relations
                       (fact_id_a, fact_id_b, relation_type, confidence, judged_by)
                       VALUES (?, ?, ?, ?, ?)""",
                    (fact_id_a, fact_id_b, relation_type, confidence, judged_by),
                )
                self._conn.commit()
                row = self._conn.execute(
                    "SELECT relation_id FROM fact_relations WHERE fact_id_a = ? AND fact_id_b = ?",
                    (fact_id_a, fact_id_b),
                ).fetchone()
                return {
                    "relation_type": relation_type,
                    "confidence": confidence,
                    "updated": False,
                    "relation_id": row["relation_id"] if row else 0,
                }

    def get_recent_sessions(self, project: str = "", limit: int = 10) -> list[dict]:
        """Alias for ``list_sessions``."""
        return self.list_sessions(project, limit)

    def search_facts(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
    ) -> list[dict]:
        """Alias for ``search``."""
        return self.search(query, limit=limit, exclude_deleted=exclude_deleted, project=project)

    @staticmethod
    def _rrf_merge(
        stream_a: list[dict],
        stream_b: list[dict],
        limit: int,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of two ranked streams.

        Args:
            stream_a: First ranked list (must have ``fact_id`` key).
            stream_b: Second ranked list.
            limit: Max items to return.
            k: RRF constant (default 60).

        Returns:
            List of merged dicts with a ``score`` key.
        """
        if not stream_a and not stream_b:
            return []
        if not stream_b:
            result = []
            for rank, item in enumerate(stream_a):
                d = dict(item)
                d["score"] = 1.0 / (k + rank + 1)
                result.append(d)
            return result[:limit]
        if not stream_a:
            result = []
            for rank, item in enumerate(stream_b):
                d = dict(item)
                d["score"] = 1.0 / (k + rank + 1)
                result.append(d)
            return result[:limit]

        scores: dict[int, float] = {}
        items: dict[int, dict] = {}

        for rank, item in enumerate(stream_a):
            fid = item.get("fact_id")
            if fid is not None:
                scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
                items.setdefault(fid, item)

        for rank, item in enumerate(stream_b):
            fid = item.get("fact_id")
            if fid is not None:
                scores[fid] = scores.get(fid, 0) + 1.0 / (k + rank + 1)
                items.setdefault(fid, item)

        ranked = sorted(scores.keys(), key=lambda fid: scores[fid], reverse=True)
        result = []
        for fid in ranked[:limit]:
            d = dict(items[fid])
            d["score"] = scores[fid]
            result.append(d)
        return result

    def search(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
    ) -> list[dict]:
        """Hybrid search: FTS5 + optional embedding vector search fused via RRF.

        Always returns FTS5 results. If an embedding provider is configured
        (non-NoopProvider), the query is embedded and vector search results
        are fused via Reciprocal Rank Fusion.

        Args:
            query: Search text.
            limit: Max results.
            exclude_deleted: Whether to exclude soft-deleted facts.
            project: Optional project filter.

        Returns list of dicts sorted by combined relevance score (``score`` key).
        """
        with self._lock:
            try:
                safe_query = _sanitize_fts5(query)
                sql = """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                                f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count,
                                f.importance, f.session_id
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?"""
                params: list = [safe_query]
                conditions: list[str] = []
                if exclude_deleted:
                    conditions.append("(f.deleted IS NULL OR f.deleted = 0)")
                if project:
                    conditions.append("f.project = ?")
                    params.append(project)
                if conditions:
                    sql += " AND " + " AND ".join(conditions)
                sql += " ORDER BY fts.rank LIMIT ?"
                params.append(limit * 2)  # fetch extra for RRF headroom
                rows = self._conn.execute(sql, params).fetchall()
                fts_results = [dict(r) for r in rows]
            except Exception:
                fts_results = []

            # Vector stream via embedding (optional)
            emb_results: list[dict] = []
            if not isinstance(self._embedding_provider, NoopProvider):
                try:
                    q_vec = self._embedding_provider.embed_query(query)
                    top_ids = self._search_by_embedding(q_vec, k=limit * 2)
                    if top_ids:
                            placeholders = ",".join("?" for _ in top_ids)
                            with self._lock:
                                rows = self._conn.execute(
                                    f"""SELECT fact_id, content, category, tags,
                                               trust_score, created_at, updated_at,
                                               project, topic_key, revision_count,
                                               importance, session_id
                                        FROM facts
                                        WHERE fact_id IN ({placeholders})
                                        AND (deleted IS NULL OR deleted = 0)""",
                                    top_ids,
                                ).fetchall()
                                # Re-sort by the embedding rank order
                                id_order = {fid: i for i, fid in enumerate(top_ids)}
                                sorted_rows = sorted(
                                    rows, key=lambda r: id_order.get(r["fact_id"], 999)
                                )
                                emb_results = [dict(r) for r in sorted_rows]
                except Exception:
                    logger.exception("Embedding search failed")

            # RRF merge
            merged = self._rrf_merge(fts_results, emb_results, limit=limit, k=60)

            # Progressive disclosure: add summary (first 200 chars) to each result
            for item in merged:
                if "content" in item and "summary" not in item:
                    item["summary"] = item["content"][:200]

            # Retrieval feedback loop — reinforce returned facts
            if merged:
                self._reinforce_facts([r["fact_id"] for r in merged])

            # Track last_retrieved_at for eviction
            if merged:
                fids = [r["fact_id"] for r in merged]
                placeholders = ",".join("?" for _ in fids)
                self._conn.execute(
                    f"UPDATE facts SET last_retrieved_at = CURRENT_TIMESTAMP "
                    f"WHERE fact_id IN ({placeholders})",
                    fids,
                )
                self._conn.commit()

            return merged

    def list_sessions(self, project: str = "", limit: int = 10) -> list[dict]:
        """Get recent ended sessions, newest first."""
        with self._lock:
            where = ["status = 'ended'"]
            params: list = []
            if project:
                where.append("project = ?")
                params.append(project)
            w = " AND ".join(where)
            rows = self._conn.execute(
                f"SELECT session_id, project, summary, fact_count, started_at, ended_at "
                f"FROM sessions WHERE {w} ORDER BY ended_at DESC, session_id DESC LIMIT ?",
                params + [limit],
            ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            if d.get("summary") and len(d["summary"]) > 200:
                d["summary"] = d["summary"][:200]
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dict with keys ``fact_count``, ``session_count``,
            ``relation_count``, ``extraction_count``, ``active_sessions``.
        """
        with self._lock:
            facts = self._conn.execute("SELECT COUNT(*) FROM facts WHERE (deleted IS NULL OR deleted = 0)").fetchone()[0]
            sessions = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            relations = 0
            extractions = 0
            active = 0
            try:
                relations = self._conn.execute("SELECT COUNT(*) FROM fact_relations").fetchone()[0]
                extractions = self._conn.execute("SELECT COUNT(*) FROM extractions").fetchone()[0]
                active = self._conn.execute("SELECT COUNT(*) FROM sessions WHERE status='active'").fetchone()[0]
            except Exception:
                pass
        return {
            "fact_count": facts,
            "session_count": sessions,
            "relation_count": relations,
            "extraction_count": extractions,
            "active_sessions": active,
        }

    def projects(self) -> list[str]:
        """List distinct project names that have facts.

        Returns:
            Sorted list of non-empty project names.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT project FROM facts WHERE project != '' AND (deleted IS NULL OR deleted = 0) ORDER BY project"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_memory(self, path: str) -> dict:
        """Export all memory data to a JSON file.

        Includes all active facts, sessions, fact relations, and turn buffer
        entries. HRR vectors and embeddings are excluded from the dump.

        Args:
            path: File path for the JSON export.

        Returns:
            Stats dict with counts of exported items.
        """
        with self._lock:
            facts = self._conn.execute(
                "SELECT fact_id, content, category, tags, trust_score, importance, "
                "project, session_id, topic_key, revision_count, retrieval_count, "
                "consolidated, deleted, deleted_reason, created_at, updated_at "
                "FROM facts ORDER BY fact_id"
            ).fetchall()

            sessions = self._conn.execute(
                "SELECT session_id, project, status, fact_count, summary, "
                "metadata, started_at, ended_at FROM sessions ORDER BY session_id"
            ).fetchall()

            relations = self._conn.execute(
                "SELECT relation_id, fact_id_a, fact_id_b, relation_type, "
                "confidence, judged_by, created_at FROM fact_relations ORDER BY relation_id"
            ).fetchall()

            turns = self._conn.execute(
                "SELECT turn_id, session_id, role, content, meaningful, created_at "
                "FROM turn_buffer ORDER BY turn_id"
            ).fetchall()

        data = {
            "version": 1,
            "facts": [dict(r) for r in facts],
            "sessions": [dict(r) for r in sessions],
            "relations": [dict(r) for r in relations],
            "turns": [dict(r) for r in turns],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        return {
            "facts": len(data["facts"]),
            "sessions": len(data["sessions"]),
            "relations": len(data["relations"]),
            "turns": len(data["turns"]),
        }

    def import_memory(self, path: str) -> dict:
        """Import memory data from a JSON file created by ``export_memory``.

        Facts are inserted via ``add_fact`` (respecting content dedup/topic upsert).
        Sessions, relations, and turn buffer entries are inserted directly.

        Args:
            path: File path to the JSON export.

        Returns:
            Stats dict with counts of imported items.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        imported = {"facts": 0, "sessions": 0, "relations": 0, "turns": 0}

        for row in data.get("facts", []):
            self.add_fact(
                content=row["content"],
                category=row.get("category", "general"),
                tags=row.get("tags", ""),
                trust_score=row.get("trust_score", 0.5),
                importance=row.get("importance", 0.5),
                project=row.get("project", ""),
                session_id=row.get("session_id", ""),
                topic_key=row.get("topic_key", ""),
            )
            imported["facts"] += 1

        with self._lock:
            for row in data.get("sessions", []):
                self._conn.execute(
                    """INSERT OR IGNORE INTO sessions
                       (session_id, project, status, fact_count, summary, metadata, started_at, ended_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        row["session_id"], row.get("project", ""),
                        row.get("status", "ended"), row.get("fact_count", 0),
                        row.get("summary", ""), row.get("metadata", "{}"),
                        row.get("started_at"), row.get("ended_at"),
                    ),
                )
                imported["sessions"] += 1

            for row in data.get("relations", []):
                self._conn.execute(
                    """INSERT OR IGNORE INTO fact_relations
                       (fact_id_a, fact_id_b, relation_type, confidence, judged_by, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        row["fact_id_a"], row["fact_id_b"], row["relation_type"],
                        row.get("confidence", 1.0), row.get("judged_by", "import"),
                        row.get("created_at"),
                    ),
                )
                imported["relations"] += 1

            for row in data.get("turns", []):
                self._conn.execute(
                    """INSERT OR IGNORE INTO turn_buffer
                       (session_id, role, content, meaningful, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        row["session_id"], row["role"], row["content"],
                        row.get("meaningful", 0), row.get("created_at"),
                    ),
                )
                imported["turns"] += 1

            self._conn.commit()

        return imported

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the store and release resources.

        Stops the HRR flush thread and closes the database connection.
        Call this when done to avoid resource leaks.
        """
        self._hrr_flush_stop.set()
        self._signal_flush()
        if self._hrr_flush_thread and self._hrr_flush_thread.is_alive():
            self._hrr_flush_thread.join(timeout=3)
        self._conn.close()
