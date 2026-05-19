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

logger = logging.getLogger(__name__)

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
    ):
        self._db_path = db_path
        self._hrr_dim = hrr_dim
        self._lock = threading.RLock()

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
            ("importance", "REAL DEFAULT 0.5"),
            ("deleted", "INTEGER DEFAULT 0"),
            ("deleted_reason", "TEXT DEFAULT ''"),
            ("replaced_by", "INTEGER DEFAULT NULL"),
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
    ) -> int:
        """Insert a new fact. Returns fact_id.

        When tags contain ``topic:<name>``, the topic_key is auto-extracted
        and an existing fact with the same key is UPDATEd (topic upsert).
        """
        if trust_score is None:
            trust_score = 0.5
        if importance is None:
            importance = 0.5

        # Auto-extract topic_key from tags if not explicitly provided
        if not topic_key:
            m = re.search(r"(?:^|,)topic:([^,]+)", tags)
            if m:
                topic_key = "topic:" + m.group(1).strip()

        with self._lock:
            # Topic upsert: if a topic_key is set, try to update existing fact
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
                           embedding = COALESCE(?, embedding)
                        WHERE fact_id = ?""",
                        (content, category, tags, trust_score, importance,
                         project, session_id, embedding, eid),
                    )
                    self._conn.commit()
                    self._invalidate_hrr_cache(eid)
                    if hrr.HAS_NUMPY:
                        self._pending_hrr.append((eid, content))
                        self._signal_flush()
                    return eid

            try:
                self._conn.execute(
                    """INSERT OR IGNORE INTO facts
                       (content, category, tags, trust_score, importance, project, session_id, topic_key, embedding)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (content, category, tags, trust_score, importance, project, session_id, topic_key, embedding),
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
                return row["fact_id"] if row else 0

            if fact_id and hrr.HAS_NUMPY:
                self._pending_hrr.append((fact_id, content))
                self._signal_flush()

            if entities:
                for entity_name in entities:
                    self._ensure_entity(fact_id, entity_name)

            return fact_id

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
        """Soft-delete a fact. It remains in DB but is excluded from searches by default."""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE facts SET deleted=1, deleted_reason=? WHERE fact_id=? AND (deleted IS NULL OR deleted=0)",
                (reason, fact_id),
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

    def search_facts(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
    ) -> list[dict]:
        """Full-text search via FTS5."""
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
        return [d for _, d in scored[:limit]]

    def update_fact(self, fact_id: int, **kwargs) -> bool:
        """Update fact fields. Keys can include: content, category, tags, trust_score, importance, project."""
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
        """Permanently delete a fact."""
        with self._lock:
            self._conn.execute("DELETE FROM facts WHERE fact_id=?", (fact_id,))
            self._conn.commit()
            self._invalidate_hrr_cache(fact_id)
        return True

    def get_fact(self, fact_id: int) -> Optional[dict]:
        """Get a single fact by ID."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d.pop("hrr_vector", None)
        d.pop("embedding", None)
        return d

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
        with self._lock:
            self._conn.execute(
                """INSERT OR IGNORE INTO sessions (session_id, project, status, metadata)
                   VALUES (?, ?, 'active', ?)""",
                (session_id, project, json.dumps(metadata or {})),
            )
            self._conn.commit()
        return True

    def end_session(self, session_id: str, summary: str = "") -> bool:
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
        with self._lock:
            row = self._conn.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        return dict(row) if row else None

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

    def search(
        self,
        query: str,
        limit: int = 10,
        exclude_deleted: bool = True,
        project: str = "",
    ) -> list[dict]:
        """Full-text search via FTS5 with optional project filter."""
        with self._lock:
            try:
                sql = """SELECT f.fact_id, f.content, f.category, f.tags, f.trust_score,
                                f.created_at, f.updated_at, f.project, f.topic_key, f.revision_count,
                                f.importance, f.session_id
                         FROM facts f
                         JOIN facts_fts fts ON fts.rowid = f.fact_id
                         WHERE facts_fts MATCH ?"""
                params: list = [query]
                conditions: list[str] = []
                if exclude_deleted:
                    conditions.append("(f.deleted IS NULL OR f.deleted = 0)")
                if project:
                    conditions.append("f.project = ?")
                    params.append(project)
                if conditions:
                    sql += " AND " + " AND ".join(conditions)
                sql += " ORDER BY fts.rank LIMIT ?"
                params.append(limit)
                rows = self._conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except Exception:
                return []

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
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT project FROM facts WHERE project != '' AND (deleted IS NULL OR deleted = 0) ORDER BY project"
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._hrr_flush_stop.set()
        self._signal_flush()
        if self._hrr_flush_thread and self._hrr_flush_thread.is_alive():
            self._hrr_flush_thread.join(timeout=3)
        self._conn.close()
