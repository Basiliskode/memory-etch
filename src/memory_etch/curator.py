"""Deterministic memory curation engine. SQL-only maintenance operations.

Three operations — decay, archive, prune — that keep the memory store
healthy without any LLM, external service, or new dependency.

Every operation uses ``cursor.rowcount`` for accurate per-statement counts
and is safe to call multiple times (idempotent by design).
"""

import logging
import time
from typing import Optional

from .circuit_breaker import LLMCircuitBreaker

logger = logging.getLogger(__name__)

# Shared module-level circuit breaker for LLM-dependent operations.
# When open, curate() will skip LLM-heavy steps and return early.
_breaker = LLMCircuitBreaker(max_failures=3, cooldown_seconds=60)

_DEFAULT_CONFIG = {
    # Decay
    "decay_interval_days": 7,
    "decay_factor_critical": 0.99,
    "decay_factor_important": 0.97,
    "decay_factor_useful": 0.95,
    "decay_factor_trivial": 0.90,
    # Archive
    "archive_trust_threshold": 0.1,
    "archive_age_days": 90,
    # Prune
    "prune_buffer_age_days": 7,
    # Vacuum
    "vacuum_free_page_pct": 20,
}


class EtchCurator:
    """Deterministic memory maintenance for EtchStore.

    Safe to call ``curate()`` multiple times — every operation is idempotent.
    Zero new dependencies. Zero LLM required.

    Args:
        store: An :class:`~memory_etch.store.EtchStore` instance.
        config: Optional dict overriding any ``_DEFAULT_CONFIG`` key.
    """

    def __init__(self, store, config: Optional[dict] = None):
        self._store = store
        self.config = {**_DEFAULT_CONFIG, **(config or {})}

    # ------------------------------------------------------------------
    # Full pass
    # ------------------------------------------------------------------

    def curate(self) -> dict:
        """Run a full curation pass.

        Operations run in dependency order: decay → archive → prune.
        If the circuit breaker is open (cooldown), curation is skipped and
        a minimal stats dict is returned immediately.

        Returns a stats dict with counts and timing.
        """
        if not _breaker.is_available():
            logger.warning("Circuit breaker open — skipping curation")
            return {
                "decayed": 0,
                "archived": 0,
                "pruned": 0,
                "vacuumed": False,
                "duration_ms": 0,
                "skipped": True,
            }
        t0 = time.time()
        stats = {
            "decayed": self.decay_trust(),
            "archived": self.archive_stale(),
            "pruned": self.prune_buffer(),
            "vacuumed": self._maybe_vacuum(),
        }
        stats["duration_ms"] = int((time.time() - t0) * 1000)
        logger.info("Curation complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Individual operations
    # ------------------------------------------------------------------

    def decay_trust(self) -> int:
        """Reduce ``trust_score`` for facts not updated within the interval.

        Higher ``importance`` facts decay slower. Trust never drops below 0.01.
        ``updated_at`` is intentionally NOT touched so decay is idempotent
        (a fact only decays again after another full interval).
        """
        interval = self.config["decay_interval_days"]
        with self._store._lock:
            cursor = self._store._conn.execute(
                """
                UPDATE facts SET trust_score = MAX(0.01, ROUND(
                    CASE
                        WHEN importance >= 0.9 THEN trust_score * ?
                        WHEN importance >= 0.7 THEN trust_score * ?
                        WHEN importance >= 0.3 THEN trust_score * ?
                        ELSE trust_score * ?
                    END, 4
                ))
                WHERE deleted = 0
                  AND consolidated = 0
                  AND julianday('now') - julianday(updated_at) > ?
                  AND trust_score > 0.01
                """,
                (
                    self.config["decay_factor_critical"],
                    self.config["decay_factor_important"],
                    self.config["decay_factor_useful"],
                    self.config["decay_factor_trivial"],
                    interval,
                ),
            )
            self._store._conn.commit()
            return cursor.rowcount

    def archive_stale(self) -> int:
        """Soft-delete facts past their useful life.

        A fact is archived when its ``trust_score`` is below the threshold
        AND it hasn't been updated in ``archive_age_days``.
        """
        threshold = self.config["archive_trust_threshold"]
        age = self.config["archive_age_days"]
        with self._store._lock:
            cursor = self._store._conn.execute(
                """
                UPDATE facts SET
                    deleted = 1,
                    deleted_reason = 'curator: trust=' || ROUND(trust_score, 3)
                        || ' after ' || CAST(
                            ROUND(julianday('now') - julianday(updated_at)) AS INTEGER
                        ) || ' days',
                    updated_at = CURRENT_TIMESTAMP
                WHERE deleted = 0
                  AND trust_score < ?
                  AND julianday('now') - julianday(updated_at) > ?
                """,
                (threshold, age),
            )
            self._store._conn.commit()
            return cursor.rowcount

    def prune_buffer(self) -> int:
        """Delete ``turn_buffer`` rows older than ``prune_buffer_age_days``."""
        age = self.config["prune_buffer_age_days"]
        with self._store._lock:
            cursor = self._store._conn.execute(
                "DELETE FROM turn_buffer"
                " WHERE created_at < datetime('now', '-' || ? || ' days')",
                (age,),
            )
            self._store._conn.commit()
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_vacuum(self) -> bool:
        """VACUUM if free pages exceed the configured threshold."""
        pct = self.config["vacuum_free_page_pct"]
        freelist = self._store._conn.execute("PRAGMA freelist_count").fetchone()[0]
        total = self._store._conn.execute("PRAGMA page_count").fetchone()[0]
        if total > 0 and (freelist * 100 / total) >= pct:
            self._store._conn.execute("VACUUM")
            return True
        return False
