"""HRR vector encoding, flush thread, and cache for EtchStore.

Extracted from ``EtchStore`` (store.py).  Module-level functions receive
``store`` (the EtchStore instance) as first argument.
"""

import logging
import threading

# Use absolute import to avoid circular import:
#   store/__init__.py → _hrr.py → memory_etch.hrr
# The hrr module is loaded as ``memory_etch.hrr`` immediately.
import memory_etch.hrr as hrr  # noqa: E402

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# HRR flush thread
# ------------------------------------------------------------------


def _start_hrr_flush(store) -> None:
    """Start the background HRR flush daemon thread (if NumPy available)."""
    if not hrr.HAS_NUMPY:
        logger.info("NumPy not available — HRR vectors disabled")
        return

    def _flush_worker():
        while not store._hrr_flush_stop.is_set():
            store._hrr_flush_signal.wait(timeout=5)
            store._hrr_flush_signal.clear()
            if store._hrr_flush_stop.is_set():
                break
            _flush_pending_hrr_batch(store)

    store._hrr_flush_thread = threading.Thread(target=_flush_worker, daemon=True)
    store._hrr_flush_thread.start()


def _signal_flush(store) -> None:
    """Signal the background HRR flush thread to process pending vectors."""
    store._hrr_flush_signal.set()


def _flush_pending_hrr_batch(store) -> None:
    """Snapshot pending list and encode under lock."""
    with store._lock:
        batch = store._pending_hrr.copy()
        store._pending_hrr.clear()
        if not batch:
            return

    try:
        dim = _get_effective_hrr_dim(store)
        encoded = [
            (fact_id, hrr.phases_to_bytes(hrr.encode_text(content, dim)))
            for fact_id, content in batch
        ]
        with store._lock:
            for fact_id, blob in encoded:
                store._conn.execute(
                    "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
                    (blob, fact_id),
                )
                _invalidate_hrr_cache(store, fact_id)
            store._conn.commit()
    except Exception:
        logger.exception("HRR flush failed")
        # Re-queue on failure
        with store._lock:
            try:
                store._conn.rollback()
            except Exception:
                logger.exception("HRR rollback failed")
            store._pending_hrr.extend(batch)


def _get_effective_hrr_dim(store) -> int:
    """Detect HRR dim from existing vectors, or return default."""
    with store._lock:
        row = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE hrr_vector IS NOT NULL LIMIT 1"
        ).fetchone()
    if row and row["hrr_vector"]:
        try:
            vec = hrr.bytes_to_phases(row["hrr_vector"])
            return len(vec)
        except Exception:
            pass
    return store._hrr_dim


def get_effective_hrr_dim(store) -> int:
    """Return the HRR dimension currently used by this store.

    Existing databases may contain HRR vectors created with a dimension
    different from the constructor default. Retrieval code should call this
    function instead of assuming its own default dimension.
    """
    return _get_effective_hrr_dim(store)


def compute_hrr_batch(store) -> None:
    """Flush pending HRR vectors synchronously.

    Public compatibility wrapper for integrations that need to force HRR
    computation before searching or shutting down.
    """
    _flush_pending_hrr_batch(store)


# ------------------------------------------------------------------
# HRR cache
# ------------------------------------------------------------------


def _get_hrr_cached(store, fact_id: int):
    """Get cached HRR vector, or decode from DB."""
    if fact_id in store._hrr_vector_cache:
        return store._hrr_vector_cache[fact_id]
    with store._lock:
        row = store._conn.execute(
            "SELECT hrr_vector FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
    if not row or not row["hrr_vector"]:
        return None
    vec = hrr.bytes_to_phases(row["hrr_vector"])
    if len(store._hrr_vector_cache) < store._hrr_cache_max:
        store._hrr_vector_cache[fact_id] = vec
    return vec


def _invalidate_hrr_cache(store, fact_id: int) -> None:
    """Remove a fact from the HRR vector cache."""
    store._hrr_vector_cache.pop(fact_id, None)
