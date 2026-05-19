"""Tests for memory_etch.plugins.bge_m3 — BGE-M3 embedding plugin.

Unit tests verify lazy import and encoding contract without model download.
Integration tests verify end-to-end wiring with EtchRetriever, and skip
gracefully when the model is unavailable.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# =========================================================================
# Helpers
# =========================================================================

_HAS_FASTEMBED = True
try:
    import fastembed  # noqa: F401
except ImportError:
    _HAS_FASTEMBED = False


# =========================================================================
# Unit Tests — Lazy Import & Contract
# =========================================================================


class TestBgeM3LazyImport:
    """Verify import-time behavior — no download, no crash."""

    def test_import_raises_without_fastembed(self):
        """Without fastembed installed, import raises ImportError."""
        if _HAS_FASTEMBED:
            pytest.skip("fastembed is installed — this test requires it to be absent")
        with pytest.raises(ImportError) as excinfo:
            from memory_etch.plugins import bge_m3  # noqa: F401, F811
        assert "pip install memory-etch[bge-m3]" in str(excinfo.value)


@pytest.mark.skipif(not _HAS_FASTEMBED, reason="fastembed not installed — skipping BGE-M3 encoding tests")
class TestBgeM3Contract:
    """Verify encoding contract (requires fastembed installed)."""

    def test_dimension_property(self):
        """BgeM3Plugin.dimension is 1024."""
        from memory_etch.plugins.bge_m3 import BgeM3Plugin

        plugin = BgeM3Plugin()
        assert plugin.dimension == 1024

    def test_encode_returns_1024_dim(self):
        """encode() returns a 1024-element vector."""
        from memory_etch.plugins.bge_m3 import BgeM3Plugin

        plugin = BgeM3Plugin()
        vec = plugin.encode("Hello world")
        assert isinstance(vec, list)
        assert len(vec) == 1024
        assert all(isinstance(v, float) for v in vec)

    def test_import_does_not_download(self):
        """Module import alone does NOT trigger model download."""
        from memory_etch.plugins.bge_m3 import BgeM3Plugin

        p = BgeM3Plugin()
        assert p._model is None  # model not loaded yet

    def test_similar_texts_close_in_vector_space(self):
        """Similar texts produce similar vectors (cosine > 0.5)."""
        from memory_etch.plugins.bge_m3 import BgeM3Plugin

        plugin = BgeM3Plugin()
        v1 = plugin.encode("Python programming language")
        v2 = plugin.encode("Python coding language")
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0
        assert cos > 0.5, f"Expected cos > 0.5, got {cos}"

    def test_different_texts_distant_in_vector_space(self):
        """Unrelated texts have lower cosine than similar texts."""
        from memory_etch.plugins.bge_m3 import BgeM3Plugin

        plugin = BgeM3Plugin()
        v1 = plugin.encode("Python programming language")
        v2 = plugin.encode("Quantum physics theory")
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = sum(a * a for a in v1) ** 0.5
        n2 = sum(b * b for b in v2) ** 0.5
        cos = dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0
        assert cos < 0.8, f"Expected cos < 0.8, got {cos}"


# =========================================================================
# Integration Tests — EtchRetriever + BgeM3Plugin
# =========================================================================


@pytest.mark.skipif(not _HAS_FASTEMBED, reason="fastembed not installed — skipping BGE-M3 integration tests")
class TestBgeM3Integration:
    """End-to-end: compute_embedding callback wired into EtchRetriever."""

    def test_retriever_wired_with_bge_m3(self):
        """EtchRetriever with compute_embedding returns vector-ranked results."""
        from memory_etch.store import EtchStore
        from memory_etch.retrieval import EtchRetriever
        from memory_etch.plugins.bge_m3 import BgeM3Plugin

        plugin = BgeM3Plugin()
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = EtchStore(str(db))
            try:
                # Add facts with different semantic content
                store.add_fact("Python programming for data science")
                store.add_fact("JavaScript frontend development")
                store.add_fact("Cooking Italian pasta recipes")

                retriever = EtchRetriever(store=store, compute_embedding=plugin.encode)
                results = retriever.search("machine learning with Python", limit=3)
                assert len(results) >= 1
                assert "score" in results[0]
                # Python fact should rank high for a Python query
                top = results[0]["content"].lower()
                assert "python" in top, f"Expected Python-related top result, got: {top}"
            finally:
                store.close()

    def test_retriever_fallback_without_embedder(self):
        """Without compute_embedding, search falls back to FTS5."""
        from memory_etch.store import EtchStore
        from memory_etch.retrieval import EtchRetriever

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "test.db"
            store = EtchStore(str(db))
            try:
                store.add_fact("PostgreSQL database connection pool")
                store.add_fact("React frontend state management")

                retriever = EtchRetriever(store=store, compute_embedding=None)
                results = retriever.search("database", limit=3)
                assert len(results) >= 1
                assert "score" in results[0]
            finally:
                store.close()
