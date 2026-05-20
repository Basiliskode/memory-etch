"""Integration tests: EtchStore with FastembedProvider for embedding storage and search."""

import struct
import pytest


class TestStoreWithFastembed:
    """Task 4.4: Integration — FastembedProvider with :memory: store."""

    def test_add_fact_stores_embedding_blob(self):
        """add_fact with FastembedProvider stores non-NULL embedding BLOB."""
        from memory_etch import EtchStore
        from memory_etch.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            store = EtchStore(":memory:", auto_migrate=True, embedding_provider=provider)
            fid = store.add_fact("Python is a programming language")
            fact = store.get_fact(fid)
            # embedding should not be in the dict (get_fact pops it)
            assert "embedding" not in fact

            # Direct DB check
            row = store._conn.execute(
                "SELECT embedding FROM facts WHERE fact_id=?", (fid,)
            ).fetchone()
            assert row is not None
            blob = row["embedding"]
            assert blob is not None, "embedding BLOB should not be NULL"
            # Decode: 384 floats * 4 bytes = 1536 bytes
            floats = struct.unpack(f"{384}f", blob)
            assert len(floats) == 384
            # Verify L2 norm ~= 1.0
            norm = sum(v * v for v in floats) ** 0.5
            assert abs(norm - 1.0) < 1e-3, f"Expected L2 norm ~1.0, got {norm}"
        finally:
            provider.close()
            store.close()

    def test_add_fact_embedding_kwarg_skips_provider(self):
        """Pre-supplied embedding kwarg is stored without provider computation."""
        from memory_etch import EtchStore
        from memory_etch.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            store = EtchStore(":memory:", auto_migrate=True, embedding_provider=provider)
            # Pre-supply an embedding (384 floats packed as bytes)
            pre_supplied = struct.pack(f"{384}f", *([0.5] * 384))
            fid = store.add_fact("Pre-embedded fact", embedding=pre_supplied)
            row = store._conn.execute(
                "SELECT embedding FROM facts WHERE fact_id=?", (fid,)
            ).fetchone()
            blob = row["embedding"]
            floats = struct.unpack(f"{384}f", blob)
            # Should be our pre-supplied values, not normalized
            assert abs(floats[0] - 0.5) < 1e-6
        finally:
            provider.close()
            store.close()

    def test_search_returns_semantic_results(self):
        """search() with FastembedProvider returns RRF-fused semantic results."""
        from memory_etch import EtchStore
        from memory_etch.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            store = EtchStore(":memory:", auto_migrate=True, embedding_provider=provider)
            store.add_fact("Python is a programming language")
            store.add_fact("FastAPI is a web framework")
            store.add_fact("SQLite is a database engine")
            store.add_fact("The sky is blue")
            store.add_fact("Machine learning uses neural networks")

            results = store.search("python programming", limit=10)
            assert len(results) > 0
            assert all("score" in r for r in results)
            # Python-related facts should rank high
            contents = [r["content"] for r in results]
            python_fact = "Python is a programming language"
            assert python_fact in contents, (
                f"Expected '{python_fact}' in results, got {contents}"
            )
        finally:
            provider.close()
            store.close()

    def test_search_with_noop_provider_fallback(self):
        """search() with NoopProvider returns only FTS5 results (no score key)."""
        from memory_etch import EtchStore
        from memory_etch.embedding import NoopProvider

        provider = NoopProvider()
        store = EtchStore(":memory:", auto_migrate=True, embedding_provider=provider)
        try:
            store.add_fact("Python is a programming language")
            store.add_fact("FastAPI is a web framework")

            results = store.search("python", limit=10)
            assert len(results) > 0
        finally:
            store.close()


class TestStoreEmbeddingEdgeCases:
    """Task 4.5, AC-6b: Edge cases for embedding search."""

    def test_search_with_no_embeddings_returns_fts(self):
        """search() with NoopProvider and no embeddings returns FTS5 results."""
        from memory_etch import EtchStore

        store = EtchStore(":memory:", auto_migrate=True)
        try:
            store.add_fact("Python is a programming language")
            results = store.search("python", limit=10)
            assert len(results) > 0
            assert any("Python" in r["content"] for r in results)
        finally:
            store.close()

    def test_search_by_embedding_empty_when_no_embeddings(self):
        """_search_by_embedding returns [] when no embeddings exist."""
        from memory_etch import EtchStore

        store = EtchStore(":memory:", auto_migrate=True)
        try:
            result = store._search_by_embedding([0.1] * 384, k=10)
            assert result == []
        finally:
            store.close()
