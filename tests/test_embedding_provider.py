"""Tests for embedding provider ABC, NoopProvider, and concrete providers."""

from unittest.mock import patch

import pytest


class TestEmbeddingProviderABC:
    """Task 1.1, 4.1: ABC contract verification."""

    def test_abc_cannot_be_instantiated_directly(self):
        """EmbeddingProvider ABC raises TypeError on direct instantiation."""
        from memento.embedding import EmbeddingProvider

        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore

    def test_noop_provider_raises_not_implemented(self):
        """NoopProvider.embed() raises NotImplementedError."""
        from memento.embedding import NoopProvider

        provider = NoopProvider()
        with pytest.raises(NotImplementedError) as excinfo:
            provider.embed(["hello"])
        assert "embeddings" in str(excinfo.value).lower()

    def test_noop_provider_close_is_noop(self):
        """NoopProvider.close() does not raise."""
        from memento.embedding import NoopProvider

        provider = NoopProvider()
        provider.close()  # must not raise

    def test_noop_provider_dimension_property(self):
        """NoopProvider dimension raises NotImplementedError."""
        from memento.embedding import NoopProvider

        provider = NoopProvider()
        with pytest.raises(NotImplementedError):
            _ = provider.dimension

    def test_embedding_provider_has_abstract_methods(self):
        """EmbeddingProvider declares embed and dimension as abstract."""
        import abc

        from memento.embedding import EmbeddingProvider

        assert issubclass(EmbeddingProvider, abc.ABC)
        assert "embed" in EmbeddingProvider.__abstractmethods__
        assert "dimension" in EmbeddingProvider.__abstractmethods__

    def test_documented_provider_imports_are_exported_lazily(self):
        """README snippets import concrete providers from memento.embedding."""
        import sys

        sys.modules.pop("fastembed", None)

        from memento.embedding import FastembedProvider, OllamaProvider

        assert FastembedProvider.__name__ == "FastembedProvider"
        assert OllamaProvider.__name__ == "OllamaProvider"
        assert "fastembed" not in sys.modules


class TestNoopProviderDefault:
    """Task 4.1, REQ-2: NoopProvider as default."""

    def test_store_init_without_provider_uses_noop(self):
        """EtchStore without embedding_provider arg uses NoopProvider."""
        from memento import EtchStore
        from memento.embedding import NoopProvider

        store = EtchStore(":memory:", auto_migrate=True)
        try:
            assert isinstance(store._embedding_provider, NoopProvider)
        finally:
            store.close()

    def test_store_init_with_none_uses_noop(self):
        """EtchStore(embedding_provider=None) uses NoopProvider."""
        from memento import EtchStore
        from memento.embedding import NoopProvider

        store = EtchStore(":memory:", auto_migrate=True, embedding_provider=None)
        try:
            assert isinstance(store._embedding_provider, NoopProvider)
        finally:
            store.close()


try:
    import fastembed  # noqa: F401
    _has_fastembed = True
except ImportError:
    _has_fastembed = False


@pytest.mark.skipif(not _has_fastembed, reason="fastembed not installed — optional dependency")
class TestFastembedProvider:
    """Task 1.2, 4.2: FastembedProvider."""

    def test_fastembed_provider_instantiation(self):
        """FastembedProvider can be instantiated with lazy import."""
        from memento.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            assert provider.dimension == 384
        finally:
            provider.close()

    def test_fastembed_embed_returns_list_of_lists(self):
        """FastembedProvider.embed() returns list[list[float]] with correct dim."""
        from memento.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            result = provider.embed(["hello world"])
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], list)
            assert len(result[0]) == 384
            # Check all floats
            for v in result[0]:
                assert isinstance(v, float)
        finally:
            provider.close()

    def test_fastembed_embed_normalized(self):
        """FastembedProvider returns L2-normalized vectors (norm ≈ 1.0)."""
        import math

        from memento.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            result = provider.embed(["hello world"])
            vec = result[0]
            norm = math.sqrt(sum(v * v for v in vec))
            # L2 norm should be approximately 1.0 (allow tiny floating point variation)
            assert abs(norm - 1.0) < 1e-4
        finally:
            provider.close()

    def test_fastembed_multiple_texts(self):
        """FastembedProvider.embed() handles multiple texts."""
        from memento.embedding.fastembed_provider import FastembedProvider

        provider = FastembedProvider()
        try:
            result = provider.embed(["hello", "world", "test"])
            assert isinstance(result, list)
            assert len(result) == 3
            for vec in result:
                assert len(vec) == 384
        finally:
            provider.close()

    def test_fastembed_import_lazy(self):
        """FastembedProvider module can be imported without importing fastembed."""
        import sys

        # Clean up
        for mod in list(sys.modules.keys()):
            if mod == "memento.embedding.fastembed_provider":
                del sys.modules[mod]
            if mod == "fastembed":
                del sys.modules[mod]

        # Import just the provider module — must not import fastembed
        from memento.embedding import fastembed_provider

        assert fastembed_provider is not None
        assert "fastembed" not in sys.modules, (
            "importing the provider module should NOT trigger fastembed import"
        )


class TestOllamaProvider:
    """Task 1.3, 4.3: OllamaProvider."""

    def test_ollama_provider_instantiation(self):
        """OllamaProvider can be instantiated with default params."""
        from memento.embedding.ollama_provider import OllamaProvider

        provider = OllamaProvider()
        try:
            assert provider.dimension == 768
        finally:
            provider.close()

    def test_ollama_provider_custom_config(self):
        """OllamaProvider accepts custom base_url and model."""
        from memento.embedding.ollama_provider import OllamaProvider

        provider = OllamaProvider(
            base_url="http://custom:11434",
            model="mxbai-embed-large",
            dimension=1024,
        )
        try:
            assert provider.dimension == 1024
            assert provider._model == "mxbai-embed-large"
            assert provider._base_url == "http://custom:11434"
        finally:
            provider.close()

    def test_ollama_embed_sends_correct_request(self):
        """OllamaProvider.embed() POSTs to /api/embeddings with correct body."""
        from memento.embedding.ollama_provider import OllamaProvider

        provider = OllamaProvider(base_url="http://test:11434", model="test-model")
        try:
            with patch.object(provider, "_client") as mock_client:
                mock_client.post.return_value.status_code = 200
                mock_client.post.return_value.json.return_value = {
                    "embedding": [0.1] * 768
                }

                result = provider.embed(["hello"])
                assert len(result) == 1
                assert len(result[0]) == 768

                # Verify POST was called correctly
                # (timeout is set at client construction, not per-request)
                mock_client.post.assert_called_once_with(
                    "http://test:11434/api/embeddings",
                    json={"model": "test-model", "prompt": "hello"},
                )
        finally:
            provider.close()

    def test_ollama_embed_connection_error(self):
        """OllamaProvider.embed() raises ConnectionError on connection failure."""
        from memento.embedding.ollama_provider import OllamaProvider

        provider = OllamaProvider(base_url="http://nonexistent:11434")
        try:
            import httpx
            with patch.object(provider, "_client") as mock_client:
                mock_client.post.side_effect = httpx.ConnectError("Connection refused")

                with pytest.raises(ConnectionError):
                    provider.embed(["hello"])
        finally:
            provider.close()

    def test_ollama_bad_status_raises(self):
        """OllamaProvider.embed() raises RuntimeError on bad HTTP status."""
        from memento.embedding.ollama_provider import OllamaProvider

        provider = OllamaProvider(base_url="http://test:11434")
        try:
            with patch.object(provider, "_client") as mock_client:
                mock_client.post.return_value.status_code = 500
                mock_client.post.return_value.text = "Internal Server Error"

                with pytest.raises(RuntimeError) as excinfo:
                    provider.embed(["hello"])
                assert "500" in str(excinfo.value)
        finally:
            provider.close()


class TestPackageExports:
    """Task 1.4, 1.5: Package exports."""

    def test_embedding_subpackage_importable(self):
        """memento.embedding subpackage is importable."""
        from memento import embedding
        assert hasattr(embedding, "EmbeddingProvider")
        assert hasattr(embedding, "NoopProvider")

    def test_top_level_exports(self):
        """EmbeddingProvider and NoopProvider exported from memento top level."""
        import memento
        assert hasattr(memento, "EmbeddingProvider")
        assert hasattr(memento, "NoopProvider")

    def test_fastembed_provider_module_importable(self):
        """FastembedProvider module is importable."""
        from memento.embedding.fastembed_provider import FastembedProvider
        assert FastembedProvider is not None

    def test_ollama_provider_module_importable(self):
        """OllamaProvider module is importable."""
        from memento.embedding.ollama_provider import OllamaProvider
        assert OllamaProvider is not None
