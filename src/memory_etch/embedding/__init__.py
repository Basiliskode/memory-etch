"""Embedding provider abstraction layer for memory-etch.

Provides the ``EmbeddingProvider`` ABC and a ``NoopProvider`` default
implementation. Concrete providers live in sibling modules.
"""

from abc import ABC, abstractmethod

__all__ = [
    "EmbeddingProvider",
    "NoopProvider",
    "FastembedProvider",
    "OllamaProvider",
]


def __getattr__(name: str):
    """Lazy exports keep optional embedding dependencies out of base imports."""
    if name == "FastembedProvider":
        from .fastembed_provider import FastembedProvider

        return FastembedProvider
    if name == "OllamaProvider":
        from .ollama_provider import OllamaProvider

        return OllamaProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers.

    Subclasses must implement ``embed()`` and ``close()``. The
    ``dimension`` property reports the output vector size.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of embeddings produced by this provider."""
        ...

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into vector representations. May be batched."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Default: delegates to embed().

        Override for BGE-style instruction-tuned models that need
        a query prefix (e.g. 'Represent this sentence for searching
        relevant passages: ').
        """
        return self.embed([text])[0]

    def close(self) -> None:
        """Release resources. Default no-op."""
        return None


class NoopProvider(EmbeddingProvider):
    """Embedding provider that always fails — FTS5-only mode.

    Used as the default when no embedding provider is configured.
    Keeps the hot path dependency-free.
    """

    @property
    def dimension(self) -> int:
        raise NotImplementedError("NoopProvider has no dimension")

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(
            "No embedding provider configured. "
            "Install with: pip install memory-etch[embeddings]"
        )

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError(
            "No embedding provider configured. "
            "Install with: pip install memory-etch[embeddings]"
        )
