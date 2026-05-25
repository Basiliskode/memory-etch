"""FastembedProvider — wraps fastembed.TextEmbedding for local ONNX inference.

Uses ``BAAI/bge-small-en-v1.5`` (384-dim) by default. The ``fastembed``
package is imported lazily (inside the constructor) so that merely
importing this module does not trigger the download or load.
"""

import logging
from typing import Optional

from . import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastembedProvider(EmbeddingProvider):
    """Embedding provider backed by fastembed's ONNX runtime.

    Args:
        model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5).
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Optional["TextEmbedding"] = None
        self._dimension: Optional[int] = None

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "fastembed is required for FastembedProvider. "
                "Install: pip install memento[embeddings]"
            ) from None

        self._model = TextEmbedding(model_name=self._model_name)
        # Determine dimension from the model's output
        sample = list(self._model.embed(["dim"]))
        if sample:
            self._dimension = len(sample[0])
        else:
            self._dimension = 384  # safe fallback for bge-small-en-v1.5

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        self._lazy_load()
        assert self._dimension is not None
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed documents with L2 normalization.

        fastembed already returns L2-normalized vectors for
        BGE models by default.
        """
        self._lazy_load()
        assert self._model is not None
        results: list[list[float]] = []
        for vec in self._model.embed(texts):
            results.append(vec.tolist() if hasattr(vec, "tolist") else list(vec))
        return results

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the BGE instruction prefix.

        BGE models are trained with different prefixes for queries
        vs documents. Without the prefix, retrieval quality degrades
        significantly because the model cannot distinguish direction.
        """
        self._lazy_load()
        assert self._model is not None
        prefixed = f"Represent this sentence for searching relevant passages: {text}"
        vec = next(self._model.embed([prefixed]))
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def close(self) -> None:
        self._model = None
