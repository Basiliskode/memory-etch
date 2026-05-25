"""BGE-M3 embedding plugin via fastembed.

Importing this module does NOT download any model (lazy load).
First call to ``encode()`` triggers the download via fastembed.

Usage:
    from memento.plugins.bge_m3 import BgeM3Plugin

    plugin = BgeM3Plugin()
    vec = plugin.encode("Some text")  # model downloaded on first call
    assert len(vec) == plugin.dimension  # 1024

Requires ``pip install memento[bge-m3]`` (or ``fastembed>=0.5.0``).
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from fastembed import TextEmbedding
except ImportError as exc:
    raise ImportError(
        "BGE-M3 plugin requires fastembed. Install with: pip install memento[bge-m3]"
    ) from exc


class BgeM3Plugin:
    """BGE-M3 embedding provider wrapping ``fastembed.TextEmbedding``.

    The underlying ONNX model is downloaded on the first ``encode()`` call
    (lazy — module import does NOT trigger a download).

    Attributes:
        dimension: Embedding vector size (1024 for BGE-M3).
    """

    dimension: int = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", **kwargs):
        self._model_name = model_name
        self._model_kwargs = kwargs
        self._model: Optional[TextEmbedding] = None

    def _lazy_load(self) -> TextEmbedding:
        """Initialize the underlying fastembed model (downloaded once)."""
        if self._model is None:
            logger.info("Loading BGE-M3 model (%s) — first call triggers download", self._model_name)
            self._model = TextEmbedding(model_name=self._model_name, **self._model_kwargs)
        return self._model

    def encode(self, text: str) -> list[float]:
        """Encode a single text string into a 1024-dim float vector.

        Downloads the model on first call (if not cached already).

        Args:
            text: Input text to embed.

        Returns:
            A list of 1024 floats.
        """
        model = self._lazy_load()
        # fastembed returns a generator of numpy arrays; convert to list
        vectors = list(model.embed([text]))
        if vectors:
            return vectors[0].tolist()
        return [0.0] * self.dimension

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode multiple texts in one batch.

        Args:
            texts: List of input strings.

        Returns:
            List of 1024-dim float vectors.
        """
        model = self._lazy_load()
        return [v.tolist() for v in model.embed(texts)]


# Convenience alias: encode(texts: list[str]) → list[list[float]]
# Matches the Spec BP‑3 interface. For single-text encoding use BgeM3Plugin().encode().
_plugin_instance = BgeM3Plugin()


def encode(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts.

    Args:
        texts: One or more input strings.

    Returns:
        List of 1024-dim float vectors, one per input text.
    """
    return _plugin_instance.encode_batch(texts)

