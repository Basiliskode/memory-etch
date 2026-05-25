"""OllamaProvider — embeds text via Ollama's HTTP API.

Sends POST requests to ``{base_url}/api/embeddings`` with the configured
model.  Uses ``httpx`` for HTTP/2-capable, connection-pooled requests.
"""

import logging
from typing import Optional

import httpx

from . import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_DIM = 768
_REQUEST_TIMEOUT = 30


class OllamaProvider(EmbeddingProvider):
    """Embedding provider backed by an Ollama server.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434).
        model: Model name to use (default: nomic-embed-text).
        dimension: Output dimension (default: 768 for nomic-embed-text).
        timeout: HTTP request timeout in seconds (default: 30).
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        model: str = _DEFAULT_MODEL,
        dimension: int = _DEFAULT_DIM,
        timeout: int = _REQUEST_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # EmbeddingProvider interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama's /api/embeddings endpoint.

        Each text is sent as an individual request (Ollama API is
        single-prompt only).

        Raises:
            ConnectionError: If the server is unreachable.
            RuntimeError: If the server returns a non-200 status.
        """
        results: list[list[float]] = []
        for text in texts:
            try:
                resp = self._client.post(
                    f"{self._base_url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                )
            except httpx.ConnectError as exc:
                raise ConnectionError(
                    f"Could not connect to Ollama at {self._base_url}: {exc}"
                ) from exc

            if resp.status_code != 200:
                raise RuntimeError(
                    f"Ollama returned HTTP {resp.status_code}: {resp.text}"
                )

            data = resp.json()
            embedding = data.get("embedding", [])
            results.append(embedding)

        return results

    def close(self) -> None:
        self._client.close()
