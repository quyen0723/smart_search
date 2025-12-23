"""Jina AI embeddings integration.

Provides embedding generation using Jina AI's embedding models.
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod
from typing import Any

import httpx

from smart_search.core.exceptions import EmbeddingGenerationError, ModelLoadError
from smart_search.embedding.models import (
    BatchEmbeddingResult,
    EmbeddingConfig,
    EmbeddingResult,
)
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with the embedding vector.
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            BatchEmbeddingResult with all embeddings.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class JinaEmbedder(BaseEmbedder):
    """Jina AI embeddings implementation.

    Uses Jina AI's embedding API to generate high-quality
    code and text embeddings.
    """

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(
        self,
        api_key: str,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize Jina embedder.

        Args:
            api_key: Jina AI API key.
            config: Embedding configuration.

        Raises:
            ModelLoadError: If initialization fails.
        """
        if not api_key:
            raise ModelLoadError("jina", ValueError("API key is required"))

        self.api_key = api_key
        self.config = config or EmbeddingConfig()
        self._client: httpx.AsyncClient | None = None
        self._initialized = False

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized.

        Returns:
            The HTTP client.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            self._initialized = True
        return self._client

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with the embedding vector.

        Raises:
            EmbeddingGenerationError: If embedding fails.
        """
        if not text.strip():
            raise EmbeddingGenerationError(0, ValueError("Empty text"))

        result = await self.embed_batch([text])
        return result.results[0]

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            BatchEmbeddingResult with all embeddings.

        Raises:
            EmbeddingGenerationError: If embedding fails.
        """
        if not texts:
            return BatchEmbeddingResult(
                results=[],
                total_tokens=0,
                model=self.config.model_name,
            )

        # Filter empty texts
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise EmbeddingGenerationError(0, ValueError("All texts are empty"))

        client = await self._ensure_client()
        results: list[EmbeddingResult] = []
        total_tokens = 0

        # Process in batches
        for i in range(0, len(valid_texts), self.config.batch_size):
            batch = valid_texts[i : i + self.config.batch_size]
            batch_results, tokens = await self._embed_batch_request(client, batch)
            results.extend(batch_results)
            total_tokens += tokens

        return BatchEmbeddingResult(
            results=results,
            total_tokens=total_tokens,
            model=self.config.model_name,
        )

    async def _embed_batch_request(
        self,
        client: httpx.AsyncClient,
        texts: list[str],
    ) -> tuple[list[EmbeddingResult], int]:
        """Make a batch embedding request.

        Args:
            client: HTTP client.
            texts: Texts to embed.

        Returns:
            Tuple of (results, total_tokens).

        Raises:
            EmbeddingGenerationError: If request fails.
        """
        payload = {
            "model": self.config.model_name,
            "input": texts,
            "dimensions": self.config.dimensions,
            "normalized": self.config.normalize,
        }

        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(self.API_URL, json=payload)
                response.raise_for_status()
                data = response.json()

                results = []
                total_tokens = 0

                for i, item in enumerate(data.get("data", [])):
                    embedding = item.get("embedding", [])
                    tokens = data.get("usage", {}).get("total_tokens", 0)
                    total_tokens = tokens

                    results.append(
                        EmbeddingResult(
                            text=texts[i],
                            embedding=embedding,
                            model=self.config.model_name,
                            dimensions=len(embedding),
                            token_count=tokens // len(texts) if texts else 0,
                        )
                    )

                logger.debug(
                    "Embedded batch",
                    batch_size=len(texts),
                    total_tokens=total_tokens,
                )
                return results, total_tokens

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        "Rate limited, waiting",
                        wait_time=wait_time,
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise EmbeddingGenerationError(
                    sum(len(t) for t in texts),
                    e,
                )
            except httpx.RequestError as e:
                if attempt < self.config.max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    logger.warning(
                        "Request failed, retrying",
                        error=str(e),
                        attempt=attempt + 1,
                    )
                    await asyncio.sleep(wait_time)
                    continue
                raise EmbeddingGenerationError(
                    sum(len(t) for t in texts),
                    e,
                )

        raise EmbeddingGenerationError(
            sum(len(t) for t in texts),
            RuntimeError("Max retries exceeded"),
        )

    async def close(self) -> None:
        """Clean up HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    async def __aenter__(self) -> "JinaEmbedder":
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing.

    Generates deterministic fake embeddings based on text hash.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize mock embedder.

        Args:
            config: Embedding configuration.
        """
        self.config = config or EmbeddingConfig()
        self._call_count = 0

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate a deterministic fake embedding.

        Args:
            text: Text to "embed".

        Returns:
            Fake embedding vector.
        """
        # Use hash for deterministic results
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Generate embedding from hash
        embedding = []
        for i in range(self.config.dimensions):
            # Cycle through hash bytes
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1  # -1 to 1
            embedding.append(value)

        # Normalize if configured
        if self.config.normalize:
            norm = sum(v**2 for v in embedding) ** 0.5
            if norm > 0:
                embedding = [v / norm for v in embedding]

        return embedding

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with fake embedding.
        """
        self._call_count += 1
        embedding = self._generate_embedding(text)
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model="mock-embedder",
            dimensions=len(embedding),
            token_count=len(text.split()),
        )

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Embed multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            BatchEmbeddingResult with fake embeddings.
        """
        results = []
        total_tokens = 0

        for text in texts:
            result = await self.embed(text)
            results.append(result)
            total_tokens += result.token_count

        return BatchEmbeddingResult(
            results=results,
            total_tokens=total_tokens,
            model="mock-embedder",
        )

    async def close(self) -> None:
        """No-op for mock."""
        pass

    @property
    def call_count(self) -> int:
        """Get number of embed calls."""
        return self._call_count


def create_embedder(
    api_key: str | None = None,
    config: EmbeddingConfig | None = None,
    use_mock: bool = False,
) -> BaseEmbedder:
    """Factory function to create an embedder.

    Args:
        api_key: API key for real embedder.
        config: Embedding configuration.
        use_mock: Whether to use mock embedder.

    Returns:
        Embedder instance.

    Raises:
        ModelLoadError: If API key missing and not using mock.
    """
    if use_mock:
        return MockEmbedder(config)

    if not api_key:
        raise ModelLoadError("jina", ValueError("API key required"))

    return JinaEmbedder(api_key, config)
