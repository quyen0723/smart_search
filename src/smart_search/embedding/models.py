"""Embedding models and data structures.

Defines the data structures for embedding operations.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EmbeddingResult:
    """Result of embedding a single text.

    Attributes:
        text: The original text that was embedded.
        embedding: The embedding vector.
        model: The model used for embedding.
        dimensions: Dimensionality of the embedding.
        token_count: Number of tokens in the text.
    """

    text: str
    embedding: list[float]
    model: str
    dimensions: int
    token_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "model": self.model,
            "dimensions": self.dimensions,
            "token_count": self.token_count,
        }


@dataclass
class BatchEmbeddingResult:
    """Result of embedding a batch of texts.

    Attributes:
        results: List of individual embedding results.
        total_tokens: Total tokens across all texts.
        model: The model used.
    """

    results: list[EmbeddingResult]
    total_tokens: int
    model: str

    @property
    def embeddings(self) -> list[list[float]]:
        """Get just the embedding vectors."""
        return [r.embedding for r in self.results]

    @property
    def count(self) -> int:
        """Get the number of embeddings."""
        return len(self.results)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding operations.

    Attributes:
        model_name: Name of the embedding model.
        dimensions: Embedding dimensions.
        batch_size: Batch size for embedding.
        max_retries: Maximum retries on failure.
        timeout: Timeout in seconds.
        normalize: Whether to normalize embeddings.
        cache_enabled: Whether to use caching.
    """

    model_name: str = "jinaai/jina-embeddings-v3"
    dimensions: int = 1024
    batch_size: int = 32
    max_retries: int = 3
    timeout: float = 30.0
    normalize: bool = True
    cache_enabled: bool = True
    cache_ttl: int = 86400  # 24 hours


@dataclass
class ChunkEmbedding:
    """Embedding for a code chunk.

    Attributes:
        chunk_id: ID of the chunk.
        unit_id: ID of the parent code unit.
        embedding: The embedding vector.
        content_hash: Hash of the content for cache invalidation.
        metadata: Additional metadata.
    """

    chunk_id: str
    unit_id: str
    embedding: list[float]
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "unit_id": self.unit_id,
            "embedding": self.embedding,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkEmbedding":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            unit_id=data["unit_id"],
            embedding=data["embedding"],
            content_hash=data["content_hash"],
            metadata=data.get("metadata", {}),
        )
