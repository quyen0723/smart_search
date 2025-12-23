"""Embedding module for semantic code representation."""

from smart_search.embedding.cache import CacheStats, EmbeddingCache, InMemoryCache
from smart_search.embedding.jina_embedder import (
    BaseEmbedder,
    JinaEmbedder,
    MockEmbedder,
    create_embedder,
)
from smart_search.embedding.models import (
    BatchEmbeddingResult,
    ChunkEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
)
from smart_search.embedding.pipeline import (
    ChunkingConfig,
    CodeChunk,
    CodeChunker,
    EmbeddingPipeline,
    PipelineResult,
)

__all__ = [
    # Models
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "EmbeddingConfig",
    "ChunkEmbedding",
    # Embedders
    "BaseEmbedder",
    "JinaEmbedder",
    "MockEmbedder",
    "create_embedder",
    # Pipeline
    "ChunkingConfig",
    "CodeChunk",
    "CodeChunker",
    "EmbeddingPipeline",
    "PipelineResult",
    # Cache
    "EmbeddingCache",
    "InMemoryCache",
    "CacheStats",
]
