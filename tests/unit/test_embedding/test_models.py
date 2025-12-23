"""Tests for embedding models."""

import pytest

from smart_search.embedding.models import (
    BatchEmbeddingResult,
    ChunkEmbedding,
    EmbeddingConfig,
    EmbeddingResult,
)


class TestEmbeddingResult:
    """Tests for EmbeddingResult."""

    def test_create_result(self) -> None:
        """Test creating an embedding result."""
        result = EmbeddingResult(
            text="hello world",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            dimensions=3,
            token_count=2,
        )
        assert result.text == "hello world"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model == "test-model"
        assert result.dimensions == 3
        assert result.token_count == 2

    def test_default_token_count(self) -> None:
        """Test default token count."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1],
            model="test",
            dimensions=1,
        )
        assert result.token_count == 0

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = EmbeddingResult(
            text="test",
            embedding=[0.1, 0.2],
            model="model",
            dimensions=2,
            token_count=1,
        )
        d = result.to_dict()
        assert d["text"] == "test"
        assert d["embedding"] == [0.1, 0.2]
        assert d["model"] == "model"
        assert d["dimensions"] == 2
        assert d["token_count"] == 1


class TestBatchEmbeddingResult:
    """Tests for BatchEmbeddingResult."""

    def test_empty_batch(self) -> None:
        """Test empty batch result."""
        result = BatchEmbeddingResult(
            results=[],
            total_tokens=0,
            model="test",
        )
        assert result.count == 0
        assert result.embeddings == []

    def test_batch_with_results(self) -> None:
        """Test batch with results."""
        results = [
            EmbeddingResult(
                text=f"text{i}",
                embedding=[float(i)],
                model="test",
                dimensions=1,
            )
            for i in range(3)
        ]
        batch = BatchEmbeddingResult(
            results=results,
            total_tokens=10,
            model="test",
        )
        assert batch.count == 3
        assert batch.total_tokens == 10
        assert len(batch.embeddings) == 3

    def test_embeddings_property(self) -> None:
        """Test embeddings property."""
        results = [
            EmbeddingResult(
                text="a",
                embedding=[1.0, 2.0],
                model="test",
                dimensions=2,
            ),
            EmbeddingResult(
                text="b",
                embedding=[3.0, 4.0],
                model="test",
                dimensions=2,
            ),
        ]
        batch = BatchEmbeddingResult(
            results=results,
            total_tokens=5,
            model="test",
        )
        embeddings = batch.embeddings
        assert embeddings == [[1.0, 2.0], [3.0, 4.0]]


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = EmbeddingConfig()
        assert config.model_name == "jinaai/jina-embeddings-v3"
        assert config.dimensions == 1024
        assert config.batch_size == 32
        assert config.max_retries == 3
        assert config.timeout == 30.0
        assert config.normalize is True
        assert config.cache_enabled is True
        assert config.cache_ttl == 86400

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="custom-model",
            dimensions=512,
            batch_size=16,
            max_retries=5,
            timeout=60.0,
            normalize=False,
            cache_enabled=False,
            cache_ttl=3600,
        )
        assert config.model_name == "custom-model"
        assert config.dimensions == 512
        assert config.batch_size == 16
        assert config.max_retries == 5
        assert config.timeout == 60.0
        assert config.normalize is False
        assert config.cache_enabled is False
        assert config.cache_ttl == 3600


class TestChunkEmbedding:
    """Tests for ChunkEmbedding."""

    def test_create_chunk_embedding(self) -> None:
        """Test creating a chunk embedding."""
        chunk = ChunkEmbedding(
            chunk_id="test::chunk_0",
            unit_id="test",
            embedding=[0.1, 0.2, 0.3],
            content_hash="abc123",
            metadata={"type": "function"},
        )
        assert chunk.chunk_id == "test::chunk_0"
        assert chunk.unit_id == "test"
        assert chunk.embedding == [0.1, 0.2, 0.3]
        assert chunk.content_hash == "abc123"
        assert chunk.metadata == {"type": "function"}

    def test_default_metadata(self) -> None:
        """Test default empty metadata."""
        chunk = ChunkEmbedding(
            chunk_id="test",
            unit_id="unit",
            embedding=[0.1],
            content_hash="hash",
        )
        assert chunk.metadata == {}

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        chunk = ChunkEmbedding(
            chunk_id="test::chunk_0",
            unit_id="test",
            embedding=[0.1, 0.2],
            content_hash="abc",
            metadata={"key": "value"},
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "test::chunk_0"
        assert d["unit_id"] == "test"
        assert d["embedding"] == [0.1, 0.2]
        assert d["content_hash"] == "abc"
        assert d["metadata"] == {"key": "value"}

    def test_from_dict(self) -> None:
        """Test creation from dictionary."""
        data = {
            "chunk_id": "test::chunk_0",
            "unit_id": "test",
            "embedding": [0.1, 0.2],
            "content_hash": "abc",
            "metadata": {"key": "value"},
        }
        chunk = ChunkEmbedding.from_dict(data)
        assert chunk.chunk_id == "test::chunk_0"
        assert chunk.unit_id == "test"
        assert chunk.embedding == [0.1, 0.2]
        assert chunk.content_hash == "abc"
        assert chunk.metadata == {"key": "value"}

    def test_from_dict_no_metadata(self) -> None:
        """Test creation from dictionary without metadata."""
        data = {
            "chunk_id": "test",
            "unit_id": "unit",
            "embedding": [0.1],
            "content_hash": "hash",
        }
        chunk = ChunkEmbedding.from_dict(data)
        assert chunk.metadata == {}

    def test_roundtrip(self) -> None:
        """Test to_dict and from_dict roundtrip."""
        original = ChunkEmbedding(
            chunk_id="test::chunk_0",
            unit_id="test",
            embedding=[0.1, 0.2, 0.3],
            content_hash="abc123",
            metadata={"type": "function", "name": "test_func"},
        )
        data = original.to_dict()
        restored = ChunkEmbedding.from_dict(data)
        assert restored.chunk_id == original.chunk_id
        assert restored.unit_id == original.unit_id
        assert restored.embedding == original.embedding
        assert restored.content_hash == original.content_hash
        assert restored.metadata == original.metadata
