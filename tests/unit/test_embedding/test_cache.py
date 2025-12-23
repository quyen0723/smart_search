"""Tests for embedding cache."""

import time
from pathlib import Path

import pytest

from smart_search.embedding.cache import (
    CacheStats,
    EmbeddingCache,
    InMemoryCache,
)
from smart_search.embedding.models import ChunkEmbedding, EmbeddingConfig


class TestCacheStats:
    """Tests for CacheStats."""

    def test_hit_rate_with_hits(self) -> None:
        """Test hit rate calculation."""
        stats = CacheStats(
            total_entries=10,
            hits=8,
            misses=2,
            size_bytes=1000,
        )
        assert stats.hit_rate == 0.8

    def test_hit_rate_no_accesses(self) -> None:
        """Test hit rate with no accesses."""
        stats = CacheStats(
            total_entries=0,
            hits=0,
            misses=0,
            size_bytes=0,
        )
        assert stats.hit_rate == 0.0


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    @pytest.fixture
    def cache(self) -> InMemoryCache:
        """Create in-memory cache."""
        return InMemoryCache()

    @pytest.fixture
    def sample_embedding(self) -> ChunkEmbedding:
        """Create a sample embedding."""
        return ChunkEmbedding(
            chunk_id="test::chunk_0",
            unit_id="test",
            embedding=[0.1, 0.2, 0.3],
            content_hash="abc123",
            metadata={"type": "function"},
        )

    def test_put_and_get(
        self, cache: InMemoryCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test put and get."""
        cache.put(sample_embedding)
        result = cache.get(sample_embedding.content_hash)
        assert result is not None
        assert result.chunk_id == sample_embedding.chunk_id
        assert result.embedding == sample_embedding.embedding

    def test_get_missing(self, cache: InMemoryCache) -> None:
        """Test get missing entry."""
        result = cache.get("nonexistent")
        assert result is None

    def test_has(self, cache: InMemoryCache, sample_embedding: ChunkEmbedding) -> None:
        """Test has method."""
        assert not cache.has(sample_embedding.content_hash)
        cache.put(sample_embedding)
        assert cache.has(sample_embedding.content_hash)

    def test_delete(
        self, cache: InMemoryCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test delete."""
        cache.put(sample_embedding)
        assert cache.delete(sample_embedding.content_hash)
        assert not cache.has(sample_embedding.content_hash)

    def test_delete_missing(self, cache: InMemoryCache) -> None:
        """Test delete missing entry."""
        assert not cache.delete("nonexistent")

    def test_put_many(self, cache: InMemoryCache) -> None:
        """Test put_many."""
        embeddings = [
            ChunkEmbedding(
                chunk_id=f"test::chunk_{i}",
                unit_id="test",
                embedding=[float(i)],
                content_hash=f"hash{i}",
            )
            for i in range(3)
        ]
        cache.put_many(embeddings)
        assert cache.get_stats().total_entries == 3

    def test_get_many(self, cache: InMemoryCache) -> None:
        """Test get_many."""
        embeddings = [
            ChunkEmbedding(
                chunk_id=f"test::chunk_{i}",
                unit_id="test",
                embedding=[float(i)],
                content_hash=f"hash{i}",
            )
            for i in range(3)
        ]
        cache.put_many(embeddings)
        results = cache.get_many(["hash0", "hash1", "nonexistent"])
        assert len(results) == 2
        assert "hash0" in results
        assert "hash1" in results

    def test_get_cached_hashes(
        self, cache: InMemoryCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test get_cached_hashes."""
        cache.put(sample_embedding)
        hashes = cache.get_cached_hashes()
        assert sample_embedding.content_hash in hashes

    def test_clear(self, cache: InMemoryCache, sample_embedding: ChunkEmbedding) -> None:
        """Test clear."""
        cache.put(sample_embedding)
        count = cache.clear()
        assert count == 1
        assert cache.get_stats().total_entries == 0

    def test_stats(self, cache: InMemoryCache, sample_embedding: ChunkEmbedding) -> None:
        """Test stats."""
        cache.put(sample_embedding)
        cache.get(sample_embedding.content_hash)  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats.total_entries == 1
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        config = EmbeddingConfig(cache_ttl=1)  # 1 second TTL
        cache = InMemoryCache(config)

        embedding = ChunkEmbedding(
            chunk_id="test",
            unit_id="unit",
            embedding=[0.1],
            content_hash="hash",
        )
        cache.put(embedding)
        assert cache.get("hash") is not None

        # Wait for expiration
        time.sleep(1.5)
        assert cache.get("hash") is None


class TestEmbeddingCache:
    """Tests for EmbeddingCache (SQLite)."""

    @pytest.fixture
    def cache_path(self, tmp_path: Path) -> Path:
        """Create temporary cache path."""
        return tmp_path / "test_cache.db"

    @pytest.fixture
    def cache(self, cache_path: Path) -> EmbeddingCache:
        """Create embedding cache."""
        return EmbeddingCache(cache_path)

    @pytest.fixture
    def sample_embedding(self) -> ChunkEmbedding:
        """Create a sample embedding."""
        return ChunkEmbedding(
            chunk_id="test::chunk_0",
            unit_id="test",
            embedding=[0.1, 0.2, 0.3],
            content_hash="abc123",
            metadata={"type": "function"},
        )

    def test_put_and_get(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test put and get."""
        cache.put(sample_embedding)
        result = cache.get(sample_embedding.content_hash)
        assert result is not None
        assert result.chunk_id == sample_embedding.chunk_id
        assert result.embedding == sample_embedding.embedding
        cache.close()

    def test_get_missing(self, cache: EmbeddingCache) -> None:
        """Test get missing entry."""
        result = cache.get("nonexistent")
        assert result is None
        cache.close()

    def test_has(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test has method."""
        assert not cache.has(sample_embedding.content_hash)
        cache.put(sample_embedding)
        assert cache.has(sample_embedding.content_hash)
        cache.close()

    def test_delete(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test delete."""
        cache.put(sample_embedding)
        assert cache.delete(sample_embedding.content_hash)
        assert not cache.has(sample_embedding.content_hash)
        cache.close()

    def test_delete_by_unit(self, cache: EmbeddingCache) -> None:
        """Test delete by unit."""
        embeddings = [
            ChunkEmbedding(
                chunk_id=f"unit1::chunk_{i}",
                unit_id="unit1",
                embedding=[float(i)],
                content_hash=f"hash1_{i}",
            )
            for i in range(3)
        ]
        embeddings.append(
            ChunkEmbedding(
                chunk_id="unit2::chunk_0",
                unit_id="unit2",
                embedding=[0.0],
                content_hash="hash2_0",
            )
        )
        cache.put_many(embeddings)

        count = cache.delete_by_unit("unit1")
        assert count == 3
        assert cache.get_stats().total_entries == 1
        cache.close()

    def test_get_by_unit(self, cache: EmbeddingCache) -> None:
        """Test get by unit."""
        embeddings = [
            ChunkEmbedding(
                chunk_id=f"unit1::chunk_{i}",
                unit_id="unit1",
                embedding=[float(i)],
                content_hash=f"hash_{i}",
            )
            for i in range(2)
        ]
        cache.put_many(embeddings)

        results = cache.get_by_unit("unit1")
        assert len(results) == 2
        cache.close()

    def test_get_cached_hashes(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test get_cached_hashes."""
        cache.put(sample_embedding)
        hashes = cache.get_cached_hashes()
        assert sample_embedding.content_hash in hashes
        cache.close()

    def test_clear(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test clear."""
        cache.put(sample_embedding)
        count = cache.clear()
        assert count == 1
        assert cache.get_stats().total_entries == 0
        cache.close()

    def test_stats(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test stats."""
        cache.put(sample_embedding)
        cache.get(sample_embedding.content_hash)  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()
        assert stats.total_entries == 1
        assert stats.hits == 1
        assert stats.misses == 1
        cache.close()

    def test_context_manager(
        self, cache_path: Path, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test context manager."""
        with EmbeddingCache(cache_path) as cache:
            cache.put(sample_embedding)
            assert cache.has(sample_embedding.content_hash)

    def test_persistence(
        self, cache_path: Path, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test data persistence."""
        # Write data
        with EmbeddingCache(cache_path) as cache:
            cache.put(sample_embedding)

        # Read data in new instance
        with EmbeddingCache(cache_path) as cache:
            result = cache.get(sample_embedding.content_hash)
            assert result is not None
            assert result.embedding == sample_embedding.embedding

    def test_vacuum(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test vacuum."""
        cache.put(sample_embedding)
        cache.clear()
        cache.vacuum()  # Should not raise
        cache.close()

    def test_cleanup_expired(self, cache_path: Path) -> None:
        """Test cleanup of expired entries."""
        config = EmbeddingConfig(cache_ttl=1)
        cache = EmbeddingCache(cache_path, config)

        embedding = ChunkEmbedding(
            chunk_id="test",
            unit_id="unit",
            embedding=[0.1],
            content_hash="hash",
        )
        cache.put(embedding)

        # Wait for expiration
        time.sleep(1.5)
        removed = cache.cleanup_expired()
        assert removed == 1
        cache.close()

    def test_compression(
        self, cache: EmbeddingCache
    ) -> None:
        """Test embedding compression."""
        # Large embedding
        embedding = ChunkEmbedding(
            chunk_id="test",
            unit_id="unit",
            embedding=[float(i) for i in range(1024)],
            content_hash="hash",
        )
        cache.put(embedding)
        result = cache.get("hash")
        assert result is not None
        assert len(result.embedding) == 1024
        assert result.embedding == embedding.embedding
        cache.close()

    def test_metadata_preserved(
        self, cache: EmbeddingCache, sample_embedding: ChunkEmbedding
    ) -> None:
        """Test metadata is preserved."""
        cache.put(sample_embedding)
        result = cache.get(sample_embedding.content_hash)
        assert result is not None
        assert result.metadata == sample_embedding.metadata
        cache.close()

    def test_ttl_check_on_get(self, cache_path: Path) -> None:
        """Test TTL is checked on get."""
        config = EmbeddingConfig(cache_ttl=1)
        cache = EmbeddingCache(cache_path, config)

        embedding = ChunkEmbedding(
            chunk_id="test",
            unit_id="unit",
            embedding=[0.1],
            content_hash="hash",
        )
        cache.put(embedding)
        assert cache.get("hash") is not None

        time.sleep(1.5)
        assert cache.get("hash") is None
        cache.close()
