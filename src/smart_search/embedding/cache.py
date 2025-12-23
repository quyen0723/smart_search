"""Embedding cache for performance optimization.

Caches embeddings to avoid redundant API calls.
"""

import gzip
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from smart_search.embedding.models import ChunkEmbedding, EmbeddingConfig
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheStats:
    """Cache statistics.

    Attributes:
        total_entries: Total number of entries.
        hits: Cache hits.
        misses: Cache misses.
        size_bytes: Approximate cache size in bytes.
    """

    total_entries: int
    hits: int
    misses: int
    size_bytes: int

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class EmbeddingCache:
    """SQLite-based embedding cache.

    Stores embeddings persistently with TTL support.
    """

    def __init__(
        self,
        cache_path: Path | str,
        config: EmbeddingConfig | None = None,
    ) -> None:
        """Initialize cache.

        Args:
            cache_path: Path to SQLite database.
            config: Embedding configuration.
        """
        self.cache_path = Path(cache_path)
        self.config = config or EmbeddingConfig()
        self._hits = 0
        self._misses = 0
        self._conn: sqlite3.Connection | None = None
        self._initialized = False

    def _ensure_connection(self) -> sqlite3.Connection:
        """Ensure database connection is open.

        Returns:
            SQLite connection.
        """
        if self._conn is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.cache_path))
            self._conn.row_factory = sqlite3.Row
            self._init_schema()
            self._initialized = True
        return self._conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._conn
        if conn is None:
            return

        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                chunk_id TEXT NOT NULL,
                unit_id TEXT NOT NULL,
                embedding BLOB NOT NULL,
                metadata TEXT,
                created_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                model TEXT NOT NULL
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_unit_id ON embeddings(unit_id)
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_created_at ON embeddings(created_at)
        """)

        conn.commit()

    def get(self, content_hash: str) -> ChunkEmbedding | None:
        """Get embedding by content hash.

        Args:
            content_hash: Content hash to look up.

        Returns:
            ChunkEmbedding if found, None otherwise.
        """
        conn = self._ensure_connection()

        row = conn.execute(
            """
            SELECT * FROM embeddings
            WHERE content_hash = ?
            """,
            (content_hash,),
        ).fetchone()

        if row is None:
            self._misses += 1
            return None

        # Check TTL
        if self.config.cache_ttl > 0:
            age = time.time() - row["created_at"]
            if age > self.config.cache_ttl:
                self.delete(content_hash)
                self._misses += 1
                return None

        # Update access time
        conn.execute(
            """
            UPDATE embeddings SET accessed_at = ? WHERE content_hash = ?
            """,
            (time.time(), content_hash),
        )
        conn.commit()

        self._hits += 1

        # Decompress embedding
        embedding = self._decompress_embedding(row["embedding"])

        return ChunkEmbedding(
            chunk_id=row["chunk_id"],
            unit_id=row["unit_id"],
            embedding=embedding,
            content_hash=content_hash,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    def get_many(self, content_hashes: list[str]) -> dict[str, ChunkEmbedding]:
        """Get multiple embeddings.

        Args:
            content_hashes: List of content hashes.

        Returns:
            Dict mapping hash to ChunkEmbedding.
        """
        result = {}
        for hash_val in content_hashes:
            embedding = self.get(hash_val)
            if embedding is not None:
                result[hash_val] = embedding
        return result

    def put(self, embedding: ChunkEmbedding) -> None:
        """Store an embedding.

        Args:
            embedding: ChunkEmbedding to store.
        """
        conn = self._ensure_connection()

        # Compress embedding
        compressed = self._compress_embedding(embedding.embedding)

        now = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings
            (content_hash, chunk_id, unit_id, embedding, metadata, created_at, accessed_at, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                embedding.content_hash,
                embedding.chunk_id,
                embedding.unit_id,
                compressed,
                json.dumps(embedding.metadata),
                now,
                now,
                self.config.model_name,
            ),
        )
        conn.commit()

    def put_many(self, embeddings: list[ChunkEmbedding]) -> None:
        """Store multiple embeddings.

        Args:
            embeddings: List of ChunkEmbeddings to store.
        """
        for embedding in embeddings:
            self.put(embedding)

    def delete(self, content_hash: str) -> bool:
        """Delete an embedding.

        Args:
            content_hash: Content hash to delete.

        Returns:
            True if deleted, False if not found.
        """
        conn = self._ensure_connection()
        cursor = conn.execute(
            """
            DELETE FROM embeddings WHERE content_hash = ?
            """,
            (content_hash,),
        )
        conn.commit()
        return cursor.rowcount > 0

    def delete_by_unit(self, unit_id: str) -> int:
        """Delete all embeddings for a unit.

        Args:
            unit_id: Unit ID to delete embeddings for.

        Returns:
            Number of deleted entries.
        """
        conn = self._ensure_connection()
        cursor = conn.execute(
            """
            DELETE FROM embeddings WHERE unit_id = ?
            """,
            (unit_id,),
        )
        conn.commit()
        return cursor.rowcount

    def has(self, content_hash: str) -> bool:
        """Check if content hash exists in cache.

        Args:
            content_hash: Content hash to check.

        Returns:
            True if exists.
        """
        conn = self._ensure_connection()
        row = conn.execute(
            """
            SELECT 1 FROM embeddings WHERE content_hash = ?
            """,
            (content_hash,),
        ).fetchone()
        return row is not None

    def get_cached_hashes(self) -> set[str]:
        """Get all cached content hashes.

        Returns:
            Set of content hashes.
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            """
            SELECT content_hash FROM embeddings
            """
        ).fetchall()
        return {row["content_hash"] for row in rows}

    def get_by_unit(self, unit_id: str) -> list[ChunkEmbedding]:
        """Get all embeddings for a unit.

        Args:
            unit_id: Unit ID to get embeddings for.

        Returns:
            List of ChunkEmbeddings.
        """
        conn = self._ensure_connection()
        rows = conn.execute(
            """
            SELECT * FROM embeddings WHERE unit_id = ?
            """,
            (unit_id,),
        ).fetchall()

        embeddings = []
        for row in rows:
            embedding = self._decompress_embedding(row["embedding"])
            embeddings.append(
                ChunkEmbedding(
                    chunk_id=row["chunk_id"],
                    unit_id=row["unit_id"],
                    embedding=embedding,
                    content_hash=row["content_hash"],
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
            )
        return embeddings

    def clear(self) -> int:
        """Clear all cached embeddings.

        Returns:
            Number of deleted entries.
        """
        conn = self._ensure_connection()
        cursor = conn.execute("DELETE FROM embeddings")
        conn.commit()
        return cursor.rowcount

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of removed entries.
        """
        if self.config.cache_ttl <= 0:
            return 0

        conn = self._ensure_connection()
        cutoff = time.time() - self.config.cache_ttl
        cursor = conn.execute(
            """
            DELETE FROM embeddings WHERE created_at < ?
            """,
            (cutoff,),
        )
        conn.commit()
        return cursor.rowcount

    def get_stats(self) -> CacheStats:
        """Get cache statistics.

        Returns:
            CacheStats object.
        """
        conn = self._ensure_connection()

        count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

        # Get approximate size
        page_count = conn.execute("PRAGMA page_count").fetchone()[0]
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        size_bytes = page_count * page_size

        return CacheStats(
            total_entries=count,
            hits=self._hits,
            misses=self._misses,
            size_bytes=size_bytes,
        )

    def vacuum(self) -> None:
        """Compact the database."""
        conn = self._ensure_connection()
        conn.execute("VACUUM")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._initialized = False

    def _compress_embedding(self, embedding: list[float]) -> bytes:
        """Compress embedding for storage.

        Args:
            embedding: Embedding vector.

        Returns:
            Compressed bytes.
        """
        json_str = json.dumps(embedding)
        return gzip.compress(json_str.encode())

    def _decompress_embedding(self, data: bytes) -> list[float]:
        """Decompress embedding from storage.

        Args:
            data: Compressed bytes.

        Returns:
            Embedding vector.
        """
        json_str = gzip.decompress(data).decode()
        return json.loads(json_str)

    def __enter__(self) -> "EmbeddingCache":
        """Enter context manager."""
        self._ensure_connection()
        return self

    def __exit__(self, *args: Any) -> None:
        """Exit context manager."""
        self.close()


class InMemoryCache:
    """In-memory embedding cache for testing.

    Simple dict-based cache without persistence.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        """Initialize cache.

        Args:
            config: Embedding configuration.
        """
        self.config = config or EmbeddingConfig()
        self._cache: dict[str, tuple[ChunkEmbedding, float]] = {}
        self._hits = 0
        self._misses = 0

    def get(self, content_hash: str) -> ChunkEmbedding | None:
        """Get embedding by content hash."""
        entry = self._cache.get(content_hash)
        if entry is None:
            self._misses += 1
            return None

        embedding, created_at = entry

        # Check TTL
        if self.config.cache_ttl > 0:
            age = time.time() - created_at
            if age > self.config.cache_ttl:
                del self._cache[content_hash]
                self._misses += 1
                return None

        self._hits += 1
        return embedding

    def get_many(self, content_hashes: list[str]) -> dict[str, ChunkEmbedding]:
        """Get multiple embeddings."""
        result = {}
        for hash_val in content_hashes:
            embedding = self.get(hash_val)
            if embedding is not None:
                result[hash_val] = embedding
        return result

    def put(self, embedding: ChunkEmbedding) -> None:
        """Store an embedding."""
        self._cache[embedding.content_hash] = (embedding, time.time())

    def put_many(self, embeddings: list[ChunkEmbedding]) -> None:
        """Store multiple embeddings."""
        for embedding in embeddings:
            self.put(embedding)

    def delete(self, content_hash: str) -> bool:
        """Delete an embedding."""
        if content_hash in self._cache:
            del self._cache[content_hash]
            return True
        return False

    def has(self, content_hash: str) -> bool:
        """Check if content hash exists."""
        return content_hash in self._cache

    def get_cached_hashes(self) -> set[str]:
        """Get all cached content hashes."""
        return set(self._cache.keys())

    def clear(self) -> int:
        """Clear all cached embeddings."""
        count = len(self._cache)
        self._cache.clear()
        return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            total_entries=len(self._cache),
            hits=self._hits,
            misses=self._misses,
            size_bytes=0,  # In-memory, no disk size
        )
