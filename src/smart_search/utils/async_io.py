"""Async file I/O utilities for non-blocking file operations.

Provides async file reading with LRU caching and parallel file reading.
"""

import asyncio
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import aiofiles

from smart_search.utils.logging import get_logger

logger = get_logger(__name__)

# Cache settings
_ASYNC_CACHE_MAX_SIZE = 500


def _get_file_mtime(file_path: str) -> float:
    """Get file modification time synchronously."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0


# Use sync LRU cache for mtime lookups (they're fast)
@lru_cache(maxsize=_ASYNC_CACHE_MAX_SIZE)
def _cached_mtime(file_path: str, check_time: int) -> float:
    """Cached mtime lookup. check_time is rounded to seconds for cache efficiency."""
    return _get_file_mtime(file_path)


class AsyncFileCache:
    """Async file content cache with mtime validation."""

    def __init__(self, max_size: int = _ASYNC_CACHE_MAX_SIZE):
        self._cache: dict[str, tuple[float, str]] = {}  # path -> (mtime, content)
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._lock = asyncio.Lock()

    async def read_file(self, file_path: str) -> str:
        """Read file content with caching.

        Args:
            file_path: Path to the file.

        Returns:
            File content or empty string if file doesn't exist.
        """
        mtime = _get_file_mtime(file_path)
        if mtime == 0.0:
            return ""

        async with self._lock:
            # Check cache
            if file_path in self._cache:
                cached_mtime, cached_content = self._cache[file_path]
                if cached_mtime == mtime:
                    self._hits += 1
                    return cached_content

            self._misses += 1

        # Read file asynchronously
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
        except (OSError, IOError) as e:
            logger.debug("Failed to read file", file_path=file_path, error=str(e))
            return ""

        # Update cache
        async with self._lock:
            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                # Remove ~10% of entries
                to_remove = list(self._cache.keys())[:self._max_size // 10]
                for key in to_remove:
                    del self._cache[key]

            self._cache[file_path] = (mtime, content)

        return content

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
            "hit_ratio": self._hits / total if total > 0 else 0.0
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def invalidate(self, file_path: str) -> bool:
        """Invalidate a specific file from cache.

        Args:
            file_path: Path to invalidate.

        Returns:
            True if file was in cache.
        """
        if file_path in self._cache:
            del self._cache[file_path]
            return True
        return False


# Global cache instance
_async_file_cache = AsyncFileCache()


async def async_read_file(file_path: str | Path) -> str:
    """Read a file asynchronously with caching.

    Args:
        file_path: Path to the file.

    Returns:
        File content or empty string if file doesn't exist.
    """
    return await _async_file_cache.read_file(str(file_path))


async def async_read_files_parallel(
    file_paths: Sequence[str | Path],
    max_concurrent: int = 50
) -> dict[str, str]:
    """Read multiple files in parallel with concurrency limit.

    Args:
        file_paths: List of file paths to read.
        max_concurrent: Maximum concurrent file reads.

    Returns:
        Dict mapping file path to content.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def read_with_semaphore(path: str | Path) -> tuple[str, str]:
        async with semaphore:
            path_str = str(path)
            content = await async_read_file(path_str)
            return path_str, content

    tasks = [read_with_semaphore(p) for p in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and build result dict
    output = {}
    for result in results:
        if isinstance(result, tuple):
            path, content = result
            output[path] = content
        elif isinstance(result, Exception):
            logger.debug("Error reading file", error=str(result))

    return output


def get_async_cache_stats() -> dict:
    """Get async file cache statistics."""
    return _async_file_cache.get_stats()


def clear_async_cache() -> None:
    """Clear the async file cache."""
    _async_file_cache.clear()


def invalidate_async_cache(file_path: str | Path) -> bool:
    """Invalidate a file from async cache.

    Args:
        file_path: Path to invalidate.

    Returns:
        True if file was in cache.
    """
    return _async_file_cache.invalidate(str(file_path))
