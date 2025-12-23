"""Graph persistence for saving and loading code graphs.

Provides serialization to JSON and binary formats.
"""

import gzip
import json
from pathlib import Path
from typing import Any

from smart_search.graph.engine import CodeGraph
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class GraphPersistence:
    """Handles saving and loading of code graphs."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        """Initialize persistence handler.

        Args:
            storage_dir: Directory for storing graphs. Defaults to current dir.
        """
        self.storage_dir = storage_dir or Path.cwd()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_json(
        self,
        graph: CodeGraph,
        filename: str,
        compress: bool = False,
    ) -> Path:
        """Save graph to JSON file.

        Args:
            graph: The graph to save.
            filename: Output filename (without extension).
            compress: Whether to gzip compress the output.

        Returns:
            Path to the saved file.
        """
        data = graph.to_dict()

        if compress:
            file_path = self.storage_dir / f"{filename}.json.gz"
            with gzip.open(file_path, "wt", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        else:
            file_path = self.storage_dir / f"{filename}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

        logger.info(
            "Saved graph to JSON",
            file_path=str(file_path),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
            compressed=compress,
        )
        return file_path

    def load_json(self, file_path: Path | str) -> CodeGraph:
        """Load graph from JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            The loaded CodeGraph.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is invalid JSON.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")

        if file_path.suffix == ".gz" or str(file_path).endswith(".json.gz"):
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                data = json.load(f)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

        graph = CodeGraph.from_dict(data)

        logger.info(
            "Loaded graph from JSON",
            file_path=str(file_path),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
        )
        return graph

    def save_incremental(
        self,
        graph: CodeGraph,
        filename: str,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save graph with metadata for incremental updates.

        Args:
            graph: The graph to save.
            filename: Output filename (without extension).
            metadata: Additional metadata to store.

        Returns:
            Path to the saved file.
        """
        data = {
            "version": "1.0",
            "metadata": metadata or {},
            "graph": graph.to_dict(),
        }

        file_path = self.storage_dir / f"{filename}.smartgraph.json.gz"
        with gzip.open(file_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        logger.info(
            "Saved incremental graph",
            file_path=str(file_path),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
        )
        return file_path

    def load_incremental(
        self,
        file_path: Path | str,
    ) -> tuple[CodeGraph, dict[str, Any]]:
        """Load graph with metadata.

        Args:
            file_path: Path to the incremental graph file.

        Returns:
            Tuple of (CodeGraph, metadata dict).

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Graph file not found: {file_path}")

        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        graph = CodeGraph.from_dict(data.get("graph", {}))
        metadata = data.get("metadata", {})

        logger.info(
            "Loaded incremental graph",
            file_path=str(file_path),
            node_count=graph.node_count,
            edge_count=graph.edge_count,
        )
        return graph, metadata

    def exists(self, filename: str) -> bool:
        """Check if a graph file exists.

        Args:
            filename: The filename to check.

        Returns:
            True if the file exists.
        """
        for ext in [".json", ".json.gz", ".smartgraph.json.gz"]:
            if (self.storage_dir / f"{filename}{ext}").exists():
                return True
        return False

    def delete(self, filename: str) -> bool:
        """Delete a graph file.

        Args:
            filename: The filename to delete.

        Returns:
            True if a file was deleted.
        """
        deleted = False
        for ext in [".json", ".json.gz", ".smartgraph.json.gz"]:
            file_path = self.storage_dir / f"{filename}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True
                logger.info("Deleted graph file", file_path=str(file_path))
        return deleted

    def list_graphs(self) -> list[str]:
        """List all saved graphs.

        Returns:
            List of graph filenames (without extensions).
        """
        graphs = set()
        for ext in ["*.json", "*.json.gz", "*.smartgraph.json.gz"]:
            for file_path in self.storage_dir.glob(ext):
                name = file_path.name
                # Remove extensions
                for suffix in [".smartgraph.json.gz", ".json.gz", ".json"]:
                    if name.endswith(suffix):
                        name = name[: -len(suffix)]
                        break
                graphs.add(name)
        return sorted(graphs)


class GraphCache:
    """In-memory cache for code graphs."""

    def __init__(self, max_size: int = 10) -> None:
        """Initialize the cache.

        Args:
            max_size: Maximum number of graphs to cache.
        """
        self.max_size = max_size
        self._cache: dict[str, CodeGraph] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> CodeGraph | None:
        """Get a graph from cache.

        Args:
            key: The cache key.

        Returns:
            The cached graph or None.
        """
        if key in self._cache:
            # Move to end of access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, graph: CodeGraph) -> None:
        """Put a graph in cache.

        Args:
            key: The cache key.
            graph: The graph to cache.
        """
        if key in self._cache:
            # Update existing entry
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            # Evict least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
            logger.debug("Evicted graph from cache", key=lru_key)

        self._cache[key] = graph
        self._access_order.append(key)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached graph.

        Args:
            key: The cache key.

        Returns:
            True if the key was in cache.
        """
        if key in self._cache:
            del self._cache[key]
            self._access_order.remove(key)
            return True
        return False

    def clear(self) -> None:
        """Clear all cached graphs."""
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        """Get the number of cached graphs."""
        return len(self._cache)

    def keys(self) -> list[str]:
        """Get all cache keys."""
        return list(self._cache.keys())


def create_persistence(storage_dir: Path | str | None = None) -> GraphPersistence:
    """Create a GraphPersistence instance.

    Args:
        storage_dir: Storage directory path.

    Returns:
        GraphPersistence instance.
    """
    if storage_dir is not None:
        storage_dir = Path(storage_dir)
    return GraphPersistence(storage_dir)
