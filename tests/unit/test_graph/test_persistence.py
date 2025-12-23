"""Tests for graph persistence."""

import gzip
import json
from pathlib import Path

import pytest

from smart_search.graph.engine import CodeGraph
from smart_search.graph.models import EdgeData, EdgeType, NodeData, NodeType
from smart_search.graph.persistence import (
    GraphCache,
    GraphPersistence,
    create_persistence,
)


class TestGraphPersistence:
    """Tests for GraphPersistence class."""

    @pytest.fixture
    def persistence(self, tmp_path: Path) -> GraphPersistence:
        """Create a persistence instance with temp directory."""
        return GraphPersistence(tmp_path)

    @pytest.fixture
    def sample_graph(self) -> CodeGraph:
        """Create a sample graph."""
        graph = CodeGraph()
        graph.add_node(
            NodeData(
                id="func1",
                name="func1",
                qualified_name="test.func1",
                node_type=NodeType.FUNCTION,
                file_path=Path("test.py"),
            )
        )
        graph.add_node(
            NodeData(
                id="func2",
                name="func2",
                qualified_name="test.func2",
                node_type=NodeType.FUNCTION,
            )
        )
        graph.add_edge("func1", "func2", EdgeData(edge_type=EdgeType.CALLS))
        return graph

    def test_save_json(
        self, persistence: GraphPersistence, sample_graph: CodeGraph, tmp_path: Path
    ) -> None:
        """Test saving to JSON."""
        file_path = persistence.save_json(sample_graph, "test_graph")
        assert file_path.exists()
        assert file_path.name == "test_graph.json"

        # Verify content
        with open(file_path, "r") as f:
            data = json.load(f)
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1

    def test_save_json_compressed(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test saving to compressed JSON."""
        file_path = persistence.save_json(sample_graph, "test_graph", compress=True)
        assert file_path.exists()
        assert file_path.name == "test_graph.json.gz"

        # Verify content
        with gzip.open(file_path, "rt") as f:
            data = json.load(f)
        assert len(data["nodes"]) == 2

    def test_load_json(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test loading from JSON."""
        file_path = persistence.save_json(sample_graph, "test_graph")
        loaded = persistence.load_json(file_path)

        assert loaded.node_count == 2
        assert loaded.edge_count == 1
        assert loaded.has_node("func1")
        assert loaded.has_edge("func1", "func2")

    def test_load_json_compressed(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test loading from compressed JSON."""
        file_path = persistence.save_json(sample_graph, "test_graph", compress=True)
        loaded = persistence.load_json(file_path)

        assert loaded.node_count == 2
        assert loaded.has_node("func1")

    def test_load_json_not_found(self, persistence: GraphPersistence) -> None:
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            persistence.load_json(Path("nonexistent.json"))

    def test_save_incremental(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test saving with metadata."""
        metadata = {"version": "1.0", "indexed_at": "2024-01-01"}
        file_path = persistence.save_incremental(
            sample_graph, "test_graph", metadata=metadata
        )
        assert file_path.exists()
        assert file_path.name == "test_graph.smartgraph.json.gz"

    def test_load_incremental(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test loading with metadata."""
        metadata = {"version": "1.0", "indexed_at": "2024-01-01"}
        file_path = persistence.save_incremental(
            sample_graph, "test_graph", metadata=metadata
        )

        loaded_graph, loaded_metadata = persistence.load_incremental(file_path)
        assert loaded_graph.node_count == 2
        assert loaded_metadata["version"] == "1.0"
        assert loaded_metadata["indexed_at"] == "2024-01-01"

    def test_load_incremental_not_found(self, persistence: GraphPersistence) -> None:
        """Test loading non-existent incremental file."""
        with pytest.raises(FileNotFoundError):
            persistence.load_incremental(Path("nonexistent.smartgraph.json.gz"))

    def test_exists(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test checking if graph exists."""
        assert not persistence.exists("test_graph")

        persistence.save_json(sample_graph, "test_graph")
        assert persistence.exists("test_graph")

    def test_delete(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test deleting a graph."""
        persistence.save_json(sample_graph, "test_graph")
        assert persistence.exists("test_graph")

        result = persistence.delete("test_graph")
        assert result is True
        assert not persistence.exists("test_graph")

    def test_delete_nonexistent(self, persistence: GraphPersistence) -> None:
        """Test deleting non-existent graph."""
        result = persistence.delete("nonexistent")
        assert result is False

    def test_list_graphs(
        self, persistence: GraphPersistence, sample_graph: CodeGraph
    ) -> None:
        """Test listing saved graphs."""
        persistence.save_json(sample_graph, "graph1")
        persistence.save_json(sample_graph, "graph2", compress=True)
        persistence.save_incremental(sample_graph, "graph3")

        graphs = persistence.list_graphs()
        assert "graph1" in graphs
        assert "graph2" in graphs
        assert "graph3" in graphs


class TestGraphCache:
    """Tests for GraphCache class."""

    @pytest.fixture
    def cache(self) -> GraphCache:
        """Create a cache instance."""
        return GraphCache(max_size=3)

    @pytest.fixture
    def sample_graph(self) -> CodeGraph:
        """Create a sample graph."""
        graph = CodeGraph()
        graph.add_node(
            NodeData(
                id="test",
                name="test",
                qualified_name="test",
                node_type=NodeType.FUNCTION,
            )
        )
        return graph

    def test_put_and_get(
        self, cache: GraphCache, sample_graph: CodeGraph
    ) -> None:
        """Test putting and getting from cache."""
        cache.put("key1", sample_graph)
        result = cache.get("key1")
        assert result is not None
        assert result.node_count == 1

    def test_get_nonexistent(self, cache: GraphCache) -> None:
        """Test getting non-existent key."""
        result = cache.get("nonexistent")
        assert result is None

    def test_eviction(self, cache: GraphCache, sample_graph: CodeGraph) -> None:
        """Test LRU eviction."""
        # Fill cache
        cache.put("key1", sample_graph)
        cache.put("key2", sample_graph)
        cache.put("key3", sample_graph)

        # Access key1 to make it recently used
        cache.get("key1")

        # Add new item, should evict key2 (least recently used)
        cache.put("key4", sample_graph)

        assert cache.get("key1") is not None
        assert cache.get("key2") is None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_invalidate(self, cache: GraphCache, sample_graph: CodeGraph) -> None:
        """Test invalidating a cached graph."""
        cache.put("key1", sample_graph)
        assert cache.get("key1") is not None

        result = cache.invalidate("key1")
        assert result is True
        assert cache.get("key1") is None

    def test_invalidate_nonexistent(self, cache: GraphCache) -> None:
        """Test invalidating non-existent key."""
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear(self, cache: GraphCache, sample_graph: CodeGraph) -> None:
        """Test clearing the cache."""
        cache.put("key1", sample_graph)
        cache.put("key2", sample_graph)

        cache.clear()
        assert cache.size == 0
        assert cache.get("key1") is None

    def test_size(self, cache: GraphCache, sample_graph: CodeGraph) -> None:
        """Test cache size."""
        assert cache.size == 0

        cache.put("key1", sample_graph)
        assert cache.size == 1

        cache.put("key2", sample_graph)
        assert cache.size == 2

    def test_keys(self, cache: GraphCache, sample_graph: CodeGraph) -> None:
        """Test getting cache keys."""
        cache.put("key1", sample_graph)
        cache.put("key2", sample_graph)

        keys = cache.keys()
        assert "key1" in keys
        assert "key2" in keys

    def test_update_existing(self, cache: GraphCache) -> None:
        """Test updating existing entry."""
        graph1 = CodeGraph()
        graph1.add_node(
            NodeData(
                id="node1",
                name="node1",
                qualified_name="node1",
                node_type=NodeType.FUNCTION,
            )
        )

        graph2 = CodeGraph()
        graph2.add_node(
            NodeData(
                id="node2",
                name="node2",
                qualified_name="node2",
                node_type=NodeType.FUNCTION,
            )
        )

        cache.put("key1", graph1)
        cache.put("key1", graph2)

        # Should have the updated graph
        result = cache.get("key1")
        assert result.has_node("node2")
        assert not result.has_node("node1")

        # Size should still be 1
        assert cache.size == 1


class TestCreatePersistence:
    """Tests for create_persistence function."""

    def test_create_with_path(self, tmp_path: Path) -> None:
        """Test creating persistence with path."""
        persistence = create_persistence(tmp_path)
        assert persistence.storage_dir == tmp_path

    def test_create_with_string_path(self, tmp_path: Path) -> None:
        """Test creating persistence with string path."""
        persistence = create_persistence(str(tmp_path))
        assert persistence.storage_dir == tmp_path

    def test_create_without_path(self) -> None:
        """Test creating persistence without path."""
        persistence = create_persistence()
        assert persistence.storage_dir == Path.cwd()
