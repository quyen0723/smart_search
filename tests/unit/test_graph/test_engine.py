"""Tests for graph engine."""

from pathlib import Path

import pytest

from smart_search.graph.engine import CodeGraph
from smart_search.graph.models import (
    EdgeData,
    EdgeType,
    NodeData,
    NodeType,
    TraversalOptions,
)


class TestCodeGraphBasics:
    """Tests for basic graph operations."""

    @pytest.fixture
    def graph(self) -> CodeGraph:
        """Create an empty graph."""
        return CodeGraph()

    @pytest.fixture
    def sample_node(self) -> NodeData:
        """Create a sample node."""
        return NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
            file_path=Path("test.py"),
        )

    def test_empty_graph(self, graph: CodeGraph) -> None:
        """Test empty graph properties."""
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_add_node(self, graph: CodeGraph, sample_node: NodeData) -> None:
        """Test adding a node."""
        index = graph.add_node(sample_node)
        assert index == 0
        assert graph.node_count == 1
        assert graph.has_node("test::func")

    def test_add_duplicate_node_raises(
        self, graph: CodeGraph, sample_node: NodeData
    ) -> None:
        """Test that adding duplicate node raises error."""
        graph.add_node(sample_node)
        with pytest.raises(ValueError, match="already exists"):
            graph.add_node(sample_node)

    def test_add_node_if_not_exists(
        self, graph: CodeGraph, sample_node: NodeData
    ) -> None:
        """Test add_node_if_not_exists."""
        index1 = graph.add_node_if_not_exists(sample_node)
        index2 = graph.add_node_if_not_exists(sample_node)
        assert index1 == index2
        assert graph.node_count == 1

    def test_get_node(self, graph: CodeGraph, sample_node: NodeData) -> None:
        """Test getting a node."""
        graph.add_node(sample_node)
        node = graph.get_node("test::func")
        assert node is not None
        assert node.data.name == "func"

    def test_get_node_not_found(self, graph: CodeGraph) -> None:
        """Test getting a non-existent node."""
        node = graph.get_node("nonexistent")
        assert node is None

    def test_get_node_by_index(
        self, graph: CodeGraph, sample_node: NodeData
    ) -> None:
        """Test getting a node by index."""
        index = graph.add_node(sample_node)
        node = graph.get_node_by_index(index)
        assert node is not None
        assert node.data.id == "test::func"

    def test_remove_node(self, graph: CodeGraph, sample_node: NodeData) -> None:
        """Test removing a node."""
        graph.add_node(sample_node)
        assert graph.remove_node("test::func")
        assert not graph.has_node("test::func")
        assert graph.node_count == 0

    def test_remove_nonexistent_node(self, graph: CodeGraph) -> None:
        """Test removing a non-existent node."""
        assert not graph.remove_node("nonexistent")


class TestCodeGraphEdges:
    """Tests for graph edge operations."""

    @pytest.fixture
    def graph_with_nodes(self) -> CodeGraph:
        """Create a graph with two nodes."""
        graph = CodeGraph()
        graph.add_node(
            NodeData(
                id="a",
                name="a",
                qualified_name="a",
                node_type=NodeType.FUNCTION,
            )
        )
        graph.add_node(
            NodeData(
                id="b",
                name="b",
                qualified_name="b",
                node_type=NodeType.FUNCTION,
            )
        )
        return graph

    def test_add_edge(self, graph_with_nodes: CodeGraph) -> None:
        """Test adding an edge."""
        edge_data = EdgeData(edge_type=EdgeType.CALLS)
        result = graph_with_nodes.add_edge("a", "b", edge_data)
        assert result is True
        assert graph_with_nodes.edge_count == 1

    def test_add_edge_missing_source(self, graph_with_nodes: CodeGraph) -> None:
        """Test adding edge with missing source."""
        edge_data = EdgeData(edge_type=EdgeType.CALLS)
        result = graph_with_nodes.add_edge("nonexistent", "b", edge_data)
        assert result is False

    def test_add_edge_missing_target(self, graph_with_nodes: CodeGraph) -> None:
        """Test adding edge with missing target."""
        edge_data = EdgeData(edge_type=EdgeType.CALLS)
        result = graph_with_nodes.add_edge("a", "nonexistent", edge_data)
        assert result is False

    def test_has_edge(self, graph_with_nodes: CodeGraph) -> None:
        """Test checking edge existence."""
        edge_data = EdgeData(edge_type=EdgeType.CALLS)
        graph_with_nodes.add_edge("a", "b", edge_data)
        assert graph_with_nodes.has_edge("a", "b")
        assert not graph_with_nodes.has_edge("b", "a")

    def test_get_edge(self, graph_with_nodes: CodeGraph) -> None:
        """Test getting an edge."""
        edge_data = EdgeData(edge_type=EdgeType.CALLS)
        graph_with_nodes.add_edge("a", "b", edge_data)
        edge = graph_with_nodes.get_edge("a", "b")
        assert edge is not None
        assert edge.data.edge_type == EdgeType.CALLS

    def test_get_edge_not_found(self, graph_with_nodes: CodeGraph) -> None:
        """Test getting a non-existent edge."""
        edge = graph_with_nodes.get_edge("a", "b")
        assert edge is None


class TestCodeGraphTraversal:
    """Tests for graph traversal."""

    @pytest.fixture
    def graph_chain(self) -> CodeGraph:
        """Create a chain graph: a -> b -> c -> d."""
        graph = CodeGraph()
        for name in ["a", "b", "c", "d"]:
            graph.add_node(
                NodeData(
                    id=name,
                    name=name,
                    qualified_name=name,
                    node_type=NodeType.FUNCTION,
                )
            )
        graph.add_edge("a", "b", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("b", "c", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("c", "d", EdgeData(edge_type=EdgeType.CALLS))
        return graph

    def test_get_successors(self, graph_chain: CodeGraph) -> None:
        """Test getting successors."""
        successors = graph_chain.get_successors("a")
        assert len(successors) == 1
        assert successors[0].data.id == "b"

    def test_get_predecessors(self, graph_chain: CodeGraph) -> None:
        """Test getting predecessors."""
        predecessors = graph_chain.get_predecessors("b")
        assert len(predecessors) == 1
        assert predecessors[0].data.id == "a"

    def test_get_neighbors(self, graph_chain: CodeGraph) -> None:
        """Test getting all neighbors."""
        neighbors = graph_chain.get_neighbors("b")
        ids = {n.data.id for n in neighbors}
        assert "a" in ids
        assert "c" in ids

    def test_traverse_outgoing(self, graph_chain: CodeGraph) -> None:
        """Test outgoing traversal."""
        options = TraversalOptions(direction="outgoing")
        nodes = graph_chain.traverse("a", options)
        ids = [n.data.id for n in nodes]
        assert "b" in ids
        assert "c" in ids
        assert "d" in ids

    def test_traverse_with_depth(self, graph_chain: CodeGraph) -> None:
        """Test traversal with depth limit."""
        options = TraversalOptions(max_depth=1, direction="outgoing")
        nodes = graph_chain.traverse("a", options)
        ids = [n.data.id for n in nodes]
        assert "b" in ids
        assert "c" not in ids

    def test_traverse_incoming(self, graph_chain: CodeGraph) -> None:
        """Test incoming traversal."""
        options = TraversalOptions(direction="incoming")
        nodes = graph_chain.traverse("d", options)
        ids = [n.data.id for n in nodes]
        assert "c" in ids
        assert "b" in ids
        assert "a" in ids

    def test_traverse_both(self, graph_chain: CodeGraph) -> None:
        """Test bidirectional traversal."""
        options = TraversalOptions(direction="both", max_depth=1)
        nodes = graph_chain.traverse("b", options)
        ids = [n.data.id for n in nodes]
        assert "a" in ids
        assert "c" in ids

    def test_traverse_with_edge_filter(self, graph_chain: CodeGraph) -> None:
        """Test traversal with edge type filter."""
        options = TraversalOptions(
            edge_types=[EdgeType.INHERITS],  # No inheritance edges exist
            direction="outgoing",
        )
        nodes = graph_chain.traverse("a", options)
        assert len(nodes) == 0


class TestCodeGraphPaths:
    """Tests for path finding."""

    @pytest.fixture
    def graph_with_paths(self) -> CodeGraph:
        """Create a graph with multiple paths."""
        graph = CodeGraph()
        for name in ["a", "b", "c", "d"]:
            graph.add_node(
                NodeData(
                    id=name,
                    name=name,
                    qualified_name=name,
                    node_type=NodeType.FUNCTION,
                )
            )
        # a -> b -> d
        # a -> c -> d
        graph.add_edge("a", "b", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("b", "d", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("a", "c", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("c", "d", EdgeData(edge_type=EdgeType.CALLS))
        return graph

    def test_find_shortest_path(self, graph_with_paths: CodeGraph) -> None:
        """Test finding shortest path."""
        result = graph_with_paths.find_shortest_path("a", "d")
        assert result is not None
        assert len(result.nodes) == 3

    def test_find_shortest_path_no_path(self, graph_with_paths: CodeGraph) -> None:
        """Test no path between nodes."""
        result = graph_with_paths.find_shortest_path("d", "a")
        assert result is None

    def test_find_all_paths(self, graph_with_paths: CodeGraph) -> None:
        """Test finding all paths."""
        paths = graph_with_paths.find_all_paths("a", "d")
        assert len(paths) == 2

    def test_find_all_paths_with_depth(self, graph_with_paths: CodeGraph) -> None:
        """Test finding paths with max depth."""
        paths = graph_with_paths.find_all_paths("a", "d", max_depth=3)
        # Both paths have length 3 (a->b->d and a->c->d), so both should be found
        assert len(paths) == 2


class TestCodeGraphSubgraph:
    """Tests for subgraph extraction."""

    @pytest.fixture
    def complex_graph(self) -> CodeGraph:
        """Create a more complex graph."""
        graph = CodeGraph()
        # Create nodes
        for i in range(5):
            graph.add_node(
                NodeData(
                    id=f"node{i}",
                    name=f"node{i}",
                    qualified_name=f"node{i}",
                    node_type=NodeType.FUNCTION,
                )
            )
        # Create edges: 0 -> 1 -> 2, 0 -> 3, 3 -> 4
        graph.add_edge("node0", "node1", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("node1", "node2", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("node0", "node3", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("node3", "node4", EdgeData(edge_type=EdgeType.CALLS))
        return graph

    def test_get_subgraph(self, complex_graph: CodeGraph) -> None:
        """Test getting a subgraph."""
        result = complex_graph.get_subgraph("node0", max_depth=1)
        assert result.center_node is not None
        assert result.center_node.data.id == "node0"
        # Should include node0, node1, node3
        ids = [n.data.id for n in result.nodes]
        assert "node0" in ids
        assert "node1" in ids
        assert "node3" in ids


class TestCodeGraphStats:
    """Tests for graph statistics."""

    def test_stats_empty_graph(self) -> None:
        """Test stats for empty graph."""
        graph = CodeGraph()
        stats = graph.get_stats()
        assert stats.node_count == 0
        assert stats.edge_count == 0
        assert stats.density == 0.0

    def test_stats_with_nodes(self) -> None:
        """Test stats with nodes and edges."""
        graph = CodeGraph()
        graph.add_node(
            NodeData(
                id="func1",
                name="func1",
                qualified_name="func1",
                node_type=NodeType.FUNCTION,
            )
        )
        graph.add_node(
            NodeData(
                id="class1",
                name="class1",
                qualified_name="class1",
                node_type=NodeType.CLASS,
            )
        )
        graph.add_edge("func1", "class1", EdgeData(edge_type=EdgeType.CALLS))

        stats = graph.get_stats()
        assert stats.node_count == 2
        assert stats.edge_count == 1
        assert stats.nodes_by_type["function"] == 1
        assert stats.nodes_by_type["class"] == 1
        assert stats.edges_by_type["calls"] == 1


class TestCodeGraphSerialization:
    """Tests for graph serialization."""

    @pytest.fixture
    def populated_graph(self) -> CodeGraph:
        """Create a graph with nodes and edges."""
        graph = CodeGraph()
        graph.add_node(
            NodeData(
                id="a",
                name="a",
                qualified_name="a",
                node_type=NodeType.FUNCTION,
                file_path=Path("test.py"),
            )
        )
        graph.add_node(
            NodeData(
                id="b",
                name="b",
                qualified_name="b",
                node_type=NodeType.FUNCTION,
            )
        )
        graph.add_edge("a", "b", EdgeData(edge_type=EdgeType.CALLS))
        return graph

    def test_to_dict(self, populated_graph: CodeGraph) -> None:
        """Test serialization to dict."""
        d = populated_graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_from_dict(self, populated_graph: CodeGraph) -> None:
        """Test deserialization from dict."""
        d = populated_graph.to_dict()
        new_graph = CodeGraph.from_dict(d)
        assert new_graph.node_count == 2
        assert new_graph.edge_count == 1
        assert new_graph.has_node("a")
        assert new_graph.has_edge("a", "b")

    def test_clear(self, populated_graph: CodeGraph) -> None:
        """Test clearing the graph."""
        populated_graph.clear()
        assert populated_graph.node_count == 0
        assert populated_graph.edge_count == 0


class TestCodeGraphEdgeLists:
    """Tests for edge list operations."""

    @pytest.fixture
    def graph_with_edges(self) -> CodeGraph:
        """Create a graph with multiple edge types."""
        graph = CodeGraph()
        for name in ["a", "b", "c"]:
            graph.add_node(
                NodeData(
                    id=name,
                    name=name,
                    qualified_name=name,
                    node_type=NodeType.FUNCTION,
                )
            )
        graph.add_edge("a", "b", EdgeData(edge_type=EdgeType.CALLS))
        graph.add_edge("a", "c", EdgeData(edge_type=EdgeType.IMPORTS))
        graph.add_edge("b", "c", EdgeData(edge_type=EdgeType.CALLS))
        return graph

    def test_get_all_nodes(self, graph_with_edges: CodeGraph) -> None:
        """Test getting all nodes."""
        nodes = graph_with_edges.get_all_nodes()
        assert len(nodes) == 3

    def test_get_all_edges(self, graph_with_edges: CodeGraph) -> None:
        """Test getting all edges."""
        edges = graph_with_edges.get_all_edges()
        assert len(edges) == 3

    def test_get_nodes_by_type(self, graph_with_edges: CodeGraph) -> None:
        """Test filtering nodes by type."""
        nodes = graph_with_edges.get_nodes_by_type(NodeType.FUNCTION)
        assert len(nodes) == 3

    def test_get_edges_by_type(self, graph_with_edges: CodeGraph) -> None:
        """Test filtering edges by type."""
        edges = graph_with_edges.get_edges_by_type(EdgeType.CALLS)
        assert len(edges) == 2

    def test_get_outgoing_edges(self, graph_with_edges: CodeGraph) -> None:
        """Test getting outgoing edges."""
        edges = graph_with_edges.get_outgoing_edges("a")
        assert len(edges) == 2

    def test_get_incoming_edges(self, graph_with_edges: CodeGraph) -> None:
        """Test getting incoming edges."""
        edges = graph_with_edges.get_incoming_edges("c")
        assert len(edges) == 2
