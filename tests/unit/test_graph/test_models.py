"""Tests for graph models."""

from pathlib import Path

import pytest

from smart_search.graph.models import (
    EdgeData,
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphStats,
    ImpactAnalysisResult,
    NodeData,
    NodeType,
    PathResult,
    SubgraphResult,
    TraversalOptions,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_types_exist(self) -> None:
        """Test all expected node types exist."""
        assert NodeType.MODULE == "module"
        assert NodeType.CLASS == "class"
        assert NodeType.FUNCTION == "function"
        assert NodeType.METHOD == "method"
        assert NodeType.VARIABLE == "variable"
        assert NodeType.IMPORT == "import"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_call_edges(self) -> None:
        """Test call relationship edges."""
        assert EdgeType.CALLS == "calls"
        assert EdgeType.CALLED_BY == "called_by"

    def test_import_edges(self) -> None:
        """Test import relationship edges."""
        assert EdgeType.IMPORTS == "imports"
        assert EdgeType.IMPORTED_BY == "imported_by"

    def test_inheritance_edges(self) -> None:
        """Test inheritance edges."""
        assert EdgeType.INHERITS == "inherits"
        assert EdgeType.INHERITED_BY == "inherited_by"

    def test_containment_edges(self) -> None:
        """Test containment edges."""
        assert EdgeType.CONTAINS == "contains"
        assert EdgeType.CONTAINED_BY == "contained_by"


class TestNodeData:
    """Tests for NodeData class."""

    def test_basic_creation(self) -> None:
        """Test basic node data creation."""
        node = NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
        )
        assert node.id == "test::func"
        assert node.name == "func"
        assert node.node_type == NodeType.FUNCTION

    def test_with_file_path(self) -> None:
        """Test node data with file path."""
        node = NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
            file_path=Path("test.py"),
            line_start=10,
            line_end=20,
        )
        assert node.file_path == Path("test.py")
        assert node.line_start == 10
        assert node.line_end == 20

    def test_with_metadata(self) -> None:
        """Test node data with metadata."""
        node = NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
            metadata={"signature": "func(x: int) -> int"},
        )
        assert node.metadata["signature"] == "func(x: int) -> int"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        node = NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
            file_path=Path("test.py"),
        )
        d = node.to_dict()
        assert d["id"] == "test::func"
        assert d["node_type"] == "function"
        assert d["file_path"] == "test.py"

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {
            "id": "test::func",
            "name": "func",
            "qualified_name": "test.func",
            "node_type": "function",
            "file_path": "test.py",
        }
        node = NodeData.from_dict(d)
        assert node.id == "test::func"
        assert node.node_type == NodeType.FUNCTION
        assert node.file_path == Path("test.py")

    def test_from_dict_without_file_path(self) -> None:
        """Test deserialization without file path."""
        d = {
            "id": "external::SomeClass",
            "name": "SomeClass",
            "qualified_name": "SomeClass",
            "node_type": "class",
        }
        node = NodeData.from_dict(d)
        assert node.file_path is None


class TestEdgeData:
    """Tests for EdgeData class."""

    def test_basic_creation(self) -> None:
        """Test basic edge data creation."""
        edge = EdgeData(edge_type=EdgeType.CALLS)
        assert edge.edge_type == EdgeType.CALLS
        assert edge.weight == 1.0

    def test_with_weight(self) -> None:
        """Test edge data with custom weight."""
        edge = EdgeData(edge_type=EdgeType.CALLS, weight=2.5)
        assert edge.weight == 2.5

    def test_with_metadata(self) -> None:
        """Test edge data with metadata."""
        edge = EdgeData(
            edge_type=EdgeType.CALLS,
            metadata={"line": 42},
        )
        assert edge.metadata["line"] == 42

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        edge = EdgeData(edge_type=EdgeType.INHERITS, weight=1.5)
        d = edge.to_dict()
        assert d["edge_type"] == "inherits"
        assert d["weight"] == 1.5

    def test_from_dict(self) -> None:
        """Test deserialization from dict."""
        d = {"edge_type": "calls", "weight": 2.0}
        edge = EdgeData.from_dict(d)
        assert edge.edge_type == EdgeType.CALLS
        assert edge.weight == 2.0


class TestGraphNode:
    """Tests for GraphNode class."""

    def test_creation(self) -> None:
        """Test graph node creation."""
        data = NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
        )
        node = GraphNode(index=0, data=data)
        assert node.index == 0
        assert node.data.id == "test::func"


class TestGraphEdge:
    """Tests for GraphEdge class."""

    def test_creation(self) -> None:
        """Test graph edge creation."""
        data = EdgeData(edge_type=EdgeType.CALLS)
        edge = GraphEdge(source=0, target=1, data=data)
        assert edge.source == 0
        assert edge.target == 1
        assert edge.data.edge_type == EdgeType.CALLS


class TestTraversalOptions:
    """Tests for TraversalOptions class."""

    def test_defaults(self) -> None:
        """Test default options."""
        opts = TraversalOptions()
        assert opts.max_depth == -1
        assert opts.edge_types is None
        assert opts.node_types is None
        assert opts.include_self is False
        assert opts.direction == "outgoing"

    def test_with_filters(self) -> None:
        """Test options with filters."""
        opts = TraversalOptions(
            max_depth=3,
            edge_types=[EdgeType.CALLS],
            node_types=[NodeType.FUNCTION],
            include_self=True,
            direction="both",
        )
        assert opts.max_depth == 3
        assert EdgeType.CALLS in opts.edge_types
        assert NodeType.FUNCTION in opts.node_types


class TestPathResult:
    """Tests for PathResult class."""

    def test_empty_path(self) -> None:
        """Test empty path result."""
        path = PathResult(nodes=[], edges=[])
        assert len(path.nodes) == 0
        assert path.total_weight == 0.0

    def test_with_weight(self) -> None:
        """Test path with total weight."""
        path = PathResult(nodes=[], edges=[], total_weight=5.5)
        assert path.total_weight == 5.5


class TestSubgraphResult:
    """Tests for SubgraphResult class."""

    def test_empty_subgraph(self) -> None:
        """Test empty subgraph."""
        subgraph = SubgraphResult(nodes=[], edges=[])
        assert len(subgraph.nodes) == 0
        assert subgraph.center_node is None


class TestGraphStats:
    """Tests for GraphStats class."""

    def test_basic_stats(self) -> None:
        """Test basic graph stats."""
        stats = GraphStats(node_count=10, edge_count=15)
        assert stats.node_count == 10
        assert stats.edge_count == 15
        assert stats.density == 0.0

    def test_with_type_counts(self) -> None:
        """Test stats with type counts."""
        stats = GraphStats(
            node_count=10,
            edge_count=15,
            nodes_by_type={"function": 5, "class": 3},
            edges_by_type={"calls": 10, "inherits": 5},
        )
        assert stats.nodes_by_type["function"] == 5
        assert stats.edges_by_type["calls"] == 10

    def test_to_dict(self) -> None:
        """Test stats serialization."""
        stats = GraphStats(
            node_count=10,
            edge_count=15,
            connected_components=2,
            density=0.15,
        )
        d = stats.to_dict()
        assert d["node_count"] == 10
        assert d["connected_components"] == 2
        assert d["density"] == 0.15


class TestImpactAnalysisResult:
    """Tests for ImpactAnalysisResult class."""

    def test_empty_result(self) -> None:
        """Test empty impact result."""
        data = NodeData(
            id="test::func",
            name="func",
            qualified_name="test.func",
            node_type=NodeType.FUNCTION,
        )
        node = GraphNode(index=0, data=data)
        result = ImpactAnalysisResult(
            source_node=node,
            affected_nodes=[],
            impact_paths=[],
        )
        assert result.impact_score == 0.0
        assert len(result.affected_nodes) == 0
