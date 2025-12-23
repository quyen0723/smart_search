"""Tests for graph builder."""

from pathlib import Path

import pytest

from smart_search.graph.builder import GraphBuilder
from smart_search.graph.engine import CodeGraph
from smart_search.graph.models import EdgeType, NodeData, NodeType
from smart_search.parsing.models import (
    CodeUnit,
    CodeUnitType,
    Language,
    ParsedFile,
    Position,
    Span,
)


def make_span(start_line: int = 1, end_line: int = 5) -> Span:
    """Helper to create a Span."""
    return Span(
        start=Position(line=start_line, column=0),
        end=Position(line=end_line, column=0),
    )


def make_unit(
    id: str,
    name: str,
    unit_type: CodeUnitType = CodeUnitType.FUNCTION,
    parent_id: str | None = None,
    base_classes: list[str] | None = None,
    calls: list[str] | None = None,
) -> CodeUnit:
    """Helper to create a CodeUnit."""
    return CodeUnit(
        id=id,
        name=name,
        qualified_name=name,
        type=unit_type,
        file_path=Path("test.py"),
        span=make_span(),
        language=Language.PYTHON,
        content=f"# {name}",
        parent_id=parent_id,
        base_classes=base_classes or [],
        calls=calls or [],
    )


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    def test_empty_build(self) -> None:
        """Test building from empty list."""
        builder = GraphBuilder()
        graph = builder.build_from_units([])
        assert graph.node_count == 0

    def test_build_single_function(self) -> None:
        """Test building with single function."""
        builder = GraphBuilder()
        unit = make_unit("test.py::func", "func")
        graph = builder.build_from_units([unit])
        assert graph.node_count == 1
        assert graph.has_node("test.py::func")

    def test_build_single_class(self) -> None:
        """Test building with single class."""
        builder = GraphBuilder()
        unit = make_unit("test.py::MyClass", "MyClass", CodeUnitType.CLASS)
        graph = builder.build_from_units([unit])
        node = graph.get_node("test.py::MyClass")
        assert node is not None
        assert node.data.node_type == NodeType.CLASS

    def test_build_with_parent_child(self) -> None:
        """Test building with parent-child relationship."""
        builder = GraphBuilder()
        parent = make_unit("test.py::Parent", "Parent", CodeUnitType.CLASS)
        method = make_unit(
            "test.py::Parent.method",
            "method",
            CodeUnitType.METHOD,
            parent_id="test.py::Parent",
        )

        graph = builder.build_from_units([parent, method])
        assert graph.has_edge("test.py::Parent", "test.py::Parent.method")
        edge = graph.get_edge("test.py::Parent", "test.py::Parent.method")
        assert edge.data.edge_type == EdgeType.CONTAINS

    def test_build_with_inheritance(self) -> None:
        """Test building with inheritance relationship."""
        builder = GraphBuilder()
        base = make_unit("test.py::Base", "Base", CodeUnitType.CLASS)
        child = make_unit(
            "test.py::Child",
            "Child",
            CodeUnitType.CLASS,
            base_classes=["Base"],
        )

        graph = builder.build_from_units([base, child])
        assert graph.has_edge("test.py::Child", "test.py::Base")
        edge = graph.get_edge("test.py::Child", "test.py::Base")
        assert edge.data.edge_type == EdgeType.INHERITS

    def test_build_with_external_base(self) -> None:
        """Test building with external base class."""
        builder = GraphBuilder()
        child = make_unit(
            "test.py::MyException",
            "MyException",
            CodeUnitType.CLASS,
            base_classes=["Exception"],
        )

        graph = builder.build_from_units([child])
        # Should create an external node
        assert graph.has_node("external::Exception")
        assert graph.has_edge("test.py::MyException", "external::Exception")

    def test_build_with_calls(self) -> None:
        """Test building with call relationships."""
        builder = GraphBuilder()
        caller = make_unit("test.py::caller", "caller", calls=["callee"])
        callee = make_unit("test.py::callee", "callee")

        graph = builder.build_from_units([caller, callee])
        assert graph.has_edge("test.py::caller", "test.py::callee")
        edge = graph.get_edge("test.py::caller", "test.py::callee")
        assert edge.data.edge_type == EdgeType.CALLS

    def test_add_unit_to_existing_graph(self) -> None:
        """Test adding a unit to existing graph."""
        builder = GraphBuilder()
        unit = make_unit("test.py::func", "func")
        builder.add_unit(unit)
        assert builder.graph.has_node("test.py::func")

    def test_add_edge_directly(self) -> None:
        """Test adding an edge directly."""
        builder = GraphBuilder()
        builder.add_unit(make_unit("a", "a"))
        builder.add_unit(make_unit("b", "b"))

        result = builder.add_edge("a", "b", EdgeType.CALLS)
        assert result is True
        assert builder.graph.has_edge("a", "b")

    def test_node_type_mapping(self) -> None:
        """Test that code unit types map to correct node types."""
        builder = GraphBuilder()
        units = [
            make_unit("mod", "mod", CodeUnitType.MODULE),
            make_unit("cls", "Cls", CodeUnitType.CLASS),
            make_unit("func", "func", CodeUnitType.FUNCTION),
        ]

        graph = builder.build_from_units(units)

        mod_node = graph.get_node("mod")
        assert mod_node.data.node_type == NodeType.MODULE

        cls_node = graph.get_node("cls")
        assert cls_node.data.node_type == NodeType.CLASS

        func_node = graph.get_node("func")
        assert func_node.data.node_type == NodeType.FUNCTION


class TestGraphBuilderParsedFiles:
    """Tests for building from parsed files."""

    def test_build_from_parsed_file(self) -> None:
        """Test building from parsed files."""
        builder = GraphBuilder()
        parsed_file = ParsedFile(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            content="# test",
            content_hash="abc123",
            imports=["os", "sys"],
            module_docstring="A test module.",
            units=[make_unit("test.py::func", "func")],
        )

        graph = builder.build_from_parsed_files([parsed_file])

        # Should have module node
        assert graph.has_node("test.py::module")
        module_node = graph.get_node("test.py::module")
        assert module_node.data.node_type == NodeType.MODULE

        # Should have function node
        assert graph.has_node("test.py::func")

    def test_import_edges(self) -> None:
        """Test that import edges are created."""
        builder = GraphBuilder()
        parsed_file = ParsedFile(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            content="# test",
            content_hash="abc123",
            imports=["os", "sys"],
            units=[],
        )

        graph = builder.build_from_parsed_files([parsed_file])

        # Should have import nodes
        assert graph.has_node("import::os")
        assert graph.has_node("import::sys")

        # Should have import edges
        assert graph.has_edge("test.py::module", "import::os")
        edge = graph.get_edge("test.py::module", "import::os")
        assert edge.data.edge_type == EdgeType.IMPORTS


class TestGraphBuilderWithExistingGraph:
    """Tests for building on existing graph."""

    def test_build_on_existing_graph(self) -> None:
        """Test building on existing graph."""
        # Create initial graph
        existing = CodeGraph()
        existing.add_node(
            NodeData(
                id="existing",
                name="existing",
                qualified_name="existing",
                node_type=NodeType.FUNCTION,
            )
        )

        # Build on existing
        builder = GraphBuilder(existing)
        unit = make_unit("new_func", "new_func")
        builder.build_from_units([unit])

        assert builder.graph.has_node("existing")
        assert builder.graph.has_node("new_func")
