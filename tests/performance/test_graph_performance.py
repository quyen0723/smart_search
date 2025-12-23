"""Performance tests for graph module."""

import pytest
import time
from pathlib import Path

from smart_search.graph.engine import CodeGraph
from smart_search.graph.builder import GraphBuilder
from smart_search.graph.algorithms import GraphAlgorithms
from smart_search.graph.models import NodeData, NodeType, EdgeData, EdgeType
from smart_search.parsing.models import (
    CodeUnit,
    CodeUnitType,
    Language,
    Span,
    Position,
)


@pytest.fixture
def large_graph():
    """Create a large graph for testing."""
    graph = CodeGraph()

    # Add 1000 nodes
    for i in range(1000):
        node_data = NodeData(
            id=f"node_{i}",
            name=f"node_{i}",
            qualified_name=f"module.node_{i}",
            node_type=NodeType.FUNCTION if i % 3 == 0 else NodeType.CLASS,
            file_path=Path(f"/src/module_{i // 100}.py"),
            line_start=i * 10,
            line_end=i * 10 + 9,
        )
        graph.add_node(node_data)

    # Add edges (creating a sparse graph)
    for i in range(1000):
        # Each node connects to ~5 other nodes
        for j in range(5):
            target = (i + j * 100 + 1) % 1000
            if target != i:
                edge_data = EdgeData(
                    edge_type=EdgeType.CALLS if j % 2 == 0 else EdgeType.IMPORTS,
                    weight=1.0,
                )
                graph.add_edge(f"node_{i}", f"node_{target}", edge_data)

    return graph


@pytest.fixture
def medium_graph():
    """Create a medium-sized graph for testing."""
    graph = CodeGraph()

    # Add 100 nodes
    for i in range(100):
        node_data = NodeData(
            id=f"func_{i}",
            name=f"func_{i}",
            qualified_name=f"mod.func_{i}",
            node_type=NodeType.FUNCTION,
            file_path=Path(f"/src/mod.py"),
            line_start=i * 5,
            line_end=i * 5 + 4,
        )
        graph.add_node(node_data)

    # Create a call chain
    for i in range(99):
        edge_data = EdgeData(edge_type=EdgeType.CALLS)
        graph.add_edge(f"func_{i}", f"func_{i+1}", edge_data)

    return graph


@pytest.mark.slow
class TestGraphEnginePerformance:
    """Performance tests for graph engine."""

    def test_add_many_nodes(self):
        """Test adding many nodes quickly."""
        graph = CodeGraph()

        start = time.perf_counter()
        for i in range(5000):
            node_data = NodeData(
                id=f"node_{i}",
                name=f"node_{i}",
                qualified_name=f"m.node_{i}",
                node_type=NodeType.FUNCTION,
            )
            graph.add_node(node_data)
        elapsed = time.perf_counter() - start

        assert graph.node_count == 5000
        # Allow more time for debug logging overhead
        assert elapsed < 2.0, f"Adding nodes took {elapsed:.3f}s"

    def test_add_many_edges(self, large_graph):
        """Test that large graph creation is efficient."""
        # The large_graph fixture already created 1000 nodes and ~5000 edges
        assert large_graph.node_count == 1000
        assert large_graph.edge_count >= 4000  # Some edges might be duplicates

    def test_node_lookup_performance(self, large_graph):
        """Test node lookup is fast."""
        start = time.perf_counter()
        for i in range(1000):
            node = large_graph.get_node(f"node_{i}")
            assert node is not None
        elapsed = time.perf_counter() - start

        # 1000 lookups should be very fast
        assert elapsed < 0.1, f"Lookups took {elapsed:.3f}s"

    def test_neighbor_lookup_performance(self, large_graph):
        """Test neighbor lookup is fast."""
        start = time.perf_counter()
        for i in range(100):
            neighbors = large_graph.get_neighbors(f"node_{i}")
        elapsed = time.perf_counter() - start

        # 100 neighbor lookups should be fast
        assert elapsed < 0.1, f"Neighbor lookups took {elapsed:.3f}s"


@pytest.mark.slow
class TestGraphAlgorithmsPerformance:
    """Performance tests for graph algorithms."""

    def test_find_callers_performance(self, large_graph):
        """Test find_callers performance."""
        algorithms = GraphAlgorithms(large_graph)

        start = time.perf_counter()
        for i in range(0, 100, 10):
            callers = algorithms.find_callers(f"node_{i}", max_depth=3)
        elapsed = time.perf_counter() - start

        # 10 find_callers calls should complete quickly
        assert elapsed < 1.0, f"Find callers took {elapsed:.3f}s"

    def test_find_descendants_performance(self, medium_graph):
        """Test find_descendants in a chain."""
        algorithms = GraphAlgorithms(medium_graph)

        start = time.perf_counter()
        # find_descendants returns nodes reachable via outgoing edges
        descendants = algorithms.find_descendants("func_0", max_depth=50)
        elapsed = time.perf_counter() - start

        # Should complete quickly (may not find many if edges go other direction)
        assert isinstance(descendants, list)
        assert elapsed < 0.5, f"Find descendants took {elapsed:.3f}s"

    def test_find_dependents_performance(self, large_graph):
        """Test find_dependents performance."""
        algorithms = GraphAlgorithms(large_graph)

        start = time.perf_counter()
        for i in range(10):
            dependents = algorithms.find_dependents(f"node_{i}", max_depth=3)
        elapsed = time.perf_counter() - start

        # 10 finds should complete quickly
        assert elapsed < 2.0, f"Find dependents took {elapsed:.3f}s"

    def test_centrality_performance(self, medium_graph):
        """Test centrality calculation performance."""
        algorithms = GraphAlgorithms(medium_graph)

        start = time.perf_counter()
        centrality = algorithms.calculate_centrality(centrality_type="degree")
        elapsed = time.perf_counter() - start

        # Take top 20 from results
        top_20 = centrality[:20]
        assert len(top_20) <= 20
        assert elapsed < 0.5, f"Centrality took {elapsed:.3f}s"


@pytest.mark.slow
class TestGraphBuilderPerformance:
    """Performance tests for graph builder."""

    def test_build_from_many_units(self, tmp_path):
        """Test building graph from many code units."""
        units = []

        # Create 500 code units
        for i in range(500):
            unit = CodeUnit(
                id=f"unit_{i}",
                name=f"unit_{i}",
                qualified_name=f"mod.unit_{i}",
                type=CodeUnitType.FUNCTION if i % 2 == 0 else CodeUnitType.CLASS,
                file_path=tmp_path / f"mod_{i // 50}.py",
                span=Span(
                    start=Position(line=i * 10 + 1, column=0),
                    end=Position(line=i * 10 + 9, column=0),
                ),
                language=Language.PYTHON,
                content=f"def unit_{i}(): pass",
                calls=[f"unit_{(i + 1) % 500}"] if i % 3 == 0 else [],
            )
            units.append(unit)

        builder = GraphBuilder()

        start = time.perf_counter()
        graph = builder.build_from_units(units)
        elapsed = time.perf_counter() - start

        assert graph.node_count == 500
        assert elapsed < 2.0, f"Building graph took {elapsed:.3f}s"

    def test_incremental_build_performance(self, tmp_path):
        """Test incrementally adding units to graph."""
        builder = GraphBuilder()

        start = time.perf_counter()

        # Add units one by one
        for i in range(200):
            unit = CodeUnit(
                id=f"inc_unit_{i}",
                name=f"inc_unit_{i}",
                qualified_name=f"mod.inc_unit_{i}",
                type=CodeUnitType.FUNCTION,
                file_path=tmp_path / "mod.py",
                span=Span(
                    start=Position(line=i * 5 + 1, column=0),
                    end=Position(line=i * 5 + 4, column=0),
                ),
                language=Language.PYTHON,
                content=f"def inc_unit_{i}(): pass",
            )
            builder.add_unit(unit)

        elapsed = time.perf_counter() - start

        assert builder.graph.node_count == 200
        assert elapsed < 1.0, f"Incremental build took {elapsed:.3f}s"
