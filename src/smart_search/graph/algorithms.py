"""Graph algorithms for code analysis.

Provides high-level algorithms for analyzing code relationships,
finding patterns, and computing metrics.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import rustworkx as rx

from smart_search.graph.engine import CodeGraph
from smart_search.graph.models import (
    EdgeType,
    GraphNode,
    ImpactAnalysisResult,
    NodeType,
    PathResult,
    TraversalOptions,
)
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CentralityResult:
    """Result of centrality analysis.

    Attributes:
        node: The node being analyzed.
        score: The centrality score.
    """

    node: GraphNode
    score: float


@dataclass
class CommunityResult:
    """Result of community detection.

    Attributes:
        communities: List of node groups (communities).
        modularity: Modularity score of the partition.
    """

    communities: list[list[GraphNode]]
    modularity: float = 0.0


@dataclass
class CycleResult:
    """Result of cycle detection.

    Attributes:
        cycles: List of cycles (each cycle is a list of nodes).
        has_cycles: Whether any cycles were found.
    """

    cycles: list[list[GraphNode]]
    has_cycles: bool = False


@dataclass
class DependencyMetrics:
    """Dependency metrics for a node.

    Attributes:
        node: The node being analyzed.
        afferent_coupling: Number of incoming dependencies (Ca).
        efferent_coupling: Number of outgoing dependencies (Ce).
        instability: Ce / (Ca + Ce), 0 = stable, 1 = unstable.
        abstractness: Ratio of abstract methods/classes.
    """

    node: GraphNode
    afferent_coupling: int = 0
    efferent_coupling: int = 0
    instability: float = 0.0
    abstractness: float = 0.0


class GraphAlgorithms:
    """High-level graph algorithms for code analysis."""

    def __init__(self, graph: CodeGraph) -> None:
        """Initialize with a code graph.

        Args:
            graph: The CodeGraph to analyze.
        """
        self.graph = graph

    def find_callers(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[GraphNode]:
        """Find all nodes that call a given function/method.

        Args:
            node_id: The function/method to find callers for.
            max_depth: Maximum call chain depth (-1 for unlimited).

        Returns:
            List of caller nodes.
        """
        options = TraversalOptions(
            max_depth=max_depth,
            edge_types=[EdgeType.CALLED_BY],
            direction="outgoing",
        )
        return self.graph.traverse(node_id, options)

    def find_callees(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[GraphNode]:
        """Find all functions/methods called by a given node.

        Args:
            node_id: The function/method to find callees for.
            max_depth: Maximum call chain depth (-1 for unlimited).

        Returns:
            List of callee nodes.
        """
        options = TraversalOptions(
            max_depth=max_depth,
            edge_types=[EdgeType.CALLS],
            direction="outgoing",
        )
        return self.graph.traverse(node_id, options)

    def find_ancestors(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[GraphNode]:
        """Find all ancestor classes (inheritance chain).

        Args:
            node_id: The class to find ancestors for.
            max_depth: Maximum inheritance depth (-1 for unlimited).

        Returns:
            List of ancestor nodes.
        """
        options = TraversalOptions(
            max_depth=max_depth,
            edge_types=[EdgeType.INHERITS],
            direction="outgoing",
        )
        return self.graph.traverse(node_id, options)

    def find_descendants(
        self,
        node_id: str,
        max_depth: int = -1,
    ) -> list[GraphNode]:
        """Find all descendant classes (subclasses).

        Args:
            node_id: The class to find descendants for.
            max_depth: Maximum inheritance depth (-1 for unlimited).

        Returns:
            List of descendant nodes.
        """
        options = TraversalOptions(
            max_depth=max_depth,
            edge_types=[EdgeType.INHERITED_BY],
            direction="outgoing",
        )
        return self.graph.traverse(node_id, options)

    def find_dependencies(
        self,
        node_id: str,
        max_depth: int = 1,
    ) -> list[GraphNode]:
        """Find all dependencies of a node.

        Includes calls, imports, and inheritance.

        Args:
            node_id: The node to find dependencies for.
            max_depth: Maximum depth (-1 for unlimited).

        Returns:
            List of dependency nodes.
        """
        options = TraversalOptions(
            max_depth=max_depth,
            edge_types=[EdgeType.CALLS, EdgeType.IMPORTS, EdgeType.INHERITS],
            direction="outgoing",
        )
        return self.graph.traverse(node_id, options)

    def find_dependents(
        self,
        node_id: str,
        max_depth: int = 1,
    ) -> list[GraphNode]:
        """Find all nodes that depend on a given node.

        Args:
            node_id: The node to find dependents for.
            max_depth: Maximum depth (-1 for unlimited).

        Returns:
            List of dependent nodes.
        """
        options = TraversalOptions(
            max_depth=max_depth,
            edge_types=[EdgeType.CALLED_BY, EdgeType.IMPORTED_BY, EdgeType.INHERITED_BY],
            direction="outgoing",
        )
        return self.graph.traverse(node_id, options)

    def analyze_impact(
        self,
        node_id: str,
        max_depth: int = 3,
    ) -> ImpactAnalysisResult:
        """Analyze the impact of changes to a node.

        Args:
            node_id: The node to analyze impact for.
            max_depth: Maximum propagation depth.

        Returns:
            ImpactAnalysisResult with affected nodes.
        """
        source_node = self.graph.get_node(node_id)
        if source_node is None:
            return ImpactAnalysisResult(
                source_node=GraphNode(
                    index=-1,
                    data=None,  # type: ignore
                ),
                affected_nodes=[],
                impact_paths=[],
                impact_score=0.0,
            )

        # Find all dependents (nodes that would be affected)
        affected = self.find_dependents(node_id, max_depth)

        # Find paths showing how impact propagates
        impact_paths: list[PathResult] = []
        for affected_node in affected[:10]:  # Limit to first 10 for performance
            paths = self.graph.find_all_paths(
                node_id,
                affected_node.data.id,
                max_depth=max_depth,
            )
            impact_paths.extend(paths[:3])  # Limit paths per node

        # Calculate impact score based on number and depth of affected nodes
        impact_score = self._calculate_impact_score(affected, max_depth)

        return ImpactAnalysisResult(
            source_node=source_node,
            affected_nodes=affected,
            impact_paths=impact_paths,
            impact_score=impact_score,
        )

    def calculate_centrality(
        self,
        centrality_type: str = "betweenness",
    ) -> list[CentralityResult]:
        """Calculate centrality for all nodes.

        Args:
            centrality_type: Type of centrality ('betweenness', 'closeness', 'degree').

        Returns:
            List of CentralityResult sorted by score (highest first).
        """
        nodes = self.graph.get_all_nodes()
        if not nodes:
            return []

        # Build index mapping
        index_to_node = {n.index: n for n in nodes}

        # Get the internal graph
        internal_graph = self.graph._graph

        if centrality_type == "betweenness":
            scores = rx.betweenness_centrality(internal_graph)
        elif centrality_type == "closeness":
            # RustworkX doesn't have closeness directly, use degree as fallback
            scores = {}
            for node in nodes:
                in_degree = len(self.graph.get_predecessors(node.data.id))
                out_degree = len(self.graph.get_successors(node.data.id))
                scores[node.index] = float(in_degree + out_degree)
        elif centrality_type == "degree":
            scores = {}
            for node in nodes:
                in_degree = len(self.graph.get_predecessors(node.data.id))
                out_degree = len(self.graph.get_successors(node.data.id))
                scores[node.index] = float(in_degree + out_degree)
        else:
            raise ValueError(f"Unknown centrality type: {centrality_type}")

        results = []
        for index, score in scores.items():
            if index in index_to_node:
                results.append(
                    CentralityResult(node=index_to_node[index], score=score)
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def find_cycles(self) -> CycleResult:
        """Find cycles in the graph.

        Returns:
            CycleResult with all cycles found.
        """
        internal_graph = self.graph._graph

        # Use RustworkX's simple cycles detection
        try:
            # Get strongly connected components with more than one node
            sccs = rx.strongly_connected_components(internal_graph)
            cycles_found = [scc for scc in sccs if len(scc) > 1]

            cycles: list[list[GraphNode]] = []
            for scc in cycles_found:
                cycle_nodes = []
                for index in scc:
                    node = self.graph.get_node_by_index(index)
                    if node:
                        cycle_nodes.append(node)
                if cycle_nodes:
                    cycles.append(cycle_nodes)

            return CycleResult(cycles=cycles, has_cycles=len(cycles) > 0)
        except Exception as e:
            logger.warning("Cycle detection failed", error=str(e))
            return CycleResult(cycles=[], has_cycles=False)

    def find_strongly_connected_components(self) -> list[list[GraphNode]]:
        """Find strongly connected components.

        Returns:
            List of components (each is a list of nodes).
        """
        internal_graph = self.graph._graph
        sccs = rx.strongly_connected_components(internal_graph)

        components = []
        for scc in sccs:
            component_nodes = []
            for index in scc:
                node = self.graph.get_node_by_index(index)
                if node:
                    component_nodes.append(node)
            if component_nodes:
                components.append(component_nodes)

        # Sort by size (largest first)
        components.sort(key=len, reverse=True)
        return components

    def calculate_dependency_metrics(
        self,
        node_id: str,
    ) -> DependencyMetrics:
        """Calculate dependency metrics for a node.

        Args:
            node_id: The node to calculate metrics for.

        Returns:
            DependencyMetrics for the node.
        """
        node = self.graph.get_node(node_id)
        if node is None:
            return DependencyMetrics(
                node=GraphNode(index=-1, data=None),  # type: ignore
            )

        # Afferent coupling: incoming dependencies
        dependents = self.find_dependents(node_id, max_depth=1)
        ca = len(dependents)

        # Efferent coupling: outgoing dependencies
        dependencies = self.find_dependencies(node_id, max_depth=1)
        ce = len(dependencies)

        # Instability: Ce / (Ca + Ce)
        instability = ce / (ca + ce) if (ca + ce) > 0 else 0.0

        return DependencyMetrics(
            node=node,
            afferent_coupling=ca,
            efferent_coupling=ce,
            instability=instability,
        )

    def find_hub_nodes(
        self,
        min_connections: int = 5,
    ) -> list[GraphNode]:
        """Find hub nodes with many connections.

        Args:
            min_connections: Minimum connections to be considered a hub.

        Returns:
            List of hub nodes sorted by connection count.
        """
        nodes = self.graph.get_all_nodes()
        hubs = []

        for node in nodes:
            in_count = len(self.graph.get_predecessors(node.data.id))
            out_count = len(self.graph.get_successors(node.data.id))
            total = in_count + out_count

            if total >= min_connections:
                hubs.append((node, total))

        # Sort by connection count
        hubs.sort(key=lambda x: x[1], reverse=True)
        return [h[0] for h in hubs]

    def find_isolated_nodes(self) -> list[GraphNode]:
        """Find nodes with no connections.

        Returns:
            List of isolated nodes.
        """
        nodes = self.graph.get_all_nodes()
        isolated = []

        for node in nodes:
            in_count = len(self.graph.get_predecessors(node.data.id))
            out_count = len(self.graph.get_successors(node.data.id))

            if in_count == 0 and out_count == 0:
                isolated.append(node)

        return isolated

    def find_dead_code(self) -> list[GraphNode]:
        """Find potentially dead code (unused functions/methods).

        Returns:
            List of nodes that are never called.
        """
        dead = []

        # Get all functions and methods
        functions = self.graph.get_nodes_by_type(NodeType.FUNCTION)
        methods = self.graph.get_nodes_by_type(NodeType.METHOD)

        for node in functions + methods:
            # Check if it has any callers
            callers = self.find_callers(node.data.id, max_depth=1)
            if not callers:
                # Check if it's a special method (like __init__)
                if not node.data.name.startswith("__"):
                    dead.append(node)

        return dead

    def get_call_hierarchy(
        self,
        node_id: str,
        direction: str = "both",
        max_depth: int = 3,
    ) -> dict:
        """Get the call hierarchy for a function/method.

        Args:
            node_id: The function/method node ID.
            direction: 'callers', 'callees', or 'both'.
            max_depth: Maximum depth to traverse.

        Returns:
            Dictionary with call hierarchy.
        """
        result = {"node_id": node_id, "callers": [], "callees": []}

        if direction in ("callers", "both"):
            result["callers"] = [
                {"id": n.data.id, "name": n.data.name}
                for n in self.find_callers(node_id, max_depth)
            ]

        if direction in ("callees", "both"):
            result["callees"] = [
                {"id": n.data.id, "name": n.data.name}
                for n in self.find_callees(node_id, max_depth)
            ]

        return result

    def get_inheritance_hierarchy(
        self,
        node_id: str,
    ) -> dict:
        """Get the inheritance hierarchy for a class.

        Args:
            node_id: The class node ID.

        Returns:
            Dictionary with inheritance hierarchy.
        """
        return {
            "node_id": node_id,
            "ancestors": [
                {"id": n.data.id, "name": n.data.name}
                for n in self.find_ancestors(node_id)
            ],
            "descendants": [
                {"id": n.data.id, "name": n.data.name}
                for n in self.find_descendants(node_id)
            ],
        }

    def _calculate_impact_score(
        self,
        affected_nodes: list[GraphNode],
        max_depth: int,
    ) -> float:
        """Calculate an impact score based on affected nodes.

        Args:
            affected_nodes: List of affected nodes.
            max_depth: Maximum depth used in analysis.

        Returns:
            Impact score between 0 and 1.
        """
        if not affected_nodes:
            return 0.0

        # Simple scoring: ratio of affected nodes to total nodes
        total_nodes = self.graph.node_count
        if total_nodes == 0:
            return 0.0

        # Weight by node type (classes have more impact than functions)
        weighted_count = 0.0
        for node in affected_nodes:
            if node.data.node_type == NodeType.CLASS:
                weighted_count += 2.0
            elif node.data.node_type == NodeType.MODULE:
                weighted_count += 3.0
            else:
                weighted_count += 1.0

        # Normalize to 0-1 range
        score = min(1.0, weighted_count / (total_nodes * 0.5))
        return round(score, 3)
