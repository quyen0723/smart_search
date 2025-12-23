"""Code graph engine using RustworkX.

Provides the core graph data structure and operations for
code dependency analysis.
"""

from collections.abc import Iterator
from typing import Any

import rustworkx as rx

from smart_search.graph.models import (
    EdgeData,
    EdgeType,
    GraphEdge,
    GraphNode,
    GraphStats,
    NodeData,
    NodeType,
    PathResult,
    SubgraphResult,
    TraversalOptions,
)
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class CodeGraph:
    """A directed graph representing code relationships.

    Uses RustworkX for high-performance graph operations.
    Supports both directed and undirected queries.
    """

    def __init__(self) -> None:
        """Initialize an empty code graph."""
        self._graph: rx.PyDiGraph[NodeData, EdgeData] = rx.PyDiGraph()
        self._node_id_to_index: dict[str, int] = {}
        self._index_to_node_id: dict[int, str] = {}

    @property
    def node_count(self) -> int:
        """Get the number of nodes in the graph."""
        return len(self._graph)

    @property
    def edge_count(self) -> int:
        """Get the number of edges in the graph."""
        return self._graph.num_edges()

    def add_node(self, data: NodeData) -> int:
        """Add a node to the graph.

        Args:
            data: The node data.

        Returns:
            The index of the added node.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if data.id in self._node_id_to_index:
            raise ValueError(f"Node with ID '{data.id}' already exists")

        index = self._graph.add_node(data)
        self._node_id_to_index[data.id] = index
        self._index_to_node_id[index] = data.id

        logger.debug("Added node", node_id=data.id, index=index)
        return index

    def add_node_if_not_exists(self, data: NodeData) -> int:
        """Add a node if it doesn't exist, or return existing index.

        Args:
            data: The node data.

        Returns:
            The index of the node (new or existing).
        """
        if data.id in self._node_id_to_index:
            return self._node_id_to_index[data.id]
        return self.add_node(data)

    def get_node(self, node_id: str) -> GraphNode | None:
        """Get a node by its ID.

        Args:
            node_id: The node's unique identifier.

        Returns:
            The GraphNode, or None if not found.
        """
        index = self._node_id_to_index.get(node_id)
        if index is None:
            return None
        return GraphNode(index=index, data=self._graph[index])

    def get_node_by_index(self, index: int) -> GraphNode | None:
        """Get a node by its index.

        Args:
            index: The node's index in the graph.

        Returns:
            The GraphNode, or None if not found.
        """
        if index not in self._index_to_node_id:
            return None
        return GraphNode(index=index, data=self._graph[index])

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists.

        Args:
            node_id: The node's unique identifier.

        Returns:
            True if the node exists.
        """
        return node_id in self._node_id_to_index

    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the graph.

        Args:
            node_id: The node's unique identifier.

        Returns:
            True if the node was removed, False if not found.
        """
        index = self._node_id_to_index.get(node_id)
        if index is None:
            return False

        self._graph.remove_node(index)
        del self._node_id_to_index[node_id]
        del self._index_to_node_id[index]

        logger.debug("Removed node", node_id=node_id)
        return True

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        data: EdgeData,
    ) -> bool:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            data: Edge data.

        Returns:
            True if edge was added, False if nodes don't exist.
        """
        source_index = self._node_id_to_index.get(source_id)
        target_index = self._node_id_to_index.get(target_id)

        if source_index is None or target_index is None:
            logger.warning(
                "Cannot add edge - node not found",
                source_id=source_id,
                target_id=target_id,
                source_exists=source_index is not None,
                target_exists=target_index is not None,
            )
            return False

        self._graph.add_edge(source_index, target_index, data)
        logger.debug(
            "Added edge",
            source_id=source_id,
            target_id=target_id,
            edge_type=data.edge_type.value,
        )
        return True

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if an edge exists between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            True if the edge exists.
        """
        source_index = self._node_id_to_index.get(source_id)
        target_index = self._node_id_to_index.get(target_id)

        if source_index is None or target_index is None:
            return False

        return self._graph.has_edge(source_index, target_index)

    def get_edge(self, source_id: str, target_id: str) -> GraphEdge | None:
        """Get an edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            The GraphEdge, or None if not found.
        """
        source_index = self._node_id_to_index.get(source_id)
        target_index = self._node_id_to_index.get(target_id)

        if source_index is None or target_index is None:
            return None

        try:
            data = self._graph.get_edge_data(source_index, target_index)
            return GraphEdge(source=source_index, target=target_index, data=data)
        except rx.NoEdgeBetweenNodes:
            return None

    def get_edges_between(
        self,
        source_id: str,
        target_id: str,
    ) -> list[GraphEdge]:
        """Get all edges between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            List of edges between the nodes.
        """
        source_index = self._node_id_to_index.get(source_id)
        target_index = self._node_id_to_index.get(target_id)

        if source_index is None or target_index is None:
            return []

        edges = []
        try:
            data = self._graph.get_edge_data(source_index, target_index)
            edges.append(
                GraphEdge(source=source_index, target=target_index, data=data)
            )
        except rx.NoEdgeBetweenNodes:
            pass

        return edges

    def get_successors(self, node_id: str) -> list[GraphNode]:
        """Get all successor nodes (outgoing edges).

        Args:
            node_id: The node's unique identifier.

        Returns:
            List of successor nodes.
        """
        index = self._node_id_to_index.get(node_id)
        if index is None:
            return []

        successors = []
        for succ_index in self._graph.successor_indices(index):
            successors.append(
                GraphNode(index=succ_index, data=self._graph[succ_index])
            )
        return successors

    def get_predecessors(self, node_id: str) -> list[GraphNode]:
        """Get all predecessor nodes (incoming edges).

        Args:
            node_id: The node's unique identifier.

        Returns:
            List of predecessor nodes.
        """
        index = self._node_id_to_index.get(node_id)
        if index is None:
            return []

        predecessors = []
        for pred_index in self._graph.predecessor_indices(index):
            predecessors.append(
                GraphNode(index=pred_index, data=self._graph[pred_index])
            )
        return predecessors

    def get_neighbors(self, node_id: str) -> list[GraphNode]:
        """Get all neighbor nodes (both directions).

        Args:
            node_id: The node's unique identifier.

        Returns:
            List of neighbor nodes (deduplicated).
        """
        successors = {n.data.id: n for n in self.get_successors(node_id)}
        predecessors = {n.data.id: n for n in self.get_predecessors(node_id)}

        # Merge and deduplicate
        all_neighbors = {**predecessors, **successors}
        return list(all_neighbors.values())

    def get_outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        """Get all outgoing edges from a node.

        Args:
            node_id: The node's unique identifier.

        Returns:
            List of outgoing edges.
        """
        index = self._node_id_to_index.get(node_id)
        if index is None:
            return []

        edges = []
        for edge in self._graph.out_edges(index):
            edges.append(
                GraphEdge(
                    source=edge[0],
                    target=edge[1],
                    data=self._graph.get_edge_data(edge[0], edge[1]),
                )
            )
        return edges

    def get_incoming_edges(self, node_id: str) -> list[GraphEdge]:
        """Get all incoming edges to a node.

        Args:
            node_id: The node's unique identifier.

        Returns:
            List of incoming edges.
        """
        index = self._node_id_to_index.get(node_id)
        if index is None:
            return []

        edges = []
        for edge in self._graph.in_edges(index):
            edges.append(
                GraphEdge(
                    source=edge[0],
                    target=edge[1],
                    data=self._graph.get_edge_data(edge[0], edge[1]),
                )
            )
        return edges

    def get_all_nodes(self) -> list[GraphNode]:
        """Get all nodes in the graph.

        Returns:
            List of all nodes.
        """
        nodes = []
        for index in self._graph.node_indices():
            nodes.append(GraphNode(index=index, data=self._graph[index]))
        return nodes

    def get_all_edges(self) -> list[GraphEdge]:
        """Get all edges in the graph.

        Returns:
            List of all edges.
        """
        edges = []
        for edge in self._graph.edge_list():
            edges.append(
                GraphEdge(
                    source=edge[0],
                    target=edge[1],
                    data=self._graph.get_edge_data(edge[0], edge[1]),
                )
            )
        return edges

    def get_nodes_by_type(self, node_type: NodeType) -> list[GraphNode]:
        """Get all nodes of a specific type.

        Args:
            node_type: The type of nodes to retrieve.

        Returns:
            List of nodes matching the type.
        """
        nodes = []
        for index in self._graph.node_indices():
            node_data = self._graph[index]
            if node_data.node_type == node_type:
                nodes.append(GraphNode(index=index, data=node_data))
        return nodes

    def get_edges_by_type(self, edge_type: EdgeType) -> list[GraphEdge]:
        """Get all edges of a specific type.

        Args:
            edge_type: The type of edges to retrieve.

        Returns:
            List of edges matching the type.
        """
        edges = []
        for edge in self._graph.edge_list():
            edge_data = self._graph.get_edge_data(edge[0], edge[1])
            if edge_data.edge_type == edge_type:
                edges.append(
                    GraphEdge(source=edge[0], target=edge[1], data=edge_data)
                )
        return edges

    def traverse(
        self,
        start_id: str,
        options: TraversalOptions | None = None,
    ) -> list[GraphNode]:
        """Traverse the graph from a starting node.

        Args:
            start_id: Starting node ID.
            options: Traversal options.

        Returns:
            List of visited nodes.
        """
        if options is None:
            options = TraversalOptions()

        start_index = self._node_id_to_index.get(start_id)
        if start_index is None:
            return []

        visited: set[int] = set()
        result: list[GraphNode] = []

        def should_include_node(node_data: NodeData) -> bool:
            if options.node_types is None:
                return True
            return node_data.node_type in options.node_types

        def should_follow_edge(edge_data: EdgeData) -> bool:
            if options.edge_types is None:
                return True
            return edge_data.edge_type in options.edge_types

        def visit(index: int, depth: int) -> None:
            if index in visited:
                return
            if options.max_depth >= 0 and depth > options.max_depth:
                return

            visited.add(index)
            node_data = self._graph[index]

            if should_include_node(node_data):
                if index != start_index or options.include_self:
                    result.append(GraphNode(index=index, data=node_data))

            # Get next nodes based on direction
            next_indices: list[int] = []
            if options.direction in ("outgoing", "both"):
                for edge in self._graph.out_edges(index):
                    edge_data = self._graph.get_edge_data(edge[0], edge[1])
                    if should_follow_edge(edge_data):
                        next_indices.append(edge[1])

            if options.direction in ("incoming", "both"):
                for edge in self._graph.in_edges(index):
                    edge_data = self._graph.get_edge_data(edge[0], edge[1])
                    if should_follow_edge(edge_data):
                        next_indices.append(edge[0])

            for next_index in next_indices:
                visit(next_index, depth + 1)

        visit(start_index, 0)
        return result

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
    ) -> PathResult | None:
        """Find the shortest path between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            PathResult if path exists, None otherwise.
        """
        source_index = self._node_id_to_index.get(source_id)
        target_index = self._node_id_to_index.get(target_id)

        if source_index is None or target_index is None:
            return None

        try:
            # Use BFS for unweighted shortest path
            path_indices = rx.dijkstra_shortest_paths(
                self._graph,
                source_index,
                target_index,
                weight_fn=lambda _: 1.0,
            )

            if target_index not in path_indices:
                return None

            path = path_indices[target_index]
            nodes = [
                GraphNode(index=i, data=self._graph[i]) for i in path
            ]

            edges = []
            total_weight = 0.0
            for i in range(len(path) - 1):
                edge_data = self._graph.get_edge_data(path[i], path[i + 1])
                edges.append(
                    GraphEdge(source=path[i], target=path[i + 1], data=edge_data)
                )
                total_weight += edge_data.weight

            return PathResult(nodes=nodes, edges=edges, total_weight=total_weight)
        except Exception:
            return None

    def find_all_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 10,
    ) -> list[PathResult]:
        """Find all paths between two nodes up to a maximum depth.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            max_depth: Maximum path length.

        Returns:
            List of PathResults.
        """
        source_index = self._node_id_to_index.get(source_id)
        target_index = self._node_id_to_index.get(target_id)

        if source_index is None or target_index is None:
            return []

        paths: list[PathResult] = []

        def dfs(
            current: int,
            path: list[int],
            visited: set[int],
        ) -> None:
            if len(path) > max_depth:
                return

            if current == target_index:
                nodes = [
                    GraphNode(index=i, data=self._graph[i]) for i in path
                ]
                edges = []
                total_weight = 0.0
                for i in range(len(path) - 1):
                    edge_data = self._graph.get_edge_data(path[i], path[i + 1])
                    edges.append(
                        GraphEdge(source=path[i], target=path[i + 1], data=edge_data)
                    )
                    total_weight += edge_data.weight
                paths.append(
                    PathResult(nodes=nodes, edges=edges, total_weight=total_weight)
                )
                return

            for succ_index in self._graph.successor_indices(current):
                if succ_index not in visited:
                    visited.add(succ_index)
                    path.append(succ_index)
                    dfs(succ_index, path, visited)
                    path.pop()
                    visited.remove(succ_index)

        dfs(source_index, [source_index], {source_index})
        return paths

    def get_subgraph(
        self,
        node_id: str,
        max_depth: int = 2,
        options: TraversalOptions | None = None,
    ) -> SubgraphResult:
        """Get a subgraph centered on a node.

        Args:
            node_id: Center node ID.
            max_depth: Maximum depth to include.
            options: Traversal options for filtering.

        Returns:
            SubgraphResult containing the subgraph.
        """
        if options is None:
            options = TraversalOptions(max_depth=max_depth, direction="both")
        else:
            options.max_depth = max_depth

        center_node = self.get_node(node_id)
        if center_node is None:
            return SubgraphResult(nodes=[], edges=[])

        # Get all nodes in the subgraph
        options.include_self = True
        nodes = self.traverse(node_id, options)

        # Get node indices for edge filtering
        node_indices = {n.index for n in nodes}

        # Get edges between subgraph nodes
        edges = []
        for edge in self._graph.edge_list():
            if edge[0] in node_indices and edge[1] in node_indices:
                edge_data = self._graph.get_edge_data(edge[0], edge[1])
                # Apply edge type filter if specified
                if options.edge_types is None or edge_data.edge_type in options.edge_types:
                    edges.append(
                        GraphEdge(source=edge[0], target=edge[1], data=edge_data)
                    )

        return SubgraphResult(nodes=nodes, edges=edges, center_node=center_node)

    def get_stats(self) -> GraphStats:
        """Get statistics about the graph.

        Returns:
            GraphStats with various metrics.
        """
        nodes_by_type: dict[str, int] = {}
        for index in self._graph.node_indices():
            node_type = self._graph[index].node_type.value
            nodes_by_type[node_type] = nodes_by_type.get(node_type, 0) + 1

        edges_by_type: dict[str, int] = {}
        for edge in self._graph.edge_list():
            edge_type = self._graph.get_edge_data(edge[0], edge[1]).edge_type.value
            edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1

        # Calculate connected components (treating as undirected)
        undirected = self._graph.to_undirected()
        components = rx.connected_components(undirected)
        num_components = len(components)

        # Calculate density
        n = len(self._graph)
        e = self._graph.num_edges()
        density = e / (n * (n - 1)) if n > 1 else 0.0

        return GraphStats(
            node_count=n,
            edge_count=e,
            nodes_by_type=nodes_by_type,
            edges_by_type=edges_by_type,
            connected_components=num_components,
            density=density,
        )

    def clear(self) -> None:
        """Clear all nodes and edges from the graph."""
        self._graph = rx.PyDiGraph()
        self._node_id_to_index.clear()
        self._index_to_node_id.clear()
        logger.debug("Graph cleared")

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization.

        Returns:
            Dictionary representation of the graph.
        """
        nodes = []
        for index in self._graph.node_indices():
            nodes.append(self._graph[index].to_dict())

        edges = []
        for edge in self._graph.edge_list():
            edge_data = self._graph.get_edge_data(edge[0], edge[1])
            edges.append({
                "source": self._index_to_node_id[edge[0]],
                "target": self._index_to_node_id[edge[1]],
                "data": edge_data.to_dict(),
            })

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeGraph":
        """Create a graph from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            New CodeGraph instance.
        """
        graph = cls()

        for node_data in data.get("nodes", []):
            graph.add_node(NodeData.from_dict(node_data))

        for edge_data in data.get("edges", []):
            graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                EdgeData.from_dict(edge_data["data"]),
            )

        return graph
