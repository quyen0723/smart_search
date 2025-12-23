"""Graph models for code dependency analysis.

Defines node and edge types for the code dependency graph,
along with query results and traversal options.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeType(str, Enum):
    """Types of nodes in the code graph."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"


class EdgeType(str, Enum):
    """Types of edges (relationships) in the code graph."""

    # Call relationships
    CALLS = "calls"  # Function/method calls another
    CALLED_BY = "called_by"  # Inverse of CALLS

    # Import relationships
    IMPORTS = "imports"  # Module imports another
    IMPORTED_BY = "imported_by"  # Inverse of IMPORTS

    # Inheritance relationships
    INHERITS = "inherits"  # Class inherits from another
    INHERITED_BY = "inherited_by"  # Inverse of INHERITS

    # Containment relationships
    CONTAINS = "contains"  # Module/class contains function/method
    CONTAINED_BY = "contained_by"  # Inverse of CONTAINS

    # Usage relationships
    USES = "uses"  # Uses a variable/constant
    USED_BY = "used_by"  # Inverse of USES

    # Reference relationships
    REFERENCES = "references"  # Generic reference
    REFERENCED_BY = "referenced_by"  # Inverse of REFERENCES


@dataclass
class NodeData:
    """Data associated with a graph node.

    Attributes:
        id: Unique identifier for the node.
        name: Short name of the node.
        qualified_name: Fully qualified name.
        node_type: Type of node (function, class, etc.).
        file_path: Path to the source file.
        line_start: Starting line number.
        line_end: Ending line number.
        metadata: Additional metadata.
    """

    id: str
    name: str
    qualified_name: str
    node_type: NodeType
    file_path: Path | None = None
    line_start: int | None = None
    line_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "node_type": self.node_type.value,
            "file_path": str(self.file_path) if self.file_path else None,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeData":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            qualified_name=data["qualified_name"],
            node_type=NodeType(data["node_type"]),
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EdgeData:
    """Data associated with a graph edge.

    Attributes:
        edge_type: Type of relationship.
        weight: Edge weight (for weighted algorithms).
        metadata: Additional metadata.
    """

    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeData":
        """Create from dictionary."""
        return cls(
            edge_type=EdgeType(data["edge_type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GraphNode:
    """A node in the graph with its index.

    Attributes:
        index: The node's index in the graph.
        data: The node's data.
    """

    index: int
    data: NodeData


@dataclass
class GraphEdge:
    """An edge in the graph with source and target.

    Attributes:
        source: Source node index.
        target: Target node index.
        data: Edge data.
    """

    source: int
    target: int
    data: EdgeData


@dataclass
class TraversalOptions:
    """Options for graph traversal.

    Attributes:
        max_depth: Maximum depth to traverse (-1 for unlimited).
        edge_types: Types of edges to follow (None for all).
        node_types: Types of nodes to include (None for all).
        include_self: Include the starting node in results.
        direction: Traversal direction ('outgoing', 'incoming', 'both').
    """

    max_depth: int = -1
    edge_types: list[EdgeType] | None = None
    node_types: list[NodeType] | None = None
    include_self: bool = False
    direction: str = "outgoing"  # 'outgoing', 'incoming', 'both'


@dataclass
class PathResult:
    """Result of a path query.

    Attributes:
        nodes: List of nodes in the path.
        edges: List of edges in the path.
        total_weight: Sum of edge weights.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    total_weight: float = 0.0


@dataclass
class SubgraphResult:
    """Result of a subgraph query.

    Attributes:
        nodes: Nodes in the subgraph.
        edges: Edges in the subgraph.
        center_node: The central node (if applicable).
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_node: GraphNode | None = None


@dataclass
class GraphStats:
    """Statistics about the graph.

    Attributes:
        node_count: Total number of nodes.
        edge_count: Total number of edges.
        nodes_by_type: Count of nodes by type.
        edges_by_type: Count of edges by type.
        connected_components: Number of connected components.
        density: Graph density (edges / possible edges).
    """

    node_count: int
    edge_count: int
    nodes_by_type: dict[str, int] = field(default_factory=dict)
    edges_by_type: dict[str, int] = field(default_factory=dict)
    connected_components: int = 0
    density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "nodes_by_type": self.nodes_by_type,
            "edges_by_type": self.edges_by_type,
            "connected_components": self.connected_components,
            "density": self.density,
        }


@dataclass
class ImpactAnalysisResult:
    """Result of impact analysis.

    Attributes:
        source_node: The node being analyzed.
        affected_nodes: Nodes that would be affected by changes.
        impact_paths: Paths showing how impact propagates.
        impact_score: Overall impact score.
    """

    source_node: GraphNode
    affected_nodes: list[GraphNode]
    impact_paths: list[PathResult]
    impact_score: float = 0.0
