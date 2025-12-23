"""Graph module for code dependency analysis.

Provides graph-based code analysis using RustworkX.
"""

from smart_search.graph.algorithms import (
    CentralityResult,
    CommunityResult,
    CycleResult,
    DependencyMetrics,
    GraphAlgorithms,
)
from smart_search.graph.builder import GraphBuilder, build_graph_from_directory
from smart_search.graph.engine import CodeGraph
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
from smart_search.graph.persistence import (
    GraphCache,
    GraphPersistence,
    create_persistence,
)

__all__ = [
    # Engine
    "CodeGraph",
    # Models
    "NodeType",
    "EdgeType",
    "NodeData",
    "EdgeData",
    "GraphNode",
    "GraphEdge",
    "TraversalOptions",
    "PathResult",
    "SubgraphResult",
    "GraphStats",
    "ImpactAnalysisResult",
    # Builder
    "GraphBuilder",
    "build_graph_from_directory",
    # Algorithms
    "GraphAlgorithms",
    "CentralityResult",
    "CommunityResult",
    "CycleResult",
    "DependencyMetrics",
    # Persistence
    "GraphPersistence",
    "GraphCache",
    "create_persistence",
]
