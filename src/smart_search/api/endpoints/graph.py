"""Graph API endpoints.

Provides endpoints for code graph operations.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/graph", tags=["graph"])


# Request/Response models
class NodeInfo(BaseModel):
    """Graph node information."""

    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: str
    line_start: int
    line_end: int
    parent_id: str | None = None
    docstring: str | None = None


class EdgeInfo(BaseModel):
    """Graph edge information."""

    source: str
    target: str
    edge_type: str
    weight: float = 1.0


class CommunityInfo(BaseModel):
    """Community information."""

    id: str
    name: str
    size: int
    members: list[str]
    key_members: list[str]
    description: str | None = None


class GraphStats(BaseModel):
    """Graph statistics."""

    total_nodes: int
    total_edges: int
    node_types: dict[str, int]
    edge_types: dict[str, int]
    files_count: int
    communities_count: int
    avg_degree: float


class PathInfo(BaseModel):
    """Path between nodes."""

    source: str
    target: str
    path: list[str]
    length: int
    edges: list[EdgeInfo]


class SubgraphResponse(BaseModel):
    """Subgraph response."""

    center: str
    nodes: list[NodeInfo]
    edges: list[EdgeInfo]
    total_nodes: int
    total_edges: int


# Dependency placeholder
_graph = None


def get_graph():
    """Get graph instance."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph service not initialized")
    return _graph


def set_graph(graph):
    """Set graph instance."""
    global _graph
    _graph = graph


# Helper to convert node
def node_to_info(node) -> NodeInfo:
    """Convert graph node to NodeInfo."""
    return NodeInfo(
        id=node.id,
        name=node.name,
        qualified_name=node.qualified_name or node.name,
        code_type=node.type.value if hasattr(node.type, "value") else str(node.type),
        file_path=str(node.file_path),
        line_start=node.line_start,
        line_end=node.line_end,
        parent_id=getattr(node, "parent_id", None),
        docstring=getattr(node, "docstring", None),
    )


# Endpoints
@router.get("/stats", response_model=GraphStats)
async def get_stats(
    graph=Depends(get_graph),
) -> GraphStats:
    """Get graph statistics.

    Returns overall statistics about the code graph.
    """
    try:
        stats = graph.get_stats()

        return GraphStats(
            total_nodes=stats.get("total_nodes", 0),
            total_edges=stats.get("total_edges", 0),
            node_types=stats.get("node_types", {}),
            edge_types=stats.get("edge_types", {}),
            files_count=stats.get("files_count", 0),
            communities_count=stats.get("communities_count", 0),
            avg_degree=stats.get("avg_degree", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats retrieval failed: {str(e)}")


@router.get("/node/{node_id}", response_model=NodeInfo)
async def get_node(
    node_id: str = Path(..., description="Node ID"),
    graph=Depends(get_graph),
) -> NodeInfo:
    """Get node by ID."""
    try:
        node = graph.get_node(node_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

        return node_to_info(node)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node retrieval failed: {str(e)}")


@router.get("/nodes", response_model=list[NodeInfo])
async def list_nodes(
    code_type: str | None = Query(default=None, description="Filter by code type"),
    file_path: str | None = Query(default=None, description="Filter by file path"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    graph=Depends(get_graph),
) -> list[NodeInfo]:
    """List nodes with optional filtering."""
    try:
        # Get all nodes (limited)
        if file_path:
            nodes = graph.get_nodes_in_file(file_path)
        else:
            nodes = graph.get_all_nodes()[:limit + offset]

        # Filter by type
        if code_type:
            nodes = [
                n for n in nodes
                if (n.type.value if hasattr(n.type, "value") else str(n.type)) == code_type
            ]

        # Apply pagination
        nodes = nodes[offset:offset + limit]

        return [node_to_info(n) for n in nodes]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node listing failed: {str(e)}")


@router.get("/edges", response_model=list[EdgeInfo])
async def list_edges(
    edge_type: str | None = Query(default=None, description="Filter by edge type"),
    source_id: str | None = Query(default=None, description="Filter by source node"),
    target_id: str | None = Query(default=None, description="Filter by target node"),
    limit: int = Query(default=100, ge=1, le=1000),
    graph=Depends(get_graph),
) -> list[EdgeInfo]:
    """List edges with optional filtering."""
    try:
        edges = []

        if source_id:
            # Get outgoing edges from source
            callees = graph.get_callees(source_id, depth=1)
            for target in callees[:limit]:
                edges.append(
                    EdgeInfo(
                        source=source_id,
                        target=target,
                        edge_type="CALLS",
                    )
                )
        elif target_id:
            # Get incoming edges to target
            callers = graph.get_callers(target_id, depth=1)
            for source in callers[:limit]:
                edges.append(
                    EdgeInfo(
                        source=source,
                        target=target_id,
                        edge_type="CALLS",
                    )
                )
        else:
            # Get sample edges
            all_nodes = graph.get_all_nodes()[:50]
            for node in all_nodes:
                callees = graph.get_callees(node.id, depth=1)[:5]
                for callee in callees:
                    edges.append(
                        EdgeInfo(
                            source=node.id,
                            target=callee,
                            edge_type="CALLS",
                        )
                    )
                    if len(edges) >= limit:
                        break
                if len(edges) >= limit:
                    break

        # Filter by edge type
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]

        return edges[:limit]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Edge listing failed: {str(e)}")


@router.get("/subgraph/{node_id}", response_model=SubgraphResponse)
async def get_subgraph(
    node_id: str = Path(..., description="Center node ID"),
    depth: int = Query(default=2, ge=1, le=5, description="Expansion depth"),
    direction: str = Query(default="both", description="Direction: in, out, both"),
    max_nodes: int = Query(default=50, ge=1, le=200),
    graph=Depends(get_graph),
) -> SubgraphResponse:
    """Get subgraph around a node.

    Returns nodes and edges within specified depth of center node.
    """
    try:
        center = graph.get_node(node_id)
        if not center:
            raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

        nodes_map = {node_id: center}
        edges = []

        # Expand outward
        to_visit = [(node_id, 0)]
        visited = {node_id}

        while to_visit and len(nodes_map) < max_nodes:
            current_id, current_depth = to_visit.pop(0)

            if current_depth >= depth:
                continue

            # Get neighbors
            neighbors = []
            if direction in ("out", "both"):
                callees = graph.get_callees(current_id, depth=1)
                for callee in callees:
                    neighbors.append((callee, "CALLS", True))  # outgoing

            if direction in ("in", "both"):
                callers = graph.get_callers(current_id, depth=1)
                for caller in callers:
                    neighbors.append((caller, "CALLS", False))  # incoming

            for neighbor_id, edge_type, is_outgoing in neighbors:
                if len(nodes_map) >= max_nodes:
                    break

                # Add edge
                if is_outgoing:
                    edges.append(
                        EdgeInfo(source=current_id, target=neighbor_id, edge_type=edge_type)
                    )
                else:
                    edges.append(
                        EdgeInfo(source=neighbor_id, target=current_id, edge_type=edge_type)
                    )

                # Add node
                if neighbor_id not in nodes_map:
                    neighbor = graph.get_node(neighbor_id)
                    if neighbor:
                        nodes_map[neighbor_id] = neighbor
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            to_visit.append((neighbor_id, current_depth + 1))

        return SubgraphResponse(
            center=node_id,
            nodes=[node_to_info(n) for n in nodes_map.values()],
            edges=edges,
            total_nodes=len(nodes_map),
            total_edges=len(edges),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subgraph retrieval failed: {str(e)}")


@router.get("/path", response_model=PathInfo | None)
async def find_path(
    source: str = Query(..., description="Source node ID"),
    target: str = Query(..., description="Target node ID"),
    max_depth: int = Query(default=5, ge=1, le=10),
    graph=Depends(get_graph),
) -> PathInfo | None:
    """Find shortest path between nodes."""
    try:
        # Verify nodes exist
        source_node = graph.get_node(source)
        target_node = graph.get_node(target)

        if not source_node:
            raise HTTPException(status_code=404, detail=f"Source not found: {source}")
        if not target_node:
            raise HTTPException(status_code=404, detail=f"Target not found: {target}")

        # BFS to find path
        from collections import deque

        queue = deque([(source, [source])])
        visited = {source}

        while queue:
            current, path = queue.popleft()

            if len(path) > max_depth:
                continue

            if current == target:
                # Build edges
                edges = []
                for i in range(len(path) - 1):
                    edges.append(
                        EdgeInfo(
                            source=path[i],
                            target=path[i + 1],
                            edge_type="CALLS",
                        )
                    )

                return PathInfo(
                    source=source,
                    target=target,
                    path=path,
                    length=len(path) - 1,
                    edges=edges,
                )

            # Expand
            neighbors = graph.get_callees(current, depth=1)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Path finding failed: {str(e)}")


@router.get("/communities", response_model=list[CommunityInfo])
async def list_communities(
    limit: int = Query(default=20, ge=1, le=100),
    graph=Depends(get_graph),
) -> list[CommunityInfo]:
    """List code communities/modules."""
    try:
        communities = []

        # Get community IDs
        community_ids = graph.get_all_communities()[:limit]

        for cid in community_ids:
            members = graph.get_community_members(cid)
            if not members:
                continue

            # Get key members (highest degree)
            key_members = members[:5]

            communities.append(
                CommunityInfo(
                    id=cid,
                    name=f"Community {cid}",
                    size=len(members),
                    members=members[:20],
                    key_members=key_members,
                )
            )

        return communities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Communities listing failed: {str(e)}")


@router.get("/community/{community_id}", response_model=CommunityInfo)
async def get_community(
    community_id: str = Path(..., description="Community ID"),
    graph=Depends(get_graph),
) -> CommunityInfo:
    """Get community details."""
    try:
        members = graph.get_community_members(community_id)
        if not members:
            raise HTTPException(status_code=404, detail=f"Community not found: {community_id}")

        # Get key members
        key_members = members[:5]

        return CommunityInfo(
            id=community_id,
            name=f"Community {community_id}",
            size=len(members),
            members=members[:100],
            key_members=key_members,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Community retrieval failed: {str(e)}")


@router.get("/files")
async def list_files(
    pattern: str | None = Query(default=None, description="Glob pattern filter"),
    limit: int = Query(default=100, ge=1, le=500),
    graph=Depends(get_graph),
) -> dict[str, Any]:
    """List indexed files."""
    try:
        files = graph.get_all_files()

        # Filter by pattern
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]

        files = files[:limit]

        return {
            "files": files,
            "total": len(files),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File listing failed: {str(e)}")


@router.get("/file/{file_path:path}/nodes", response_model=list[NodeInfo])
async def get_file_nodes(
    file_path: str = Path(..., description="File path"),
    graph=Depends(get_graph),
) -> list[NodeInfo]:
    """Get all nodes in a file."""
    try:
        nodes = graph.get_nodes_in_file(file_path)
        return [node_to_info(n) for n in nodes]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File nodes retrieval failed: {str(e)}")
