"""Analysis API endpoints.

Provides endpoints for code analysis operations.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/analyze", tags=["analyze"])


# Request/Response models
class ExplainRequest(BaseModel):
    """Code explanation request."""

    code_id: str = Field(..., description="Code unit ID to explain")
    question: str | None = Field(default=None, description="Specific question about the code")
    include_context: bool = Field(default=True, description="Include related code context")


class ExplainResponse(BaseModel):
    """Code explanation response."""

    code_id: str
    name: str
    explanation: str
    related_code: list[dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float


class ImpactRequest(BaseModel):
    """Impact analysis request."""

    code_id: str = Field(..., description="Code unit ID to analyze")
    max_depth: int = Field(default=3, ge=1, le=10, description="Maximum traversal depth")
    include_indirect: bool = Field(default=True, description="Include indirect dependencies")


class ImpactResult(BaseModel):
    """Impact analysis result for a single file/component."""

    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: str
    distance: int
    impact_type: str  # "direct" or "indirect"


class ImpactResponse(BaseModel):
    """Impact analysis response."""

    code_id: str
    name: str
    total_affected: int
    direct_count: int
    indirect_count: int
    affected: list[ImpactResult]
    files_affected: list[str]
    explanation: str | None = None


class ComplexityMetrics(BaseModel):
    """Code complexity metrics."""

    cyclomatic: int | None = None
    cognitive: int | None = None
    lines_of_code: int
    comment_lines: int | None = None
    blank_lines: int | None = None


class CodeMetrics(BaseModel):
    """Code metrics for a unit."""

    id: str
    name: str
    code_type: str
    file_path: str
    complexity: ComplexityMetrics
    dependencies_in: int
    dependencies_out: int
    maintainability_index: float | None = None


class MetricsResponse(BaseModel):
    """Metrics analysis response."""

    code_id: str
    metrics: CodeMetrics
    suggestions: list[str] = Field(default_factory=list)


class DependencyNode(BaseModel):
    """Dependency graph node."""

    id: str
    name: str
    code_type: str
    file_path: str


class DependencyEdge(BaseModel):
    """Dependency graph edge."""

    source: str
    target: str
    edge_type: str


class DependencyGraph(BaseModel):
    """Dependency graph."""

    nodes: list[DependencyNode]
    edges: list[DependencyEdge]


class DependencyResponse(BaseModel):
    """Dependency analysis response."""

    code_id: str
    graph: DependencyGraph
    circular_dependencies: list[list[str]] = Field(default_factory=list)


class DuplicateMatch(BaseModel):
    """Duplicate code match."""

    id: str
    name: str
    file_path: str
    line_start: int
    line_end: int
    similarity: float


class DuplicateResponse(BaseModel):
    """Duplicate detection response."""

    code_id: str
    duplicates: list[DuplicateMatch]
    total: int


# Dependency placeholders
_graph = None
_graphrag = None


def get_graph():
    """Get graph instance."""
    if _graph is None:
        raise HTTPException(status_code=503, detail="Graph service not initialized")
    return _graph


def get_graphrag():
    """Get GraphRAG instance."""
    if _graphrag is None:
        raise HTTPException(status_code=503, detail="GraphRAG service not initialized")
    return _graphrag


def set_graph(graph):
    """Set graph instance."""
    global _graph
    _graph = graph


def set_graphrag(graphrag):
    """Set GraphRAG instance."""
    global _graphrag
    _graphrag = graphrag


# Endpoints
@router.post("/explain", response_model=ExplainResponse)
async def explain_code(
    request: ExplainRequest,
    graphrag=Depends(get_graphrag),
    graph=Depends(get_graph),
) -> ExplainResponse:
    """Explain code using AI.

    Provides a natural language explanation of what the code does.
    """
    import time

    start = time.time()

    try:
        node = graph.get_node(request.code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {request.code_id}")

        # Get AI explanation
        result = await graphrag.explain_code(
            code_id=request.code_id,
            question=request.question,
        )

        # Get related code if requested
        related_code = []
        if request.include_context:
            related_ids = graph.get_callees(request.code_id, depth=1)[:5]
            for rid in related_ids:
                related = graph.get_node(rid)
                if related:
                    related_code.append({
                        "id": related.id,
                        "name": related.name,
                        "code_type": related.type.value if hasattr(related.type, "value") else str(related.type),
                        "relationship": "calls",
                    })

        processing_time = (time.time() - start) * 1000

        return ExplainResponse(
            code_id=request.code_id,
            name=node.name,
            explanation=result.response,
            related_code=related_code,
            processing_time_ms=processing_time,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@router.post("/impact", response_model=ImpactResponse)
async def analyze_impact(
    request: ImpactRequest,
    graph=Depends(get_graph),
) -> ImpactResponse:
    """Analyze impact of changing code.

    Shows what other code would be affected by changes to this unit.
    """
    try:
        node = graph.get_node(request.code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {request.code_id}")

        # Get all dependents (code that calls this code)
        direct_ids = set(graph.get_callers(request.code_id, depth=1))

        indirect_ids = set()
        if request.include_indirect and request.max_depth > 1:
            for depth in range(2, request.max_depth + 1):
                all_callers = graph.get_callers(request.code_id, depth=depth)
                indirect_ids.update(set(all_callers) - direct_ids)

        affected = []
        files_affected = set()

        # Process direct dependencies
        for did in direct_ids:
            dep_node = graph.get_node(did)
            if dep_node:
                affected.append(
                    ImpactResult(
                        id=dep_node.id,
                        name=dep_node.name,
                        qualified_name=dep_node.qualified_name or dep_node.name,
                        code_type=dep_node.type.value if hasattr(dep_node.type, "value") else str(dep_node.type),
                        file_path=str(dep_node.file_path),
                        distance=1,
                        impact_type="direct",
                    )
                )
                files_affected.add(str(dep_node.file_path))

        # Process indirect dependencies
        for iid in indirect_ids:
            ind_node = graph.get_node(iid)
            if ind_node:
                # Calculate actual distance
                distance = 2  # Simplified - could calculate exact path length
                affected.append(
                    ImpactResult(
                        id=ind_node.id,
                        name=ind_node.name,
                        qualified_name=ind_node.qualified_name or ind_node.name,
                        code_type=ind_node.type.value if hasattr(ind_node.type, "value") else str(ind_node.type),
                        file_path=str(ind_node.file_path),
                        distance=distance,
                        impact_type="indirect",
                    )
                )
                files_affected.add(str(ind_node.file_path))

        # Sort by distance
        affected.sort(key=lambda x: (x.distance, x.name))

        return ImpactResponse(
            code_id=request.code_id,
            name=node.name,
            total_affected=len(affected),
            direct_count=len(direct_ids),
            indirect_count=len(indirect_ids),
            affected=affected[:100],  # Limit results
            files_affected=sorted(files_affected),
            explanation=f"Changes to {node.name} may affect {len(affected)} code units across {len(files_affected)} files.",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impact analysis failed: {str(e)}")


@router.get("/metrics/{code_id}", response_model=MetricsResponse)
async def get_metrics(
    code_id: str = Path(..., description="Code unit ID"),
    graph=Depends(get_graph),
) -> MetricsResponse:
    """Get code metrics for a unit.

    Returns complexity metrics and suggestions for improvement.
    """
    try:
        node = graph.get_node(code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {code_id}")

        # Calculate metrics
        content = getattr(node, "content", "")
        lines = content.split("\n") if content else []

        code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")]
        comment_lines = [l for l in lines if l.strip().startswith("#")]
        blank_lines = [l for l in lines if not l.strip()]

        # Get dependency counts
        deps_in = len(graph.get_callers(code_id, depth=1))
        deps_out = len(graph.get_callees(code_id, depth=1))

        complexity = ComplexityMetrics(
            lines_of_code=len(code_lines),
            comment_lines=len(comment_lines),
            blank_lines=len(blank_lines),
        )

        metrics = CodeMetrics(
            id=code_id,
            name=node.name,
            code_type=node.type.value if hasattr(node.type, "value") else str(node.type),
            file_path=str(node.file_path),
            complexity=complexity,
            dependencies_in=deps_in,
            dependencies_out=deps_out,
        )

        # Generate suggestions
        suggestions = []
        if len(code_lines) > 50:
            suggestions.append("Consider breaking this into smaller functions")
        if deps_out > 10:
            suggestions.append("High number of dependencies - consider simplifying")
        if deps_in > 20:
            suggestions.append("Many dependents - changes here have wide impact")

        return MetricsResponse(
            code_id=code_id,
            metrics=metrics,
            suggestions=suggestions,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics calculation failed: {str(e)}")


@router.get("/dependencies/{code_id}", response_model=DependencyResponse)
async def get_dependencies(
    code_id: str = Path(..., description="Code unit ID"),
    direction: str = Query(default="both", description="Direction: in, out, both"),
    depth: int = Query(default=2, ge=1, le=5),
    graph=Depends(get_graph),
) -> DependencyResponse:
    """Get dependency graph for code unit."""
    try:
        node = graph.get_node(code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {code_id}")

        nodes_map = {code_id: node}
        edges = []

        # Get incoming dependencies (callers)
        if direction in ("in", "both"):
            caller_ids = graph.get_callers(code_id, depth=depth)
            for cid in caller_ids[:50]:
                caller = graph.get_node(cid)
                if caller:
                    nodes_map[cid] = caller
                    edges.append(
                        DependencyEdge(
                            source=cid,
                            target=code_id,
                            edge_type="calls",
                        )
                    )

        # Get outgoing dependencies (callees)
        if direction in ("out", "both"):
            callee_ids = graph.get_callees(code_id, depth=depth)
            for cid in callee_ids[:50]:
                callee = graph.get_node(cid)
                if callee:
                    nodes_map[cid] = callee
                    edges.append(
                        DependencyEdge(
                            source=code_id,
                            target=cid,
                            edge_type="calls",
                        )
                    )

        # Build response
        dep_nodes = [
            DependencyNode(
                id=n.id,
                name=n.name,
                code_type=n.type.value if hasattr(n.type, "value") else str(n.type),
                file_path=str(n.file_path),
            )
            for n in nodes_map.values()
        ]

        # Detect circular dependencies (simplified)
        circular = []
        for edge in edges:
            # Check if reverse edge exists
            if any(e.source == edge.target and e.target == edge.source for e in edges):
                pair = sorted([edge.source, edge.target])
                if pair not in circular:
                    circular.append(pair)

        return DependencyResponse(
            code_id=code_id,
            graph=DependencyGraph(nodes=dep_nodes, edges=edges),
            circular_dependencies=circular,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dependency analysis failed: {str(e)}")


@router.get("/duplicates/{code_id}", response_model=DuplicateResponse)
async def find_duplicates(
    code_id: str = Path(..., description="Code unit ID"),
    threshold: float = Query(default=0.8, ge=0.5, le=1.0, description="Similarity threshold"),
    limit: int = Query(default=10, ge=1, le=50),
    graphrag=Depends(get_graphrag),
    graph=Depends(get_graph),
) -> DuplicateResponse:
    """Find duplicate or similar code."""
    try:
        node = graph.get_node(code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {code_id}")

        content = getattr(node, "content", "")
        if not content:
            return DuplicateResponse(code_id=code_id, duplicates=[], total=0)

        # Find similar code
        similar = await graphrag.find_similar(
            code_content=content,
            limit=limit + 1,  # +1 to account for self
        )

        duplicates = []
        for ctx in similar:
            if ctx.code_id == code_id:
                continue
            if ctx.relevance_score >= threshold:
                duplicates.append(
                    DuplicateMatch(
                        id=ctx.code_id,
                        name=ctx.name,
                        file_path=str(ctx.file_path),
                        line_start=ctx.line_start,
                        line_end=ctx.line_end,
                        similarity=ctx.relevance_score,
                    )
                )

        # Sort by similarity
        duplicates.sort(key=lambda x: x.similarity, reverse=True)

        return DuplicateResponse(
            code_id=code_id,
            duplicates=duplicates[:limit],
            total=len(duplicates),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Duplicate detection failed: {str(e)}")


@router.post("/summarize")
async def summarize_file(
    file_path: str = Query(..., description="File path to summarize"),
    graphrag=Depends(get_graphrag),
    graph=Depends(get_graph),
) -> dict[str, Any]:
    """Generate a summary of a file.

    Provides an AI-generated overview of file contents.
    """
    try:
        # Get all nodes in file
        nodes = graph.get_nodes_in_file(file_path)
        if not nodes:
            return {
                "file_path": file_path,
                "summary": "No code units found in file.",
                "components": [],
            }

        # Build contexts for summarization
        contexts = []
        for node in nodes[:10]:  # Limit for context size
            contexts.append({
                "name": node.name,
                "file_path": str(node.file_path),
                "content": getattr(node, "content", "")[:500],
            })

        # Get summary from generator
        from smart_search.rag.generator import ResponseGenerator

        base_generator = (
            graphrag.generator.generator
            if hasattr(graphrag.generator, "generator")
            else graphrag.generator
        )

        summary = await base_generator.generate_summary(contexts)

        components = [
            {
                "name": node.name,
                "type": node.type.value if hasattr(node.type, "value") else str(node.type),
                "line_start": node.line_start,
            }
            for node in nodes
        ]

        return {
            "file_path": file_path,
            "summary": summary,
            "components": components,
            "total_components": len(nodes),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")
