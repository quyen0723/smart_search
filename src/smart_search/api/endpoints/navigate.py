"""Navigation API endpoints.

Provides endpoints for code navigation operations.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/navigate", tags=["navigate"])


# Request/Response models
class CodeLocation(BaseModel):
    """Code location model."""

    file_path: str
    line_start: int
    line_end: int
    column_start: int | None = None
    column_end: int | None = None


class CodeReference(BaseModel):
    """Code reference model."""

    id: str
    name: str
    qualified_name: str
    code_type: str
    location: CodeLocation
    content: str | None = None


class DefinitionRequest(BaseModel):
    """Go to definition request."""

    file_path: str = Field(..., description="Current file path")
    line: int = Field(..., ge=1, description="Line number")
    column: int = Field(..., ge=0, description="Column number")
    symbol: str | None = Field(default=None, description="Symbol name if known")


class DefinitionResponse(BaseModel):
    """Go to definition response."""

    found: bool
    definition: CodeReference | None = None
    message: str | None = None


class ReferencesRequest(BaseModel):
    """Find references request."""

    code_id: str = Field(..., description="Code unit ID")
    include_definition: bool = Field(default=False, description="Include definition in results")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum references")


class ReferencesResponse(BaseModel):
    """Find references response."""

    code_id: str
    total: int
    references: list[CodeReference]


class CallersResponse(BaseModel):
    """Callers response."""

    code_id: str
    name: str
    callers: list[CodeReference]
    total: int


class CalleesResponse(BaseModel):
    """Callees response."""

    code_id: str
    name: str
    callees: list[CodeReference]
    total: int


class HierarchyNode(BaseModel):
    """Hierarchy node."""

    id: str
    name: str
    qualified_name: str
    code_type: str
    children: list["HierarchyNode"] = Field(default_factory=list)
    parent_id: str | None = None


class HierarchyResponse(BaseModel):
    """Type hierarchy response."""

    root: HierarchyNode
    total_nodes: int


class OutlineItem(BaseModel):
    """Document outline item."""

    id: str
    name: str
    code_type: str
    line_start: int
    line_end: int
    children: list["OutlineItem"] = Field(default_factory=list)


class OutlineResponse(BaseModel):
    """Document outline response."""

    file_path: str
    items: list[OutlineItem]


# Dependency placeholder
_graph = None
_searcher = None


def get_graph():
    """Get graph instance."""
    if _graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph service not initialized",
        )
    return _graph


def get_searcher():
    """Get searcher instance."""
    if _searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized",
        )
    return _searcher


def set_graph(graph):
    """Set graph instance."""
    global _graph
    _graph = graph


def set_searcher(searcher):
    """Set searcher instance."""
    global _searcher
    _searcher = searcher


# Endpoints
@router.post("/definition", response_model=DefinitionResponse)
async def go_to_definition(
    request: DefinitionRequest,
    graph=Depends(get_graph),
) -> DefinitionResponse:
    """Go to definition.

    Find the definition of a symbol at a given position.
    """
    try:
        # Find node at position
        node = graph.find_node_at_position(
            file_path=request.file_path,
            line=request.line,
            column=request.column,
        )

        if not node:
            return DefinitionResponse(
                found=False,
                message="No symbol found at position",
            )

        # Get definition
        definition = graph.get_definition(node.id)

        if not definition:
            return DefinitionResponse(
                found=False,
                message=f"Definition not found for {node.name}",
            )

        return DefinitionResponse(
            found=True,
            definition=CodeReference(
                id=definition.id,
                name=definition.name,
                qualified_name=definition.qualified_name or definition.name,
                code_type=definition.type.value if hasattr(definition.type, "value") else str(definition.type),
                location=CodeLocation(
                    file_path=str(definition.file_path),
                    line_start=definition.line_start,
                    line_end=definition.line_end,
                ),
                content=getattr(definition, "content", None),
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Definition lookup failed: {str(e)}")


@router.get("/references/{code_id}", response_model=ReferencesResponse)
async def find_references(
    code_id: str = Path(..., description="Code unit ID"),
    include_definition: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    graph=Depends(get_graph),
) -> ReferencesResponse:
    """Find all references to a code unit."""
    try:
        references = graph.get_references(code_id, limit=limit)

        refs = []
        for ref in references:
            if not include_definition and ref.id == code_id:
                continue
            refs.append(
                CodeReference(
                    id=ref.id,
                    name=ref.name,
                    qualified_name=ref.qualified_name or ref.name,
                    code_type=ref.type.value if hasattr(ref.type, "value") else str(ref.type),
                    location=CodeLocation(
                        file_path=str(ref.file_path),
                        line_start=ref.line_start,
                        line_end=ref.line_end,
                    ),
                )
            )

        return ReferencesResponse(
            code_id=code_id,
            total=len(refs),
            references=refs,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reference lookup failed: {str(e)}")


@router.get("/callers/{code_id}", response_model=CallersResponse)
async def get_callers(
    code_id: str = Path(..., description="Code unit ID"),
    depth: int = Query(default=1, ge=1, le=5, description="Traversal depth"),
    limit: int = Query(default=50, ge=1, le=200),
    graph=Depends(get_graph),
) -> CallersResponse:
    """Get all callers of a function/method."""
    try:
        node = graph.get_node(code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {code_id}")

        caller_ids = graph.get_callers(code_id, depth=depth)[:limit]

        callers = []
        for cid in caller_ids:
            caller = graph.get_node(cid)
            if caller:
                callers.append(
                    CodeReference(
                        id=caller.id,
                        name=caller.name,
                        qualified_name=caller.qualified_name or caller.name,
                        code_type=caller.type.value if hasattr(caller.type, "value") else str(caller.type),
                        location=CodeLocation(
                            file_path=str(caller.file_path),
                            line_start=caller.line_start,
                            line_end=caller.line_end,
                        ),
                    )
                )

        return CallersResponse(
            code_id=code_id,
            name=node.name,
            callers=callers,
            total=len(callers),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Callers lookup failed: {str(e)}")


@router.get("/callees/{code_id}", response_model=CalleesResponse)
async def get_callees(
    code_id: str = Path(..., description="Code unit ID"),
    depth: int = Query(default=1, ge=1, le=5, description="Traversal depth"),
    limit: int = Query(default=50, ge=1, le=200),
    graph=Depends(get_graph),
) -> CalleesResponse:
    """Get all functions/methods called by a code unit."""
    try:
        node = graph.get_node(code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {code_id}")

        callee_ids = graph.get_callees(code_id, depth=depth)[:limit]

        callees = []
        for cid in callee_ids:
            callee = graph.get_node(cid)
            if callee:
                callees.append(
                    CodeReference(
                        id=callee.id,
                        name=callee.name,
                        qualified_name=callee.qualified_name or callee.name,
                        code_type=callee.type.value if hasattr(callee.type, "value") else str(callee.type),
                        location=CodeLocation(
                            file_path=str(callee.file_path),
                            line_start=callee.line_start,
                            line_end=callee.line_end,
                        ),
                    )
                )

        return CalleesResponse(
            code_id=code_id,
            name=node.name,
            callees=callees,
            total=len(callees),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Callees lookup failed: {str(e)}")


@router.get("/hierarchy/{code_id}", response_model=HierarchyResponse)
async def get_hierarchy(
    code_id: str = Path(..., description="Code unit ID"),
    direction: str = Query(default="both", description="Direction: ancestors, descendants, both"),
    max_depth: int = Query(default=3, ge=1, le=10),
    graph=Depends(get_graph),
) -> HierarchyResponse:
    """Get type hierarchy for a class."""
    try:
        node = graph.get_node(code_id)
        if not node:
            raise HTTPException(status_code=404, detail=f"Code unit not found: {code_id}")

        def build_node(nid: str, depth: int = 0) -> HierarchyNode | None:
            n = graph.get_node(nid)
            if not n:
                return None

            children = []
            if direction in ("descendants", "both") and depth < max_depth:
                child_ids = graph.get_children(nid)
                for cid in child_ids[:20]:  # Limit children
                    child = build_node(cid, depth + 1)
                    if child:
                        children.append(child)

            return HierarchyNode(
                id=n.id,
                name=n.name,
                qualified_name=n.qualified_name or n.name,
                code_type=n.type.value if hasattr(n.type, "value") else str(n.type),
                children=children,
                parent_id=getattr(n, "parent_id", None),
            )

        root = build_node(code_id)
        if not root:
            raise HTTPException(status_code=404, detail="Failed to build hierarchy")

        # Count total nodes
        def count_nodes(node: HierarchyNode) -> int:
            return 1 + sum(count_nodes(c) for c in node.children)

        return HierarchyResponse(
            root=root,
            total_nodes=count_nodes(root),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hierarchy lookup failed: {str(e)}")


@router.get("/outline", response_model=OutlineResponse)
async def get_outline(
    file_path: str = Query(..., description="File path"),
    graph=Depends(get_graph),
) -> OutlineResponse:
    """Get document outline for a file.

    Returns a hierarchical view of all symbols in a file.
    """
    try:
        # Get all nodes in file
        nodes = graph.get_nodes_in_file(file_path)

        if not nodes:
            return OutlineResponse(file_path=file_path, items=[])

        # Build outline tree
        def build_outline(node) -> OutlineItem:
            children = []
            child_nodes = [n for n in nodes if getattr(n, "parent_id", None) == node.id]
            for child in sorted(child_nodes, key=lambda x: x.line_start):
                children.append(build_outline(child))

            return OutlineItem(
                id=node.id,
                name=node.name,
                code_type=node.type.value if hasattr(node.type, "value") else str(node.type),
                line_start=node.line_start,
                line_end=node.line_end,
                children=children,
            )

        # Find root nodes (no parent or parent not in file)
        root_nodes = [
            n for n in nodes
            if not getattr(n, "parent_id", None)
            or not any(p.id == n.parent_id for p in nodes)
        ]

        items = [
            build_outline(n)
            for n in sorted(root_nodes, key=lambda x: x.line_start)
        ]

        return OutlineResponse(file_path=file_path, items=items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Outline generation failed: {str(e)}")


@router.get("/symbols")
async def search_symbols(
    query: str = Query(..., min_length=1, description="Symbol name query"),
    code_type: str | None = Query(default=None, description="Filter by type"),
    limit: int = Query(default=20, ge=1, le=100),
    searcher=Depends(get_searcher),
) -> dict[str, Any]:
    """Search for symbols by name.

    Quick symbol search for navigation purposes.
    """
    from smart_search.search.schemas import SearchQuery, SearchType

    try:
        search_query = SearchQuery(
            query=query,
            search_type=SearchType.KEYWORD,
            limit=limit,
        )

        result = await searcher.search(search_query)

        symbols = []
        for hit in result.hits:
            if code_type and hit.code_type != code_type:
                continue
            symbols.append({
                "id": hit.id,
                "name": hit.name,
                "qualified_name": hit.qualified_name,
                "code_type": hit.code_type,
                "file_path": hit.file_path,
                "line": hit.line_start,
            })

        return {"query": query, "symbols": symbols, "total": len(symbols)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbol search failed: {str(e)}")
