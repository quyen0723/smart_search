"""Search API endpoints.

Provides endpoints for code search operations.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from smart_search.rag import GraphRAG, QueryType, SearchMode
from smart_search.search.schemas import SearchQuery, SearchResult, SearchType

router = APIRouter(prefix="/search", tags=["search"])


# Request/Response models
class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    search_type: SearchType = Field(
        default=SearchType.HYBRID,
        description="Type of search (keyword, semantic, hybrid)",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Result offset")
    language: str | None = Field(default=None, description="Filter by language")
    file_path: str | None = Field(default=None, description="Filter by file path prefix")
    code_type: str | None = Field(default=None, description="Filter by code type")

    model_config = {"json_schema_extra": {"example": {"query": "authentication handler", "limit": 10}}}


class SearchHitResponse(BaseModel):
    """Search hit response."""

    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    language: str
    score: float
    highlights: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    total_hits: int
    hits: list[SearchHitResponse]
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "authentication",
                "total_hits": 5,
                "hits": [],
                "processing_time_ms": 45.2,
            }
        }
    }


class RAGSearchRequest(BaseModel):
    """RAG-enhanced search request."""

    query: str = Field(..., min_length=1, max_length=2000, description="Natural language query")
    mode: SearchMode = Field(
        default=SearchMode.HYBRID,
        description="Search mode (local, global, hybrid, drift)",
    )
    language: str | None = Field(default=None, description="Filter by language")
    file_path: str | None = Field(default=None, description="Filter by file path prefix")
    include_explanation: bool = Field(default=True, description="Include AI explanation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How does the authentication flow work?",
                "mode": "hybrid",
            }
        }
    }


class RAGSearchResponse(BaseModel):
    """RAG-enhanced search response."""

    query: str
    mode: str
    answer: str
    contexts: list[dict[str, Any]]
    citations: list[str]
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "How does authentication work?",
                "mode": "hybrid",
                "answer": "Authentication is handled by...",
                "contexts": [],
                "citations": ["auth.login_user"],
                "processing_time_ms": 1250.5,
            }
        }
    }


class SimilarCodeRequest(BaseModel):
    """Request to find similar code."""

    code: str = Field(..., min_length=1, max_length=10000, description="Code snippet")
    limit: int = Field(default=5, ge=1, le=20, description="Maximum results")
    exclude_ids: list[str] = Field(default_factory=list, description="IDs to exclude")


class SimilarCodeResponse(BaseModel):
    """Similar code response."""

    results: list[SearchHitResponse]
    processing_time_ms: float


# Dependency injection placeholder
# In production, this would be configured via app state
_searcher = None
_graphrag = None


def get_searcher():
    """Get search instance."""
    if _searcher is None:
        raise HTTPException(
            status_code=503,
            detail="Search service not initialized",
        )
    return _searcher


def get_graphrag():
    """Get GraphRAG instance."""
    if _graphrag is None:
        raise HTTPException(
            status_code=503,
            detail="GraphRAG service not initialized",
        )
    return _graphrag


def set_searcher(searcher):
    """Set search instance (for testing/initialization)."""
    global _searcher
    _searcher = searcher


def set_graphrag(graphrag):
    """Set GraphRAG instance (for testing/initialization)."""
    global _graphrag
    _graphrag = graphrag


# Endpoints
@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    searcher=Depends(get_searcher),
) -> SearchResponse:
    """Search for code.

    Performs keyword, semantic, or hybrid search across indexed code.
    """
    import time

    start = time.time()

    # Build filters
    filters = {}
    if request.language:
        filters["language"] = request.language
    if request.file_path:
        filters["file_path"] = request.file_path
    if request.code_type:
        filters["code_type"] = request.code_type

    # Create search query
    search_query = SearchQuery(
        query=request.query,
        search_type=request.search_type,
        limit=request.limit,
        offset=request.offset,
    )

    # Execute search
    try:
        result = await searcher.search(search_query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # Convert to response
    hits = []
    for hit in result.hits:
        # Convert highlights dict to list
        highlight_list = []
        if hit.highlights:
            for field, snippets in hit.highlights.items():
                highlight_list.extend(snippets if isinstance(snippets, list) else [snippets])
        hits.append(
            SearchHitResponse(
                id=hit.id,
                name=hit.name,
                qualified_name=hit.qualified_name,
                code_type=hit.code_type,
                file_path=str(hit.file_path),
                line_start=hit.line_start,
                line_end=hit.line_end,
                content=hit.content,
                language=hit.language,
                score=hit.score,
                highlights=highlight_list,
            )
        )

    processing_time = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        total_hits=result.total,
        hits=hits,
        processing_time_ms=processing_time,
    )


@router.get("", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    search_type: SearchType = Query(default=SearchType.HYBRID),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    language: str | None = Query(default=None),
    searcher=Depends(get_searcher),
) -> SearchResponse:
    """Search for code (GET).

    Alternative GET endpoint for simple searches.
    """
    request = SearchRequest(
        query=q,
        search_type=search_type,
        limit=limit,
        offset=offset,
        language=language,
    )
    return await search(request, searcher)


@router.post("/rag", response_model=RAGSearchResponse)
async def rag_search(
    request: RAGSearchRequest,
    graphrag=Depends(get_graphrag),
) -> RAGSearchResponse:
    """RAG-enhanced search.

    Uses GraphRAG to provide intelligent code understanding with
    natural language answers.
    """
    # Build filters
    filters = {}
    if request.language:
        filters["language"] = request.language
    if request.file_path:
        filters["file_path"] = request.file_path

    # Execute GraphRAG query
    try:
        result = await graphrag.query(
            query=request.query,
            mode=request.mode,
            filters=filters if filters else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")

    # Convert contexts to dict
    contexts = [ctx.to_dict() for ctx in result.contexts]

    return RAGSearchResponse(
        query=result.query,
        mode=result.mode.value,
        answer=result.response if request.include_explanation else "",
        contexts=contexts,
        citations=result.generation.citations,
        processing_time_ms=result.total_time_ms,
    )


@router.post("/similar", response_model=SimilarCodeResponse)
async def find_similar(
    request: SimilarCodeRequest,
    graphrag=Depends(get_graphrag),
) -> SimilarCodeResponse:
    """Find similar code.

    Given a code snippet, finds similar code in the indexed codebase.
    """
    import time

    start = time.time()

    try:
        results = await graphrag.find_similar(
            code_content=request.code,
            limit=request.limit,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}")

    hits = [
        SearchHitResponse(
            id=ctx.code_id,
            name=ctx.name,
            qualified_name=ctx.qualified_name,
            code_type=ctx.code_type,
            file_path=str(ctx.file_path),
            line_start=ctx.line_start,
            line_end=ctx.line_end,
            content=ctx.content,
            language=ctx.language,
            score=ctx.relevance_score,
            highlights=[],
        )
        for ctx in results
        if ctx.code_id not in request.exclude_ids
    ]

    processing_time = (time.time() - start) * 1000

    return SimilarCodeResponse(
        results=hits,
        processing_time_ms=processing_time,
    )


@router.get("/suggest")
async def suggest(
    q: str = Query(..., min_length=1, max_length=100, description="Partial query"),
    limit: int = Query(default=10, ge=1, le=20),
    searcher=Depends(get_searcher),
) -> dict[str, Any]:
    """Get search suggestions.

    Provides autocomplete suggestions based on indexed code.
    """
    try:
        # Use prefix search for suggestions
        search_query = SearchQuery(
            query=q,
            search_type=SearchType.KEYWORD,
            limit=limit,
        )
        result = await searcher.search(search_query)

        suggestions = [
            {
                "text": hit.name,
                "qualified_name": hit.qualified_name,
                "type": hit.code_type,
            }
            for hit in result.hits
        ]

        return {"query": q, "suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggest failed: {str(e)}")
