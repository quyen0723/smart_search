"""Main API router aggregating all endpoint routers.

This module provides the central router that includes all API endpoints
organized by domain.
"""

from fastapi import APIRouter

from smart_search.api.endpoints import health, search, navigate, analyze, graph, index

# Create main API router with version prefix
api_router = APIRouter(prefix="/api/v1")

# Include endpoint routers
api_router.include_router(health.router)
api_router.include_router(search.router, prefix="/search", tags=["Search"])
api_router.include_router(navigate.router, prefix="/navigate", tags=["Navigate"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["Analyze"])
api_router.include_router(graph.router, prefix="/graph", tags=["Graph"])
api_router.include_router(index.router, prefix="/index", tags=["Index"])
