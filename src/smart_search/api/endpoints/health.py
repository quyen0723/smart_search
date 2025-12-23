"""Health check endpoints for Smart Search API.

Provides endpoints for monitoring application health and readiness.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

from smart_search import __version__
from smart_search.config import get_settings

router = APIRouter(tags=["Health"])


class HealthStatus(BaseModel):
    """Health check response model."""

    status: str
    version: str
    environment: str
    timestamp: str


class DetailedHealthStatus(BaseModel):
    """Detailed health check with component status."""

    status: str
    version: str
    environment: str
    timestamp: str
    components: dict[str, dict[str, Any]]


@router.get(
    "/health",
    response_model=HealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="Returns basic health status of the application.",
)
async def health_check() -> HealthStatus:
    """Check if the application is running.

    This is a lightweight check suitable for load balancer health probes.

    Returns:
        HealthStatus: Basic health information.
    """
    settings = get_settings()
    return HealthStatus(
        status="healthy",
        version=__version__,
        environment=settings.app.env,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get(
    "/health/ready",
    response_model=DetailedHealthStatus,
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Checks if the application is ready to serve requests.",
)
async def readiness_check() -> DetailedHealthStatus:
    """Check if all components are ready.

    This performs deeper checks on dependencies like Meilisearch.
    Use for Kubernetes readiness probes.

    Returns:
        DetailedHealthStatus: Detailed health status with component checks.
    """
    settings = get_settings()
    components: dict[str, dict[str, Any]] = {}

    # Check Meilisearch connectivity
    meilisearch_status = await _check_meilisearch()
    components["meilisearch"] = meilisearch_status

    # Determine overall status
    all_healthy = all(c.get("status") == "healthy" for c in components.values())
    overall_status = "healthy" if all_healthy else "degraded"

    return DetailedHealthStatus(
        status=overall_status,
        version=__version__,
        environment=settings.app.env,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Simple check to verify the application process is running.",
)
async def liveness_check() -> dict[str, str]:
    """Check if the application process is alive.

    This is the simplest possible check - if this responds, the app is alive.
    Use for Kubernetes liveness probes.

    Returns:
        dict: Simple alive status.
    """
    return {"status": "alive"}


async def _check_meilisearch() -> dict[str, Any]:
    """Check Meilisearch connectivity and health.

    Returns:
        dict: Meilisearch component status.
    """
    settings = get_settings()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.meilisearch.url}/health")
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "url": settings.meilisearch.url,
                }
            else:
                return {
                    "status": "unhealthy",
                    "url": settings.meilisearch.url,
                    "error": f"HTTP {response.status_code}",
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "url": settings.meilisearch.url,
            "error": str(e),
        }
