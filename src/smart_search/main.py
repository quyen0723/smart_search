"""FastAPI application entry point for Smart Search.

This module creates and configures the FastAPI application with all
necessary middleware, routers, and lifecycle hooks.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import time

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from smart_search import __version__
from smart_search.api.router import api_router
from smart_search.config import get_settings
from smart_search.core.exceptions import SmartSearchError
from smart_search.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)

# Performance threshold for slow request warnings (seconds)
_SLOW_REQUEST_THRESHOLD = 1.0


class PerformanceLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request performance metrics.

    Logs duration for every request and warns when requests exceed threshold.
    """

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        response = await call_next(request)

        duration = time.perf_counter() - start_time

        # Log performance metrics
        log_data = {
            "method": request.method,
            "path": request.url.path,
            "duration_ms": round(duration * 1000, 2),
            "status_code": response.status_code,
        }

        if duration > _SLOW_REQUEST_THRESHOLD:
            logger.warning("Slow request detected", **log_data)
        else:
            logger.debug("Request completed", **log_data)

        # Add timing header for debugging
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

        return response


from smart_search.api.orchestrator import APIOrchestrator, ServiceConfig

# Global orchestrator instance
_orchestrator: APIOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Handles startup and shutdown events for the application.
    Initializes logging, connections, and cleans up on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control returns to the application.
    """
    global _orchestrator

    # Startup
    setup_logging()
    settings = get_settings()

    logger.info(
        "Starting Smart Search",
        version=__version__,
        environment=settings.app.env,
        debug=settings.app.debug,
    )

    # Initialize orchestrator and services
    _orchestrator = APIOrchestrator(ServiceConfig.from_env())
    try:
        await _orchestrator.initialize()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.warning(f"Some services failed to initialize: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Smart Search")
    if _orchestrator:
        await _orchestrator.shutdown()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title="Smart Search API",
        description="Intelligent Source Code Navigation System with Hybrid Search, Graph Analysis, and GraphRAG",
        version=__version__,
        docs_url="/docs" if settings.app.debug else None,
        redoc_url="/redoc" if settings.app.debug else None,
        openapi_url="/openapi.json" if settings.app.debug else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app.env == "development" else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add performance logging middleware
    app.add_middleware(PerformanceLoggingMiddleware)

    # Register exception handlers
    app.add_exception_handler(SmartSearchError, smart_search_exception_handler)

    # Include routers
    app.include_router(api_router)

    return app


async def smart_search_exception_handler(
    request: Request,
    exc: SmartSearchError,
) -> JSONResponse:
    """Handle SmartSearchError exceptions.

    Converts SmartSearchError instances to consistent JSON responses.

    Args:
        request: The incoming request.
        exc: The SmartSearchError exception.

    Returns:
        JSONResponse: Formatted error response.
    """
    logger.error(
        "Request failed",
        error=exc.__class__.__name__,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=_get_status_code(exc),
        content=exc.to_dict(),
    )


def _get_status_code(exc: SmartSearchError) -> int:
    """Map exception types to HTTP status codes.

    Args:
        exc: The exception instance.

    Returns:
        int: Appropriate HTTP status code.
    """
    from smart_search.core.exceptions import (
        ConfigurationError,
        NodeNotFoundError,
        SearchConnectionError,
        SearchTimeoutError,
        UnsupportedLanguageError,
    )

    status_map: dict[type, int] = {
        NodeNotFoundError: status.HTTP_404_NOT_FOUND,
        UnsupportedLanguageError: status.HTTP_400_BAD_REQUEST,
        SearchTimeoutError: status.HTTP_504_GATEWAY_TIMEOUT,
        SearchConnectionError: status.HTTP_503_SERVICE_UNAVAILABLE,
        ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    }

    for exc_type, status_code in status_map.items():
        if isinstance(exc, exc_type):
            return status_code

    return status.HTTP_500_INTERNAL_SERVER_ERROR


# Create the application instance
app = create_app()


def main() -> None:
    """Run the application using uvicorn.

    This is the entry point for the CLI command.
    """
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "smart_search.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers if not settings.api.reload else 1,
        log_level=settings.app.log_level.lower(),
    )


if __name__ == "__main__":
    main()
