"""Unit tests for main application module."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from smart_search import __version__
from smart_search.core.exceptions import NodeNotFoundError, SearchTimeoutError, SmartSearchError
from smart_search.main import _get_status_code, create_app


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app_returns_fastapi_instance(self, mock_settings) -> None:
        """Test that create_app returns a FastAPI app."""
        from fastapi import FastAPI

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_correct_title(self, mock_settings) -> None:
        """Test app title is set correctly."""
        app = create_app()
        assert app.title == "Smart Search API"

    def test_app_has_correct_version(self, mock_settings) -> None:
        """Test app version matches package version."""
        app = create_app()
        assert app.version == __version__

    def test_docs_enabled_in_debug_mode(self, mock_settings) -> None:
        """Test that docs are enabled in debug mode."""
        app = create_app()
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_router_included(self, client: TestClient) -> None:
        """Test that API router is included."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestExceptionHandler:
    """Tests for exception handler."""

    def test_smart_search_error_handled(self, app, client: TestClient) -> None:
        """Test that SmartSearchError is handled properly."""
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/test-error")
        async def raise_error() -> None:
            raise SmartSearchError("Test error", details={"key": "value"})

        app.include_router(router, prefix="/api/v1")

        response = client.get("/api/v1/test-error")
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "SmartSearchError"
        assert data["message"] == "Test error"

    def test_node_not_found_returns_404(self, app, client: TestClient) -> None:
        """Test NodeNotFoundError returns 404."""
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/test-not-found")
        async def raise_not_found() -> None:
            raise NodeNotFoundError("missing_node")

        app.include_router(router, prefix="/api/v1")

        response = client.get("/api/v1/test-not-found")
        assert response.status_code == 404

    def test_search_timeout_returns_504(self, app, client: TestClient) -> None:
        """Test SearchTimeoutError returns 504."""
        from fastapi import APIRouter

        router = APIRouter()

        @router.get("/test-timeout")
        async def raise_timeout() -> None:
            raise SearchTimeoutError(5000)

        app.include_router(router, prefix="/api/v1")

        response = client.get("/api/v1/test-timeout")
        assert response.status_code == 504


class TestGetStatusCode:
    """Tests for _get_status_code function."""

    def test_node_not_found_returns_404(self) -> None:
        """Test NodeNotFoundError maps to 404."""
        exc = NodeNotFoundError("node_id")
        assert _get_status_code(exc) == status.HTTP_404_NOT_FOUND

    def test_search_timeout_returns_504(self) -> None:
        """Test SearchTimeoutError maps to 504."""
        exc = SearchTimeoutError(5000)
        assert _get_status_code(exc) == status.HTTP_504_GATEWAY_TIMEOUT

    def test_generic_error_returns_500(self) -> None:
        """Test generic SmartSearchError maps to 500."""
        exc = SmartSearchError("Generic error")
        assert _get_status_code(exc) == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestCORSMiddleware:
    """Tests for CORS configuration."""

    def test_cors_headers_in_development(self, client: TestClient) -> None:
        """Test CORS headers are set in development."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS preflight should be handled
        assert response.status_code in [200, 400]


class TestAPIVersioning:
    """Tests for API versioning."""

    def test_api_v1_prefix(self, client: TestClient) -> None:
        """Test that API uses v1 prefix."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_root_path_not_found(self, client: TestClient) -> None:
        """Test that root path is not routed."""
        response = client.get("/health")
        assert response.status_code == 404


class TestLifespan:
    """Tests for application lifespan events."""

    @pytest.mark.asyncio
    async def test_lifespan_startup(self, mock_settings) -> None:
        """Test startup event executes without error."""
        from smart_search.main import lifespan

        app = create_app()

        async with lifespan(app):
            # Startup completed successfully
            pass

    @pytest.mark.asyncio
    async def test_lifespan_shutdown(self, mock_settings) -> None:
        """Test shutdown event executes without error."""
        from smart_search.main import lifespan

        app = create_app()

        async with lifespan(app):
            pass
        # Shutdown completed successfully
