"""Unit tests for health check endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from smart_search import __version__


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_returns_200(self, client: TestClient) -> None:
        """Test that health check returns 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self, client: TestClient) -> None:
        """Test health check response has correct structure."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data

    def test_health_check_status_healthy(self, client: TestClient) -> None:
        """Test that status is 'healthy'."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_check_version(self, client: TestClient) -> None:
        """Test that version matches package version."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == __version__

    def test_health_check_environment(self, client: TestClient) -> None:
        """Test that environment is development in tests."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["environment"] == "development"

    def test_health_check_timestamp_format(self, client: TestClient) -> None:
        """Test that timestamp is ISO format."""
        response = client.get("/api/v1/health")
        data = response.json()
        # ISO format check - should be parseable
        from datetime import datetime

        datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))


class TestLivenessEndpoint:
    """Tests for /health/live endpoint."""

    def test_liveness_returns_200(self, client: TestClient) -> None:
        """Test that liveness check returns 200 OK."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200

    def test_liveness_response(self, client: TestClient) -> None:
        """Test liveness response structure."""
        response = client.get("/api/v1/health/live")
        data = response.json()
        assert data == {"status": "alive"}


class TestReadinessEndpoint:
    """Tests for /health/ready endpoint."""

    def test_readiness_returns_200(self, client: TestClient) -> None:
        """Test that readiness check returns 200."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

    def test_readiness_response_structure(self, client: TestClient) -> None:
        """Test readiness response has correct structure."""
        response = client.get("/api/v1/health/ready")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "environment" in data
        assert "timestamp" in data
        assert "components" in data
        assert isinstance(data["components"], dict)

    def test_readiness_includes_meilisearch_check(self, client: TestClient) -> None:
        """Test that Meilisearch component is checked."""
        response = client.get("/api/v1/health/ready")
        data = response.json()
        assert "meilisearch" in data["components"]

    def test_readiness_meilisearch_component_present(self, client: TestClient) -> None:
        """Test that Meilisearch component is present in readiness check."""
        response = client.get("/api/v1/health/ready")
        data = response.json()

        # Meilisearch component should always be checked
        assert "meilisearch" in data["components"]
        assert "status" in data["components"]["meilisearch"]
        assert data["components"]["meilisearch"]["status"] in ["healthy", "unhealthy"]

    def test_readiness_status_degraded_when_component_unhealthy(
        self, client: TestClient
    ) -> None:
        """Test that overall status is degraded when a component is unhealthy."""
        response = client.get("/api/v1/health/ready")
        data = response.json()

        # Since Meilisearch is not running in tests, it should be unhealthy
        # and overall status should be degraded
        meilisearch_status = data["components"]["meilisearch"]["status"]
        if meilisearch_status == "unhealthy":
            assert data["status"] == "degraded"


class TestHealthEndpointContentType:
    """Tests for response content types."""

    def test_health_content_type_json(self, client: TestClient) -> None:
        """Test that health endpoint returns JSON."""
        response = client.get("/api/v1/health")
        assert response.headers["content-type"] == "application/json"

    def test_readiness_content_type_json(self, client: TestClient) -> None:
        """Test that readiness endpoint returns JSON."""
        response = client.get("/api/v1/health/ready")
        assert response.headers["content-type"] == "application/json"

    def test_liveness_content_type_json(self, client: TestClient) -> None:
        """Test that liveness endpoint returns JSON."""
        response = client.get("/api/v1/health/live")
        assert response.headers["content-type"] == "application/json"
