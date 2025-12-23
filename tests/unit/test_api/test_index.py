"""Tests for index API endpoints."""

import pytest
from fastapi.testclient import TestClient

from smart_search.api import create_test_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestIndexEndpoint:
    """Tests for /index endpoint."""

    def test_start_index(self, client):
        """Test start indexing."""
        response = client.post(
            "/api/v1/index",
            json={"paths": ["/test/project"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "pending"

    def test_start_index_with_options(self, client):
        """Test start indexing with options."""
        response = client.post(
            "/api/v1/index",
            json={
                "paths": ["/test/project"],
                "recursive": True,
                "languages": ["python", "javascript"],
                "force_reindex": True,
            },
        )
        assert response.status_code == 200

    def test_start_index_empty_paths(self, client):
        """Test start indexing with empty paths."""
        response = client.post(
            "/api/v1/index",
            json={"paths": []},
        )
        assert response.status_code == 422  # Validation error


class TestProgressEndpoint:
    """Tests for /index/progress endpoint."""

    def test_get_progress(self, client):
        """Test get progress for job."""
        # First start a job
        response = client.post(
            "/api/v1/index",
            json={"paths": ["/test"]},
        )
        job_id = response.json()["job_id"]

        # Check progress
        response = client.get(f"/api/v1/index/progress/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress_percent" in data

    def test_get_progress_not_found(self, client):
        """Test progress for non-existent job."""
        response = client.get("/api/v1/index/progress/nonexistent")
        assert response.status_code == 404


class TestUpdateEndpoint:
    """Tests for /index/update endpoint."""

    def test_update_index(self, client):
        """Test update index."""
        response = client.post(
            "/api/v1/index/update",
            json={"paths": ["/test/project"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data

    def test_update_index_full(self, client):
        """Test full update."""
        response = client.post(
            "/api/v1/index/update",
            json={"paths": ["/test"], "mode": "full"},
        )
        assert response.status_code == 200


class TestRemoveEndpoint:
    """Tests for /index/remove endpoint."""

    def test_remove_from_index(self, client):
        """Test remove from index."""
        response = client.post(
            "/api/v1/index/remove",
            json={"paths": ["/test/file.py"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "removed_files" in data
        assert "status" in data


class TestStatsEndpoint:
    """Tests for /index/stats endpoint."""

    def test_get_stats(self, client):
        """Test get index stats."""
        response = client.get("/api/v1/index/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_files" in data
        assert "total_code_units" in data
        assert "languages" in data


class TestFilesEndpoint:
    """Tests for /index/files endpoint."""

    def test_list_files(self, client):
        """Test list indexed files."""
        response = client.get("/api/v1/index/files")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_list_files_with_filter(self, client):
        """Test list files with filter."""
        response = client.get(
            "/api/v1/index/files",
            params={"pattern": "*.py", "language": "python"},
        )
        assert response.status_code == 200


class TestFileInfoEndpoint:
    """Tests for /index/file endpoint."""

    def test_get_file_info(self, client):
        """Test get file info."""
        response = client.get("/api/v1/index/file/test.py")
        assert response.status_code == 404  # Mock returns None


class TestRefreshEndpoint:
    """Tests for /index/refresh endpoint."""

    def test_refresh_index(self, client):
        """Test refresh index."""
        response = client.post("/api/v1/index/refresh")
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data


class TestClearEndpoint:
    """Tests for /index/clear endpoint."""

    def test_clear_index(self, client):
        """Test clear index."""
        response = client.delete(
            "/api/v1/index/clear",
            params={"confirm": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_clear_index_no_confirm(self, client):
        """Test clear without confirmation."""
        response = client.delete(
            "/api/v1/index/clear",
            params={"confirm": False},
        )
        assert response.status_code == 400


class TestHealthEndpoint:
    """Tests for /index/health endpoint."""

    def test_health_check(self, client):
        """Test index health check."""
        response = client.get("/api/v1/index/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestIndexModels:
    """Tests for index models."""

    def test_index_request_model(self):
        """Test IndexRequest model."""
        from smart_search.api.endpoints.index import IndexRequest

        request = IndexRequest(
            paths=["/test"],
            recursive=True,
            languages=["python"],
            force_reindex=False,
        )
        assert request.paths == ["/test"]
        assert request.recursive is True

    def test_index_progress_model(self):
        """Test IndexProgress model."""
        from smart_search.api.endpoints.index import IndexProgress

        progress = IndexProgress(
            job_id="abc123",
            status="running",
            total_files=100,
            processed_files=50,
            indexed_files=45,
            failed_files=5,
            progress_percent=50.0,
        )
        assert progress.progress_percent == 50.0
        assert progress.indexed_files == 45

    def test_file_info_model(self):
        """Test FileInfo model."""
        from smart_search.api.endpoints.index import FileInfo

        info = FileInfo(
            file_path="/test.py",
            language="python",
            size_bytes=1024,
            last_modified="2024-01-01",
            code_units=10,
        )
        assert info.file_path == "/test.py"
        assert info.language == "python"

    def test_index_stats_model(self):
        """Test IndexStats model."""
        from smart_search.api.endpoints.index import IndexStats

        stats = IndexStats(
            total_files=100,
            total_code_units=500,
            languages={"python": 80, "javascript": 20},
            files_by_type={".py": 80, ".js": 20},
            index_size_bytes=1024000,
        )
        assert stats.total_files == 100
        assert stats.languages["python"] == 80


class TestIndexHelpers:
    """Tests for index helper functions."""

    def test_generate_job_id(self):
        """Test job ID generation."""
        from smart_search.api.endpoints.index import _generate_job_id

        job_id = _generate_job_id()
        assert isinstance(job_id, str)
        assert len(job_id) == 8

    def test_ext_to_language(self):
        """Test extension to language mapping."""
        from smart_search.api.endpoints.index import _ext_to_language

        assert _ext_to_language(".py") == "python"
        assert _ext_to_language(".js") == "javascript"
        assert _ext_to_language(".ts") == "typescript"
        assert _ext_to_language(".go") == "go"
        assert _ext_to_language(".rs") == "rust"
        assert _ext_to_language(".unknown") == "unknown"


class TestIndexDependencies:
    """Tests for index dependencies."""

    def test_set_get_indexer(self):
        """Test indexer setter/getter."""
        from smart_search.api.endpoints import index

        original = index._indexer
        try:
            index.set_indexer(None)
            with pytest.raises(Exception):
                index.get_indexer()

            mock_indexer = object()
            index.set_indexer(mock_indexer)
            assert index.get_indexer() is mock_indexer
        finally:
            index._indexer = original

    def test_get_jobs(self):
        """Test jobs getter."""
        from smart_search.api.endpoints.index import get_jobs

        jobs = get_jobs()
        assert isinstance(jobs, dict)
