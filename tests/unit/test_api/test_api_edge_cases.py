"""Edge case tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from smart_search.api import create_test_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestSearchEdgeCases:
    """Edge case tests for search endpoints."""

    def test_search_long_query(self, client):
        """Test search with long query."""
        long_query = "test " * 100
        response = client.post(
            "/api/v1/search",
            json={"query": long_query[:1000], "limit": 5},
        )
        assert response.status_code == 200

    def test_search_special_characters(self, client):
        """Test search with special characters."""
        response = client.post(
            "/api/v1/search",
            json={"query": "function(arg1, arg2)", "limit": 5},
        )
        assert response.status_code == 200

    def test_search_unicode(self, client):
        """Test search with unicode."""
        response = client.post(
            "/api/v1/search",
            json={"query": "函数 функция função", "limit": 5},
        )
        assert response.status_code == 200

    def test_rag_search_modes(self, client):
        """Test RAG search with different modes."""
        for mode in ["local", "global", "hybrid", "drift"]:
            response = client.post(
                "/api/v1/search/rag",
                json={"query": "How does authentication work?", "mode": mode},
            )
            assert response.status_code == 200

    def test_similar_empty_code(self, client):
        """Test similar with minimal code."""
        response = client.post(
            "/api/v1/search/similar",
            json={"code": "x"},
        )
        assert response.status_code == 200


class TestNavigateEdgeCases:
    """Edge case tests for navigate endpoints."""

    def test_definition_boundary_line(self, client):
        """Test definition at line boundary."""
        response = client.post(
            "/api/v1/navigate/definition",
            json={"file_path": "/test.py", "line": 1, "column": 0},
        )
        assert response.status_code == 200

    def test_references_with_large_limit(self, client):
        """Test references with large limit."""
        response = client.get(
            "/api/v1/navigate/references/test_id",
            params={"limit": 200},
        )
        assert response.status_code == 200

    def test_hierarchy_deep_depth(self, client):
        """Test hierarchy with deep depth."""
        response = client.get(
            "/api/v1/navigate/hierarchy/test_id",
            params={"max_depth": 10},
        )
        assert response.status_code == 404

    def test_outline_path_with_spaces(self, client):
        """Test outline with path containing spaces."""
        response = client.get(
            "/api/v1/navigate/outline",
            params={"file_path": "/path with spaces/test.py"},
        )
        assert response.status_code == 200


class TestGraphEdgeCases:
    """Edge case tests for graph endpoints."""

    def test_nodes_with_pagination(self, client):
        """Test nodes with pagination."""
        response = client.get(
            "/api/v1/graph/nodes",
            params={"limit": 10, "offset": 5},
        )
        assert response.status_code == 200

    def test_edges_with_type_filter(self, client):
        """Test edges with type filter."""
        response = client.get(
            "/api/v1/graph/edges",
            params={"edge_type": "CALLS"},
        )
        assert response.status_code == 200

    def test_subgraph_max_nodes(self, client):
        """Test subgraph with max nodes."""
        response = client.get(
            "/api/v1/graph/subgraph/test_id",
            params={"max_nodes": 200, "depth": 5},
        )
        assert response.status_code == 404

    def test_path_max_depth(self, client):
        """Test path with max depth."""
        response = client.get(
            "/api/v1/graph/path",
            params={"source": "id1", "target": "id2", "max_depth": 10},
        )
        assert response.status_code == 404

    def test_files_with_pattern(self, client):
        """Test files with pattern."""
        response = client.get(
            "/api/v1/graph/files",
            params={"pattern": "**/*.py", "limit": 50},
        )
        assert response.status_code == 200


class TestIndexEdgeCases:
    """Edge case tests for index endpoints."""

    def test_index_multiple_paths(self, client):
        """Test index with multiple paths."""
        response = client.post(
            "/api/v1/index",
            json={
                "paths": ["/path1", "/path2", "/path3"],
                "recursive": True,
                "languages": ["python"],
            },
        )
        assert response.status_code == 200

    def test_index_with_excludes(self, client):
        """Test index with exclude patterns."""
        response = client.post(
            "/api/v1/index",
            json={
                "paths": ["/project"],
                "exclude_patterns": ["**/test/**", "**/node_modules/**", "**/__pycache__/**"],
            },
        )
        assert response.status_code == 200

    def test_update_full_mode(self, client):
        """Test update in full mode."""
        response = client.post(
            "/api/v1/index/update",
            json={"paths": ["/project"], "mode": "full"},
        )
        assert response.status_code == 200

    def test_files_with_offset(self, client):
        """Test files with offset."""
        response = client.get(
            "/api/v1/index/files",
            params={"offset": 10, "limit": 20},
        )
        assert response.status_code == 200


class TestAnalyzeEdgeCases:
    """Edge case tests for analyze endpoints."""

    def test_explain_without_context(self, client):
        """Test explain without context."""
        response = client.post(
            "/api/v1/analyze/explain",
            json={"code_id": "test_id", "include_context": False},
        )
        assert response.status_code == 404

    def test_impact_no_indirect(self, client):
        """Test impact without indirect deps."""
        response = client.post(
            "/api/v1/analyze/impact",
            json={"code_id": "test_id", "include_indirect": False},
        )
        assert response.status_code == 404

    def test_dependencies_deep_depth(self, client):
        """Test dependencies with deep depth."""
        response = client.get(
            "/api/v1/analyze/dependencies/test_id",
            params={"depth": 5},
        )
        assert response.status_code == 404

    def test_duplicates_high_threshold(self, client):
        """Test duplicates with high threshold."""
        response = client.get(
            "/api/v1/analyze/duplicates/test_id",
            params={"threshold": 0.99},
        )
        assert response.status_code == 404


class TestRootAndHealthEndpoints:
    """Tests for root and health endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestValidationErrors:
    """Tests for validation errors."""

    def test_search_invalid_limit(self, client):
        """Test search with invalid limit."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "limit": 1000},
        )
        assert response.status_code == 422

    def test_search_negative_offset(self, client):
        """Test search with negative offset."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test", "offset": -1},
        )
        assert response.status_code == 422

    def test_rag_empty_query(self, client):
        """Test RAG with empty query."""
        response = client.post(
            "/api/v1/search/rag",
            json={"query": ""},
        )
        assert response.status_code == 422

    def test_similar_empty_code(self, client):
        """Test similar with empty code."""
        response = client.post(
            "/api/v1/search/similar",
            json={"code": ""},
        )
        assert response.status_code == 422

    def test_definition_negative_line(self, client):
        """Test definition with negative line."""
        response = client.post(
            "/api/v1/navigate/definition",
            json={"file_path": "/test.py", "line": -1, "column": 0},
        )
        assert response.status_code == 422

    def test_callers_invalid_depth(self, client):
        """Test callers with invalid depth."""
        response = client.get(
            "/api/v1/navigate/callers/test_id",
            params={"depth": 100},
        )
        assert response.status_code == 422

    def test_impact_invalid_depth(self, client):
        """Test impact with invalid depth."""
        response = client.post(
            "/api/v1/analyze/impact",
            json={"code_id": "test", "max_depth": 100},
        )
        assert response.status_code == 422

    def test_duplicates_invalid_threshold(self, client):
        """Test duplicates with invalid threshold."""
        response = client.get(
            "/api/v1/analyze/duplicates/test_id",
            params={"threshold": 2.0},
        )
        assert response.status_code == 422

    def test_index_invalid_paths(self, client):
        """Test index with invalid paths type."""
        response = client.post(
            "/api/v1/index",
            json={"paths": "not_a_list"},
        )
        assert response.status_code == 422

    def test_clear_no_confirm(self, client):
        """Test clear without confirm param."""
        response = client.delete("/api/v1/index/clear")
        assert response.status_code == 422
