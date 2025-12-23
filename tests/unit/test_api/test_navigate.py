"""Tests for navigate API endpoints."""

import pytest
from fastapi.testclient import TestClient

from smart_search.api import create_test_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestDefinitionEndpoint:
    """Tests for /navigate/definition endpoint."""

    def test_go_to_definition(self, client):
        """Test go to definition."""
        response = client.post(
            "/api/v1/navigate/definition",
            json={
                "file_path": "/test.py",
                "line": 10,
                "column": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "found" in data
        # Mock returns not found
        assert data["found"] is False

    def test_go_to_definition_with_symbol(self, client):
        """Test go to definition with symbol hint."""
        response = client.post(
            "/api/v1/navigate/definition",
            json={
                "file_path": "/test.py",
                "line": 10,
                "column": 5,
                "symbol": "my_function",
            },
        )
        assert response.status_code == 200


class TestReferencesEndpoint:
    """Tests for /navigate/references endpoint."""

    def test_find_references(self, client):
        """Test find references."""
        response = client.get("/api/v1/navigate/references/test_id")
        assert response.status_code == 200
        data = response.json()
        assert "code_id" in data
        assert "total" in data
        assert "references" in data

    def test_find_references_include_definition(self, client):
        """Test find references with definition."""
        response = client.get(
            "/api/v1/navigate/references/test_id",
            params={"include_definition": True},
        )
        assert response.status_code == 200


class TestCallersEndpoint:
    """Tests for /navigate/callers endpoint."""

    def test_get_callers(self, client):
        """Test get callers."""
        response = client.get("/api/v1/navigate/callers/test_id")
        assert response.status_code == 404  # Mock graph has no nodes

    def test_get_callers_with_depth(self, client):
        """Test get callers with depth."""
        response = client.get(
            "/api/v1/navigate/callers/test_id",
            params={"depth": 2},
        )
        assert response.status_code == 404


class TestCalleesEndpoint:
    """Tests for /navigate/callees endpoint."""

    def test_get_callees(self, client):
        """Test get callees."""
        response = client.get("/api/v1/navigate/callees/test_id")
        assert response.status_code == 404  # Mock graph has no nodes

    def test_get_callees_with_limit(self, client):
        """Test get callees with limit."""
        response = client.get(
            "/api/v1/navigate/callees/test_id",
            params={"limit": 10},
        )
        assert response.status_code == 404


class TestHierarchyEndpoint:
    """Tests for /navigate/hierarchy endpoint."""

    def test_get_hierarchy(self, client):
        """Test get type hierarchy."""
        response = client.get("/api/v1/navigate/hierarchy/test_id")
        assert response.status_code == 404

    def test_get_hierarchy_with_direction(self, client):
        """Test hierarchy with direction."""
        response = client.get(
            "/api/v1/navigate/hierarchy/test_id",
            params={"direction": "descendants"},
        )
        assert response.status_code == 404


class TestOutlineEndpoint:
    """Tests for /navigate/outline endpoint."""

    def test_get_outline(self, client):
        """Test get document outline."""
        response = client.get(
            "/api/v1/navigate/outline",
            params={"file_path": "/test.py"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_path" in data
        assert "items" in data

    def test_get_outline_empty_file(self, client):
        """Test outline for empty/nonexistent file."""
        response = client.get(
            "/api/v1/navigate/outline",
            params={"file_path": "/nonexistent.py"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []


class TestSymbolsEndpoint:
    """Tests for /navigate/symbols endpoint."""

    def test_search_symbols(self, client):
        """Test symbol search."""
        response = client.get(
            "/api/v1/navigate/symbols",
            params={"query": "test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "symbols" in data
        assert "total" in data

    def test_search_symbols_with_type(self, client):
        """Test symbol search with type filter."""
        response = client.get(
            "/api/v1/navigate/symbols",
            params={"query": "test", "code_type": "function"},
        )
        assert response.status_code == 200


class TestNavigateModels:
    """Tests for navigate models."""

    def test_code_location_model(self):
        """Test CodeLocation model."""
        from smart_search.api.endpoints.navigate import CodeLocation

        loc = CodeLocation(
            file_path="/test.py",
            line_start=1,
            line_end=10,
            column_start=0,
            column_end=20,
        )
        assert loc.file_path == "/test.py"
        assert loc.line_start == 1

    def test_code_reference_model(self):
        """Test CodeReference model."""
        from smart_search.api.endpoints.navigate import CodeReference, CodeLocation

        ref = CodeReference(
            id="test_id",
            name="test_func",
            qualified_name="module.test_func",
            code_type="function",
            location=CodeLocation(
                file_path="/test.py",
                line_start=1,
                line_end=10,
            ),
        )
        assert ref.id == "test_id"
        assert ref.location.file_path == "/test.py"

    def test_hierarchy_node_model(self):
        """Test HierarchyNode model."""
        from smart_search.api.endpoints.navigate import HierarchyNode

        node = HierarchyNode(
            id="test_id",
            name="TestClass",
            qualified_name="module.TestClass",
            code_type="class",
            children=[],
        )
        assert node.id == "test_id"
        assert node.children == []

    def test_outline_item_model(self):
        """Test OutlineItem model."""
        from smart_search.api.endpoints.navigate import OutlineItem

        item = OutlineItem(
            id="test_id",
            name="test_func",
            code_type="function",
            line_start=1,
            line_end=10,
        )
        assert item.name == "test_func"


class TestNavigateDependencies:
    """Tests for navigate dependencies."""

    def test_set_get_graph(self):
        """Test graph setter/getter."""
        from smart_search.api.endpoints import navigate

        original = navigate._graph
        try:
            navigate.set_graph(None)
            with pytest.raises(Exception):
                navigate.get_graph()

            mock_graph = object()
            navigate.set_graph(mock_graph)
            assert navigate.get_graph() is mock_graph
        finally:
            navigate._graph = original

    def test_set_get_searcher(self):
        """Test searcher setter/getter."""
        from smart_search.api.endpoints import navigate

        original = navigate._searcher
        try:
            navigate.set_searcher(None)
            with pytest.raises(Exception):
                navigate.get_searcher()

            mock_searcher = object()
            navigate.set_searcher(mock_searcher)
            assert navigate.get_searcher() is mock_searcher
        finally:
            navigate._searcher = original
