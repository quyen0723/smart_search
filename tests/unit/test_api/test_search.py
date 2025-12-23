"""Tests for search API endpoints."""

import pytest
from fastapi.testclient import TestClient

from smart_search.api import create_test_app
from smart_search.api.endpoints import search
from smart_search.search.schemas import SearchType


@pytest.fixture
def client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestSearchEndpoint:
    """Tests for /search endpoint."""

    def test_search_post(self, client):
        """Test POST search."""
        response = client.post(
            "/api/v1/search",
            json={"query": "test function", "limit": 10},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "total_hits" in data
        assert "hits" in data
        assert "processing_time_ms" in data

    def test_search_get(self, client):
        """Test GET search."""
        response = client.get(
            "/api/v1/search",
            params={"q": "authentication"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "authentication"

    def test_search_with_filters(self, client):
        """Test search with filters."""
        response = client.post(
            "/api/v1/search",
            json={
                "query": "handler",
                "search_type": "hybrid",
                "language": "python",
                "limit": 5,
            },
        )
        assert response.status_code == 200

    def test_search_with_offset(self, client):
        """Test search with pagination."""
        response = client.post(
            "/api/v1/search",
            json={"query": "class", "limit": 10, "offset": 5},
        )
        assert response.status_code == 200

    def test_search_empty_query(self, client):
        """Test search with empty query."""
        response = client.post(
            "/api/v1/search",
            json={"query": ""},
        )
        assert response.status_code == 422  # Validation error


class TestRAGSearchEndpoint:
    """Tests for /search/rag endpoint."""

    def test_rag_search(self, client):
        """Test RAG search."""
        response = client.post(
            "/api/v1/search/rag",
            json={"query": "How does authentication work?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "mode" in data
        assert "answer" in data
        assert "contexts" in data

    def test_rag_search_with_mode(self, client):
        """Test RAG search with mode."""
        response = client.post(
            "/api/v1/search/rag",
            json={
                "query": "What is the main function?",
                "mode": "local",
            },
        )
        assert response.status_code == 200

    def test_rag_search_without_explanation(self, client):
        """Test RAG search without explanation."""
        response = client.post(
            "/api/v1/search/rag",
            json={
                "query": "Find error handlers",
                "include_explanation": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == ""  # No explanation


class TestSimilarSearchEndpoint:
    """Tests for /search/similar endpoint."""

    def test_find_similar(self, client):
        """Test find similar code."""
        response = client.post(
            "/api/v1/search/similar",
            json={
                "code": "def hello():\n    print('hello')",
                "limit": 5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "processing_time_ms" in data

    def test_find_similar_with_exclude(self, client):
        """Test find similar with exclusions."""
        response = client.post(
            "/api/v1/search/similar",
            json={
                "code": "class MyClass:\n    pass",
                "exclude_ids": ["id1", "id2"],
            },
        )
        assert response.status_code == 200


class TestSuggestEndpoint:
    """Tests for /search/suggest endpoint."""

    def test_suggest(self, client):
        """Test search suggestions."""
        response = client.get(
            "/api/v1/search/suggest",
            params={"q": "auth"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "suggestions" in data

    def test_suggest_with_limit(self, client):
        """Test suggestions with limit."""
        response = client.get(
            "/api/v1/search/suggest",
            params={"q": "test", "limit": 5},
        )
        assert response.status_code == 200


class TestSearchModels:
    """Tests for search models."""

    def test_search_request_model(self):
        """Test SearchRequest model."""
        from smart_search.api.endpoints.search import SearchRequest

        request = SearchRequest(
            query="test",
            search_type=SearchType.HYBRID,
            limit=20,
        )
        assert request.query == "test"
        assert request.search_type == SearchType.HYBRID
        assert request.limit == 20

    def test_search_response_model(self):
        """Test SearchResponse model."""
        from smart_search.api.endpoints.search import SearchResponse, SearchHitResponse

        response = SearchResponse(
            query="test",
            total_hits=1,
            hits=[
                SearchHitResponse(
                    id="1",
                    name="test_func",
                    qualified_name="module.test_func",
                    code_type="function",
                    file_path="/test.py",
                    line_start=1,
                    line_end=10,
                    content="def test_func():\n    pass",
                    language="python",
                    score=0.95,
                )
            ],
            processing_time_ms=50.0,
        )
        assert response.total_hits == 1
        assert len(response.hits) == 1

    def test_rag_search_request_model(self):
        """Test RAGSearchRequest model."""
        from smart_search.api.endpoints.search import RAGSearchRequest
        from smart_search.rag import SearchMode

        request = RAGSearchRequest(
            query="How does X work?",
            mode=SearchMode.HYBRID,
        )
        assert request.query == "How does X work?"
        assert request.mode == SearchMode.HYBRID


class TestSearchDependencies:
    """Tests for search dependencies."""

    def test_set_get_searcher(self):
        """Test searcher setter/getter."""
        from smart_search.api.endpoints import search

        original = search._searcher
        try:
            search.set_searcher(None)
            with pytest.raises(Exception):
                search.get_searcher()

            mock_searcher = object()
            search.set_searcher(mock_searcher)
            assert search.get_searcher() is mock_searcher
        finally:
            search._searcher = original

    def test_set_get_graphrag(self):
        """Test graphrag setter/getter."""
        from smart_search.api.endpoints import search

        original = search._graphrag
        try:
            search.set_graphrag(None)
            with pytest.raises(Exception):
                search.get_graphrag()

            mock_graphrag = object()
            search.set_graphrag(mock_graphrag)
            assert search.get_graphrag() is mock_graphrag
        finally:
            search._graphrag = original
