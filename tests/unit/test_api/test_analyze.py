"""Tests for analyze API endpoints."""

import pytest
from fastapi.testclient import TestClient

from smart_search.api import create_test_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_test_app()
    return TestClient(app)


class TestExplainEndpoint:
    """Tests for /analyze/explain endpoint."""

    def test_explain_code(self, client):
        """Test code explanation."""
        response = client.post(
            "/api/v1/analyze/explain",
            json={"code_id": "test_id"},
        )
        # Mock graph has no nodes, so 404
        assert response.status_code == 404

    def test_explain_code_with_question(self, client):
        """Test explanation with specific question."""
        response = client.post(
            "/api/v1/analyze/explain",
            json={
                "code_id": "test_id",
                "question": "What does this function do?",
            },
        )
        assert response.status_code == 404


class TestImpactEndpoint:
    """Tests for /analyze/impact endpoint."""

    def test_analyze_impact(self, client):
        """Test impact analysis."""
        response = client.post(
            "/api/v1/analyze/impact",
            json={"code_id": "test_id"},
        )
        assert response.status_code == 404

    def test_analyze_impact_with_depth(self, client):
        """Test impact analysis with depth."""
        response = client.post(
            "/api/v1/analyze/impact",
            json={
                "code_id": "test_id",
                "max_depth": 5,
                "include_indirect": True,
            },
        )
        assert response.status_code == 404


class TestMetricsEndpoint:
    """Tests for /analyze/metrics endpoint."""

    def test_get_metrics(self, client):
        """Test get code metrics."""
        response = client.get("/api/v1/analyze/metrics/test_id")
        assert response.status_code == 404

    def test_metrics_validation(self, client):
        """Test metrics parameter validation."""
        response = client.get("/api/v1/analyze/metrics/")
        assert response.status_code in [404, 405]  # Missing path param


class TestDependenciesEndpoint:
    """Tests for /analyze/dependencies endpoint."""

    def test_get_dependencies(self, client):
        """Test get dependencies."""
        response = client.get("/api/v1/analyze/dependencies/test_id")
        assert response.status_code == 404

    def test_get_dependencies_with_direction(self, client):
        """Test dependencies with direction."""
        response = client.get(
            "/api/v1/analyze/dependencies/test_id",
            params={"direction": "out", "depth": 3},
        )
        assert response.status_code == 404


class TestDuplicatesEndpoint:
    """Tests for /analyze/duplicates endpoint."""

    def test_find_duplicates(self, client):
        """Test find duplicates."""
        response = client.get("/api/v1/analyze/duplicates/test_id")
        assert response.status_code == 404

    def test_find_duplicates_with_threshold(self, client):
        """Test duplicates with threshold."""
        response = client.get(
            "/api/v1/analyze/duplicates/test_id",
            params={"threshold": 0.9},
        )
        assert response.status_code == 404


class TestSummarizeEndpoint:
    """Tests for /analyze/summarize endpoint."""

    def test_summarize_file(self, client):
        """Test file summarization."""
        response = client.post(
            "/api/v1/analyze/summarize",
            params={"file_path": "/test.py"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "file_path" in data
        assert "components" in data


class TestAnalyzeModels:
    """Tests for analyze models."""

    def test_explain_request_model(self):
        """Test ExplainRequest model."""
        from smart_search.api.endpoints.analyze import ExplainRequest

        request = ExplainRequest(
            code_id="test_id",
            question="What does this do?",
            include_context=True,
        )
        assert request.code_id == "test_id"
        assert request.question == "What does this do?"

    def test_impact_result_model(self):
        """Test ImpactResult model."""
        from smart_search.api.endpoints.analyze import ImpactResult

        result = ImpactResult(
            id="test_id",
            name="test_func",
            qualified_name="module.test_func",
            code_type="function",
            file_path="/test.py",
            distance=1,
            impact_type="direct",
        )
        assert result.distance == 1
        assert result.impact_type == "direct"

    def test_complexity_metrics_model(self):
        """Test ComplexityMetrics model."""
        from smart_search.api.endpoints.analyze import ComplexityMetrics

        metrics = ComplexityMetrics(
            cyclomatic=5,
            cognitive=10,
            lines_of_code=50,
            comment_lines=10,
            blank_lines=5,
        )
        assert metrics.cyclomatic == 5
        assert metrics.lines_of_code == 50

    def test_code_metrics_model(self):
        """Test CodeMetrics model."""
        from smart_search.api.endpoints.analyze import CodeMetrics, ComplexityMetrics

        metrics = CodeMetrics(
            id="test_id",
            name="test_func",
            code_type="function",
            file_path="/test.py",
            complexity=ComplexityMetrics(lines_of_code=50),
            dependencies_in=5,
            dependencies_out=3,
        )
        assert metrics.dependencies_in == 5

    def test_dependency_graph_model(self):
        """Test DependencyGraph model."""
        from smart_search.api.endpoints.analyze import (
            DependencyGraph,
            DependencyNode,
            DependencyEdge,
        )

        graph = DependencyGraph(
            nodes=[
                DependencyNode(
                    id="n1",
                    name="func1",
                    code_type="function",
                    file_path="/test.py",
                )
            ],
            edges=[
                DependencyEdge(
                    source="n1",
                    target="n2",
                    edge_type="calls",
                )
            ],
        )
        assert len(graph.nodes) == 1
        assert len(graph.edges) == 1

    def test_duplicate_match_model(self):
        """Test DuplicateMatch model."""
        from smart_search.api.endpoints.analyze import DuplicateMatch

        match = DuplicateMatch(
            id="test_id",
            name="similar_func",
            file_path="/other.py",
            line_start=10,
            line_end=20,
            similarity=0.92,
        )
        assert match.similarity == 0.92


class TestAnalyzeDependencies:
    """Tests for analyze dependencies."""

    def test_set_get_graph(self):
        """Test graph setter/getter."""
        from smart_search.api.endpoints import analyze

        original = analyze._graph
        try:
            analyze.set_graph(None)
            with pytest.raises(Exception):
                analyze.get_graph()

            mock_graph = object()
            analyze.set_graph(mock_graph)
            assert analyze.get_graph() is mock_graph
        finally:
            analyze._graph = original

    def test_set_get_graphrag(self):
        """Test graphrag setter/getter."""
        from smart_search.api.endpoints import analyze

        original = analyze._graphrag
        try:
            analyze.set_graphrag(None)
            with pytest.raises(Exception):
                analyze.get_graphrag()

            mock_graphrag = object()
            analyze.set_graphrag(mock_graphrag)
            assert analyze.get_graphrag() is mock_graphrag
        finally:
            analyze._graphrag = original
