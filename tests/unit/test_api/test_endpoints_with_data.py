"""Tests for API endpoints with mock data."""

import pytest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient


class MockNodeType(Enum):
    """Mock node type enum."""
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"


@dataclass
class MockNode:
    """Mock node for testing."""
    id: str
    name: str
    qualified_name: str
    type: MockNodeType
    file_path: str
    line_start: int
    line_end: int
    parent_id: str | None = None
    docstring: str | None = None
    content: str = ""


class MockGraphWithData:
    """Mock graph with test data."""

    def __init__(self):
        self._nodes = {
            "func1": MockNode(
                id="func1",
                name="login_user",
                qualified_name="auth.login_user",
                type=MockNodeType.FUNCTION,
                file_path="/project/auth.py",
                line_start=10,
                line_end=25,
                content="def login_user(username, password):\n    pass",
            ),
            "func2": MockNode(
                id="func2",
                name="logout_user",
                qualified_name="auth.logout_user",
                type=MockNodeType.FUNCTION,
                file_path="/project/auth.py",
                line_start=30,
                line_end=40,
            ),
            "class1": MockNode(
                id="class1",
                name="UserService",
                qualified_name="services.UserService",
                type=MockNodeType.CLASS,
                file_path="/project/services.py",
                line_start=5,
                line_end=50,
            ),
            "method1": MockNode(
                id="method1",
                name="get_user",
                qualified_name="services.UserService.get_user",
                type=MockNodeType.METHOD,
                file_path="/project/services.py",
                line_start=10,
                line_end=20,
                parent_id="class1",
            ),
        }
        self._callers = {
            "func1": ["func2", "method1"],
            "func2": [],
            "method1": [],
        }
        self._callees = {
            "func2": ["func1"],
            "method1": ["func1"],
        }
        self._communities = {
            "comm_auth": ["func1", "func2"],
            "comm_service": ["class1", "method1"],
        }

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def get_all_nodes(self):
        return list(self._nodes.values())

    def get_nodes_in_file(self, file_path: str):
        return [n for n in self._nodes.values() if n.file_path == file_path]

    def get_callers(self, node_id: str, depth: int = 1):
        return self._callers.get(node_id, [])

    def get_callees(self, node_id: str, depth: int = 1):
        return self._callees.get(node_id, [])

    def get_stats(self):
        return {
            "total_nodes": len(self._nodes),
            "total_edges": 3,
            "node_types": {"function": 2, "class": 1, "method": 1},
            "edge_types": {"CALLS": 3},
            "files_count": 2,
            "communities_count": 2,
            "avg_degree": 1.5,
        }

    def get_all_communities(self):
        return list(self._communities.keys())

    def get_community_members(self, community_id: str):
        return self._communities.get(community_id, [])

    def get_all_files(self):
        return list(set(n.file_path for n in self._nodes.values()))

    def find_node_at_position(self, file_path: str, line: int, column: int):
        for node in self._nodes.values():
            if node.file_path == file_path and node.line_start <= line <= node.line_end:
                return node
        return None

    def get_definition(self, node_id: str):
        return self._nodes.get(node_id)

    def get_references(self, node_id: str, limit: int = 50):
        refs = []
        for caller_id in self._callers.get(node_id, []):
            if caller_id in self._nodes:
                refs.append(self._nodes[caller_id])
        return refs[:limit]

    def get_children(self, node_id: str):
        return [n.id for n in self._nodes.values() if n.parent_id == node_id]

    def load(self, path):
        pass

    def save(self, path):
        pass


@dataclass
class MockSearchHit:
    """Mock search hit."""
    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: Path
    line_start: int
    line_end: int
    content: str
    language: str
    score: float
    highlights: dict


class MockSearcherWithData:
    """Mock searcher with data."""

    async def search(self, query):
        from smart_search.search.schemas import SearchResult, SearchHit, SearchType

        hits = [
            SearchHit(
                id="func1",
                name="login_user",
                qualified_name="auth.login_user",
                code_type="function",
                file_path=Path("/project/auth.py"),
                line_start=10,
                line_end=25,
                content="def login_user():\n    pass",
                language="python",
                score=0.95,
                highlights={"content": ["<em>login</em>_user"]},
            ),
        ]
        return SearchResult(
            query=query.query,
            hits=hits,
            total=1,
            search_type=query.search_type,
        )

    async def initialize(self):
        pass

    async def close(self):
        pass


@dataclass
class MockCodeContext:
    """Mock code context for GraphRAG."""
    code_id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: Path
    line_start: int
    line_end: int
    content: str
    language: str
    relevance_score: float

    def to_dict(self):
        return {
            "code_id": self.code_id,
            "name": self.name,
            "content": self.content,
        }


class MockGraphRAGWithData:
    """Mock GraphRAG with data."""

    async def query(self, query: str, mode=None, filters=None):
        from smart_search.rag import SearchMode
        from dataclasses import dataclass, field

        @dataclass
        class MockGeneration:
            citations: list = field(default_factory=lambda: ["auth.login_user"])

        @dataclass
        class MockResult:
            query: str
            mode: SearchMode
            response: str = "Login is handled by the login_user function."
            contexts: list = field(default_factory=list)
            total_time_ms: float = 150.0
            generation: MockGeneration = field(default_factory=MockGeneration)

        ctx = MockCodeContext(
            code_id="func1",
            name="login_user",
            qualified_name="auth.login_user",
            code_type="function",
            file_path=Path("/project/auth.py"),
            line_start=10,
            line_end=25,
            content="def login_user():\n    pass",
            language="python",
            relevance_score=0.95,
        )

        return MockResult(
            query=query,
            mode=mode or SearchMode.HYBRID,
            contexts=[ctx],
        )

    async def find_similar(self, code_content: str, limit: int = 5):
        return [
            MockCodeContext(
                code_id="func2",
                name="logout_user",
                qualified_name="auth.logout_user",
                code_type="function",
                file_path=Path("/project/auth.py"),
                line_start=30,
                line_end=40,
                content="def logout_user():\n    pass",
                language="python",
                relevance_score=0.85,
            )
        ]

    async def explain_code(self, code_id: str, question: str | None = None):
        @dataclass
        class MockExplainResult:
            response: str = "This function handles user login."

        return MockExplainResult()


class MockIndexerWithData:
    """Mock indexer with data."""

    async def index_file(self, file_path, force=False):
        pass

    async def remove_path(self, path: str) -> int:
        return 2

    async def get_stats(self):
        return {
            "total_files": 10,
            "total_code_units": 50,
            "languages": {"python": 8, "javascript": 2},
            "files_by_type": {".py": 8, ".js": 2},
            "last_indexed": "2024-01-01T00:00:00Z",
            "index_size_bytes": 1024000,
        }

    async def list_files(self, pattern=None, language=None, limit=100, offset=0):
        return [
            {
                "file_path": "/project/auth.py",
                "language": "python",
                "size_bytes": 1024,
                "last_modified": "2024-01-01",
                "code_units": 5,
                "indexed_at": "2024-01-01T12:00:00Z",
            }
        ]

    async def get_file_info(self, file_path: str):
        return {
            "file_path": file_path,
            "language": "python",
            "size_bytes": 1024,
            "last_modified": "2024-01-01",
            "code_units": 5,
            "indexed_at": "2024-01-01T12:00:00Z",
        }

    async def refresh(self):
        return {"updated": 5}

    async def clear(self):
        pass

    async def health_check(self):
        return {"healthy": True, "services": {"meilisearch": True}}


def create_app_with_data() -> FastAPI:
    """Create test app with mock data."""
    app = FastAPI(title="Smart Search Test API")

    # Create mock services with data
    mock_graph = MockGraphWithData()
    mock_searcher = MockSearcherWithData()
    mock_indexer = MockIndexerWithData()
    mock_graphrag = MockGraphRAGWithData()

    # Import and configure endpoints
    from smart_search.api.endpoints import search, navigate, analyze, graph, index

    search.set_searcher(mock_searcher)
    search.set_graphrag(mock_graphrag)
    navigate.set_graph(mock_graph)
    navigate.set_searcher(mock_searcher)
    analyze.set_graph(mock_graph)
    analyze.set_graphrag(mock_graphrag)
    graph.set_graph(mock_graph)
    index.set_indexer(mock_indexer)

    app.include_router(search.router, prefix="/api/v1")
    app.include_router(navigate.router, prefix="/api/v1")
    app.include_router(analyze.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(index.router, prefix="/api/v1")

    return app


@pytest.fixture
def client_with_data():
    """Create test client with data."""
    app = create_app_with_data()
    return TestClient(app)


class TestSearchWithData:
    """Tests for search endpoints with data."""

    def test_search_with_results(self, client_with_data):
        """Test search returning results."""
        response = client_with_data.post(
            "/api/v1/search",
            json={"query": "login"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_hits"] == 1
        assert len(data["hits"]) == 1
        assert data["hits"][0]["name"] == "login_user"

    def test_rag_search_with_answer(self, client_with_data):
        """Test RAG search with answer."""
        response = client_with_data.post(
            "/api/v1/search/rag",
            json={"query": "How does login work?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "login_user" in data["answer"]
        assert len(data["contexts"]) > 0
        assert len(data["citations"]) > 0

    def test_find_similar_with_results(self, client_with_data):
        """Test find similar with results."""
        response = client_with_data.post(
            "/api/v1/search/similar",
            json={"code": "def login():\n    pass"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) > 0


class TestNavigateWithData:
    """Tests for navigate endpoints with data."""

    def test_definition_found(self, client_with_data):
        """Test definition found."""
        response = client_with_data.post(
            "/api/v1/navigate/definition",
            json={"file_path": "/project/auth.py", "line": 15, "column": 5},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["definition"]["name"] == "login_user"

    def test_callers_found(self, client_with_data):
        """Test callers found."""
        response = client_with_data.get("/api/v1/navigate/callers/func1")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_callees_found(self, client_with_data):
        """Test callees found."""
        response = client_with_data.get("/api/v1/navigate/callees/func2")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1

    def test_references_found(self, client_with_data):
        """Test references found."""
        response = client_with_data.get("/api/v1/navigate/references/func1")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 0

    def test_hierarchy_found(self, client_with_data):
        """Test hierarchy found."""
        response = client_with_data.get("/api/v1/navigate/hierarchy/class1")
        assert response.status_code == 200
        data = response.json()
        assert data["root"]["name"] == "UserService"

    def test_outline_found(self, client_with_data):
        """Test outline found."""
        response = client_with_data.get(
            "/api/v1/navigate/outline",
            params={"file_path": "/project/auth.py"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) > 0

    def test_symbols_found(self, client_with_data):
        """Test symbols search."""
        response = client_with_data.get(
            "/api/v1/navigate/symbols",
            params={"query": "login"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["symbols"]) > 0


class TestAnalyzeWithData:
    """Tests for analyze endpoints with data."""

    def test_explain_found(self, client_with_data):
        """Test code explanation."""
        response = client_with_data.post(
            "/api/v1/analyze/explain",
            json={"code_id": "func1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["code_id"] == "func1"
        assert "explanation" in data

    def test_impact_found(self, client_with_data):
        """Test impact analysis."""
        response = client_with_data.post(
            "/api/v1/analyze/impact",
            json={"code_id": "func1"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_affected"] >= 0

    def test_metrics_found(self, client_with_data):
        """Test metrics."""
        response = client_with_data.get("/api/v1/analyze/metrics/func1")
        assert response.status_code == 200
        data = response.json()
        assert data["code_id"] == "func1"

    def test_dependencies_found(self, client_with_data):
        """Test dependencies."""
        response = client_with_data.get("/api/v1/analyze/dependencies/func1")
        assert response.status_code == 200
        data = response.json()
        assert "graph" in data


class TestAnalyzeWithDataMore:
    """More tests for analyze endpoints."""

    def test_duplicates_not_found(self, client_with_data):
        """Test duplicates for non-existent code."""
        response = client_with_data.get("/api/v1/analyze/duplicates/nonexistent")
        assert response.status_code == 404

    def test_explain_not_found(self, client_with_data):
        """Test explain for non-existent code."""
        response = client_with_data.post(
            "/api/v1/analyze/explain",
            json={"code_id": "nonexistent"},
        )
        assert response.status_code == 404

    def test_impact_with_indirect(self, client_with_data):
        """Test impact with indirect dependencies."""
        response = client_with_data.post(
            "/api/v1/analyze/impact",
            json={"code_id": "func1", "max_depth": 3, "include_indirect": True},
        )
        assert response.status_code == 200

    def test_dependencies_out(self, client_with_data):
        """Test dependencies outgoing only."""
        response = client_with_data.get(
            "/api/v1/analyze/dependencies/func2",
            params={"direction": "out"},
        )
        assert response.status_code == 200

    def test_dependencies_in(self, client_with_data):
        """Test dependencies incoming only."""
        response = client_with_data.get(
            "/api/v1/analyze/dependencies/func1",
            params={"direction": "in"},
        )
        assert response.status_code == 200


class TestGraphWithData:
    """Tests for graph endpoints with data."""

    def test_stats_with_data(self, client_with_data):
        """Test stats with data."""
        response = client_with_data.get("/api/v1/graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_nodes"] == 4
        assert data["total_edges"] == 3

    def test_node_found(self, client_with_data):
        """Test node found."""
        response = client_with_data.get("/api/v1/graph/node/func1")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "login_user"

    def test_nodes_list(self, client_with_data):
        """Test nodes list."""
        response = client_with_data.get("/api/v1/graph/nodes")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 4

    def test_nodes_by_type(self, client_with_data):
        """Test nodes by type."""
        response = client_with_data.get(
            "/api/v1/graph/nodes",
            params={"code_type": "function"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_edges_list(self, client_with_data):
        """Test edges list."""
        response = client_with_data.get("/api/v1/graph/edges")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_subgraph_found(self, client_with_data):
        """Test subgraph found."""
        response = client_with_data.get("/api/v1/graph/subgraph/func1")
        assert response.status_code == 200
        data = response.json()
        assert data["center"] == "func1"

    def test_communities_list(self, client_with_data):
        """Test communities list."""
        response = client_with_data.get("/api/v1/graph/communities")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_community_found(self, client_with_data):
        """Test community found."""
        response = client_with_data.get("/api/v1/graph/community/comm_auth")
        assert response.status_code == 200
        data = response.json()
        assert data["size"] == 2

    def test_files_list(self, client_with_data):
        """Test files list."""
        response = client_with_data.get("/api/v1/graph/files")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_file_nodes(self, client_with_data):
        """Test file nodes."""
        response = client_with_data.get("/api/v1/graph/file//project/auth.py/nodes")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_subgraph_with_direction(self, client_with_data):
        """Test subgraph with direction."""
        response = client_with_data.get(
            "/api/v1/graph/subgraph/func2",
            params={"direction": "out"},
        )
        assert response.status_code == 200

    def test_subgraph_in_direction(self, client_with_data):
        """Test subgraph incoming direction."""
        response = client_with_data.get(
            "/api/v1/graph/subgraph/func1",
            params={"direction": "in"},
        )
        assert response.status_code == 200

    def test_edges_by_source(self, client_with_data):
        """Test edges by source."""
        response = client_with_data.get(
            "/api/v1/graph/edges",
            params={"source_id": "func2"},
        )
        assert response.status_code == 200

    def test_edges_by_target(self, client_with_data):
        """Test edges by target."""
        response = client_with_data.get(
            "/api/v1/graph/edges",
            params={"target_id": "func1"},
        )
        assert response.status_code == 200

    def test_nodes_by_file(self, client_with_data):
        """Test nodes by file path."""
        response = client_with_data.get(
            "/api/v1/graph/nodes",
            params={"file_path": "/project/auth.py"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2


class TestIndexWithData:
    """Tests for index endpoints with data."""

    def test_stats_with_data(self, client_with_data):
        """Test stats with data."""
        response = client_with_data.get("/api/v1/index/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 10
        assert data["total_code_units"] == 50

    def test_files_list(self, client_with_data):
        """Test files list."""
        response = client_with_data.get("/api/v1/index/files")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1

    def test_file_info_found(self, client_with_data):
        """Test file info found."""
        response = client_with_data.get("/api/v1/index/file/project/auth.py")
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "python"

    def test_remove_with_count(self, client_with_data):
        """Test remove with count."""
        response = client_with_data.post(
            "/api/v1/index/remove",
            json={"paths": ["/project/old.py"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["removed_files"] == 2

    def test_health_healthy(self, client_with_data):
        """Test health healthy."""
        response = client_with_data.get("/api/v1/index/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
