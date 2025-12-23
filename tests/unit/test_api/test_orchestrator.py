"""Tests for API orchestrator."""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient


class TestServiceConfig:
    """Tests for ServiceConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from smart_search.api.orchestrator import ServiceConfig

        config = ServiceConfig()
        assert config.project_root == Path.cwd()
        assert config.meilisearch_url == "http://localhost:7700"
        assert config.embedding_model == "jinaai/jina-embeddings-v3"
        assert config.embedding_dimensions == 1024
        assert config.llm_provider == "openai"
        assert config.enable_cache is True

    def test_custom_config(self):
        """Test custom configuration."""
        from smart_search.api.orchestrator import ServiceConfig

        config = ServiceConfig(
            project_root=Path("/custom/path"),
            meilisearch_url="http://custom:7700",
            llm_provider="anthropic",
            enable_cache=False,
        )
        assert config.project_root == Path("/custom/path")
        assert config.meilisearch_url == "http://custom:7700"
        assert config.llm_provider == "anthropic"
        assert config.enable_cache is False

    def test_from_env(self, monkeypatch):
        """Test configuration from environment."""
        from smart_search.api.orchestrator import ServiceConfig

        monkeypatch.setenv("PROJECT_ROOT", "/env/path")
        monkeypatch.setenv("MEILISEARCH_URL", "http://env:7700")
        monkeypatch.setenv("LLM_PROVIDER", "mock")

        config = ServiceConfig.from_env()
        assert config.project_root == Path("/env/path")
        assert config.meilisearch_url == "http://env:7700"
        assert config.llm_provider == "mock"


class TestServiceRegistry:
    """Tests for ServiceRegistry."""

    def test_empty_registry(self):
        """Test empty registry."""
        from smart_search.api.orchestrator import ServiceRegistry

        registry = ServiceRegistry()
        assert registry.graph is None
        assert registry.searcher is None
        assert registry.indexer is None
        assert registry.graphrag is None
        assert registry.embedder is None
        assert registry.is_initialized() is False

    def test_initialized_registry(self):
        """Test initialized registry."""
        from smart_search.api.orchestrator import ServiceRegistry

        registry = ServiceRegistry(
            graph=object(),
            searcher=object(),
        )
        assert registry.is_initialized() is True


class TestAPIOrchestrator:
    """Tests for APIOrchestrator."""

    def test_create_orchestrator(self):
        """Test creating orchestrator."""
        from smart_search.api.orchestrator import APIOrchestrator, ServiceConfig

        config = ServiceConfig()
        orchestrator = APIOrchestrator(config)
        assert orchestrator.config is config
        assert orchestrator.services is not None
        assert orchestrator._initialized is False

    def test_create_orchestrator_default_config(self):
        """Test orchestrator with default config."""
        from smart_search.api.orchestrator import APIOrchestrator

        orchestrator = APIOrchestrator()
        assert orchestrator.config is not None

    def test_create_app(self):
        """Test creating FastAPI app."""
        from smart_search.api.orchestrator import APIOrchestrator

        orchestrator = APIOrchestrator()
        app = orchestrator.create_app()
        assert app is not None
        assert app.title == "Smart Search API"
        assert app.version == "2.1.0"


class TestCreateApp:
    """Tests for create_app function."""

    def test_create_app(self):
        """Test create_app convenience function."""
        from smart_search.api.orchestrator import create_app

        app = create_app()
        assert app is not None


class TestCreateTestApp:
    """Tests for create_test_app function."""

    def test_create_test_app(self):
        """Test create_test_app function."""
        from smart_search.api import create_test_app

        app = create_test_app()
        assert app is not None
        assert "Test" in app.title

    def test_test_app_root(self):
        """Test test app root endpoint."""
        from smart_search.api import create_test_app

        app = create_test_app()
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Smart Search API (Test)"

    def test_test_app_health(self):
        """Test test app health endpoint."""
        from smart_search.api import create_test_app

        app = create_test_app()
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "test"


class TestMockGraph:
    """Tests for MockGraph."""

    def test_mock_graph(self):
        """Test MockGraph implementation."""
        from smart_search.api import MockGraph

        graph = MockGraph()
        assert graph.get_node("test") is None
        assert graph.get_all_nodes() == []
        assert graph.get_nodes_in_file("/test.py") == []
        assert graph.get_callers("test") == []
        assert graph.get_callees("test") == []
        assert graph.get_all_communities() == []
        assert graph.get_community_members("comm") == []
        assert graph.get_all_files() == []
        assert graph.find_node_at_position("/test.py", 1, 1) is None
        assert graph.get_references("test") == []
        assert graph.get_children("test") == []

    def test_mock_graph_stats(self):
        """Test MockGraph stats."""
        from smart_search.api import MockGraph

        graph = MockGraph()
        stats = graph.get_stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0

    def test_mock_graph_persistence(self):
        """Test MockGraph load/save."""
        from smart_search.api import MockGraph

        graph = MockGraph()
        # Should not raise
        graph.load(Path("/test"))
        graph.save(Path("/test"))


class TestMockSearcher:
    """Tests for MockSearcher."""

    @pytest.mark.asyncio
    async def test_mock_searcher(self):
        """Test MockSearcher implementation."""
        from smart_search.api import MockSearcher
        from smart_search.search.schemas import SearchQuery

        searcher = MockSearcher()
        query = SearchQuery(query="test")
        result = await searcher.search(query)
        assert result.query == "test"
        assert result.hits == []
        assert result.total == 0

    @pytest.mark.asyncio
    async def test_mock_searcher_lifecycle(self):
        """Test MockSearcher lifecycle."""
        from smart_search.api import MockSearcher

        searcher = MockSearcher()
        await searcher.initialize()
        await searcher.close()


class TestMockIndexer:
    """Tests for MockIndexer."""

    @pytest.mark.asyncio
    async def test_mock_indexer(self):
        """Test MockIndexer implementation."""
        from smart_search.api import MockIndexer

        indexer = MockIndexer()
        await indexer.index_file(Path("/test.py"))
        assert await indexer.remove_path("/test") == 0

    @pytest.mark.asyncio
    async def test_mock_indexer_stats(self):
        """Test MockIndexer stats."""
        from smart_search.api import MockIndexer

        indexer = MockIndexer()
        stats = await indexer.get_stats()
        assert stats["total_files"] == 0

    @pytest.mark.asyncio
    async def test_mock_indexer_files(self):
        """Test MockIndexer file operations."""
        from smart_search.api import MockIndexer

        indexer = MockIndexer()
        files = await indexer.list_files()
        assert files == []

        info = await indexer.get_file_info("/test.py")
        assert info is None

    @pytest.mark.asyncio
    async def test_mock_indexer_refresh(self):
        """Test MockIndexer refresh."""
        from smart_search.api import MockIndexer

        indexer = MockIndexer()
        result = await indexer.refresh()
        assert result["updated"] == 0

    @pytest.mark.asyncio
    async def test_mock_indexer_clear(self):
        """Test MockIndexer clear."""
        from smart_search.api import MockIndexer

        indexer = MockIndexer()
        await indexer.clear()  # Should not raise

    @pytest.mark.asyncio
    async def test_mock_indexer_health(self):
        """Test MockIndexer health check."""
        from smart_search.api import MockIndexer

        indexer = MockIndexer()
        health = await indexer.health_check()
        assert health["healthy"] is True


class TestMockGraphRAG:
    """Tests for MockGraphRAG."""

    @pytest.mark.asyncio
    async def test_mock_graphrag_query(self):
        """Test MockGraphRAG query."""
        from smart_search.api import MockGraphRAG

        graphrag = MockGraphRAG()
        result = await graphrag.query("How does X work?")
        assert result.query == "How does X work?"
        assert result.response == "Mock response"

    @pytest.mark.asyncio
    async def test_mock_graphrag_similar(self):
        """Test MockGraphRAG find similar."""
        from smart_search.api import MockGraphRAG

        graphrag = MockGraphRAG()
        results = await graphrag.find_similar("def test():\n    pass")
        assert results == []

    @pytest.mark.asyncio
    async def test_mock_graphrag_explain(self):
        """Test MockGraphRAG explain code."""
        from smart_search.api import MockGraphRAG

        graphrag = MockGraphRAG()
        result = await graphrag.explain_code("test_id")
        assert result.response == "Mock explanation"


class TestAPIModuleExports:
    """Tests for API module exports."""

    def test_exports(self):
        """Test module exports."""
        from smart_search.api import (
            APIOrchestrator,
            ServiceConfig,
            ServiceRegistry,
            create_app,
            create_test_app,
            MockGraph,
            MockSearcher,
            MockIndexer,
            MockGraphRAG,
        )

        # All exports should be accessible
        assert APIOrchestrator is not None
        assert ServiceConfig is not None
        assert ServiceRegistry is not None
        assert create_app is not None
        assert create_test_app is not None
        assert MockGraph is not None
        assert MockSearcher is not None
        assert MockIndexer is not None
        assert MockGraphRAG is not None
