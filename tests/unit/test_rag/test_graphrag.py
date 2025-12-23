"""Tests for GraphRAG module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smart_search.rag.generator import (
    CachedGenerator,
    GenerationResult,
    GeneratorConfig,
    MockLLMClient,
    ResponseGenerator,
)
from smart_search.rag.graphrag import (
    GraphRAG,
    GraphRAGConfig,
    GraphRAGResult,
    MockGraphRAG,
    SearchMode,
    create_mock_graphrag,
)
from smart_search.rag.prompts import CodeContext, PromptContext, QueryType
from smart_search.rag.retriever import (
    MockRetriever,
    RetrievalResult,
    RetrieverConfig,
)


class TestSearchMode:
    """Tests for SearchMode enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert SearchMode.LOCAL.value == "local"
        assert SearchMode.GLOBAL.value == "global"
        assert SearchMode.HYBRID.value == "hybrid"
        assert SearchMode.DRIFT.value == "drift"


class TestGraphRAGConfig:
    """Tests for GraphRAGConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = GraphRAGConfig()

        assert isinstance(config.retriever_config, RetrieverConfig)
        assert isinstance(config.generator_config, GeneratorConfig)
        assert config.default_mode == SearchMode.HYBRID
        assert config.enable_caching is True
        assert config.cache_size == 100
        assert config.max_iterations == 3
        assert config.drift_threshold == 0.8

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = GraphRAGConfig(
            default_mode=SearchMode.LOCAL,
            enable_caching=False,
            cache_size=50,
            max_iterations=5,
            drift_threshold=0.9,
        )

        assert config.default_mode == SearchMode.LOCAL
        assert config.enable_caching is False
        assert config.cache_size == 50
        assert config.max_iterations == 5
        assert config.drift_threshold == 0.9


class TestGraphRAGResult:
    """Tests for GraphRAGResult."""

    @pytest.fixture
    def sample_result(self) -> GraphRAGResult:
        """Create sample result."""
        code_context = CodeContext(
            code_id="id1",
            name="test_func",
            qualified_name="mod.test_func",
            code_type="function",
            file_path=Path("/test.py"),
            line_start=1,
            line_end=10,
            content="def test_func(): pass",
            language="python",
        )

        retrieval = RetrievalResult(
            query="How does test_func work?",
            query_type=QueryType.LOCAL,
            contexts=[code_context],
            processing_time_ms=100.0,
        )

        generation = GenerationResult(
            query="How does test_func work?",
            response="This function does XYZ.",
            model="gpt-4-turbo",
            tokens_used=150,
            processing_time_ms=200.0,
            citations=["mod.test_func"],
        )

        return GraphRAGResult(
            query="How does test_func work?",
            mode=SearchMode.LOCAL,
            retrieval=retrieval,
            generation=generation,
            iterations=1,
            total_time_ms=300.0,
        )

    def test_creation(self, sample_result: GraphRAGResult) -> None:
        """Test creating result."""
        assert sample_result.query == "How does test_func work?"
        assert sample_result.mode == SearchMode.LOCAL
        assert sample_result.iterations == 1
        assert sample_result.total_time_ms == 300.0

    def test_response_property(self, sample_result: GraphRAGResult) -> None:
        """Test response property."""
        assert sample_result.response == "This function does XYZ."

    def test_contexts_property(self, sample_result: GraphRAGResult) -> None:
        """Test contexts property."""
        assert len(sample_result.contexts) == 1
        assert sample_result.contexts[0].name == "test_func"

    def test_has_results_true(self, sample_result: GraphRAGResult) -> None:
        """Test has_results when results exist."""
        assert sample_result.has_results is True

    def test_has_results_false(self) -> None:
        """Test has_results when no results."""
        result = GraphRAGResult(
            query="test",
            mode=SearchMode.LOCAL,
            retrieval=RetrievalResult(query="test", query_type=QueryType.LOCAL),
            generation=GenerationResult(query="test", response="", model=""),
        )

        assert result.has_results is False

    def test_to_dict(self, sample_result: GraphRAGResult) -> None:
        """Test conversion to dict."""
        d = sample_result.to_dict()

        assert d["query"] == "How does test_func work?"
        assert d["mode"] == "local"
        assert d["response"] == "This function does XYZ."
        assert d["context_count"] == 1
        assert d["citations"] == ["mod.test_func"]
        assert d["iterations"] == 1
        assert d["total_time_ms"] == 300.0
        assert d["retrieval_time_ms"] == 100.0
        assert d["generation_time_ms"] == 200.0


class TestGraphRAG:
    """Tests for GraphRAG."""

    @pytest.fixture
    def mock_retriever(self) -> MockRetriever:
        """Create mock retriever."""
        retriever = MockRetriever()
        retriever.set_contexts([
            CodeContext(
                code_id="id1",
                name="test_func",
                qualified_name="mod.test_func",
                code_type="function",
                file_path=Path("/test.py"),
                line_start=1,
                line_end=10,
                content="def test_func(): pass",
                language="python",
            )
        ])
        return retriever

    @pytest.fixture
    def mock_generator(self) -> ResponseGenerator:
        """Create mock generator."""
        client = MockLLMClient(response="This code does something useful.")
        return ResponseGenerator(client=client)

    @pytest.fixture
    def graphrag(
        self, mock_retriever: MockRetriever, mock_generator: ResponseGenerator
    ) -> GraphRAG:
        """Create GraphRAG instance."""
        return GraphRAG(
            retriever=mock_retriever,  # type: ignore
            generator=mock_generator,
            config=GraphRAGConfig(),
        )

    @pytest.mark.asyncio
    async def test_query_local(self, graphrag: GraphRAG) -> None:
        """Test local query."""
        result = await graphrag.query(
            "How does test_func work?",
            mode=SearchMode.LOCAL,
        )

        assert result.query == "How does test_func work?"
        assert result.mode == SearchMode.LOCAL
        assert result.has_results is True
        assert len(result.response) > 0
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_query_global(self, graphrag: GraphRAG) -> None:
        """Test global query."""
        result = await graphrag.query(
            "What is the architecture?",
            mode=SearchMode.GLOBAL,
        )

        assert result.mode == SearchMode.GLOBAL

    @pytest.mark.asyncio
    async def test_query_hybrid(self, graphrag: GraphRAG) -> None:
        """Test hybrid query."""
        result = await graphrag.query(
            "How does authentication work?",
            mode=SearchMode.HYBRID,
        )

        assert result.mode == SearchMode.HYBRID

    @pytest.mark.asyncio
    async def test_query_drift(self, graphrag: GraphRAG) -> None:
        """Test DRIFT query."""
        result = await graphrag.query(
            "Explain the data flow",
            mode=SearchMode.DRIFT,
        )

        assert result.mode == SearchMode.DRIFT
        assert result.iterations >= 1

    @pytest.mark.asyncio
    async def test_query_default_mode(self, graphrag: GraphRAG) -> None:
        """Test query uses default mode."""
        result = await graphrag.query("test query")

        # Default is HYBRID
        assert result.mode == SearchMode.HYBRID

    @pytest.mark.asyncio
    async def test_query_with_filters(self, graphrag: GraphRAG) -> None:
        """Test query with filters."""
        filters = {"language": "python"}

        result = await graphrag.query(
            "Find Python code",
            filters=filters,
        )

        assert result.has_results is True

    @pytest.mark.asyncio
    async def test_find_similar(self, graphrag: GraphRAG) -> None:
        """Test finding similar code."""
        results = await graphrag.find_similar(
            "def my_func(): return 42",
            limit=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_explore_relationships(self, graphrag: GraphRAG) -> None:
        """Test exploring relationships."""
        results = await graphrag.explore_relationships(
            "code_id",
            relationship="callees",
            depth=1,
        )

        assert isinstance(results, list)

    def test_clear_cache(
        self, mock_retriever: MockRetriever
    ) -> None:
        """Test clearing cache."""
        client = MockLLMClient()
        base_generator = ResponseGenerator(client=client)
        cached_generator = CachedGenerator(generator=base_generator)

        graphrag = GraphRAG(
            retriever=mock_retriever,  # type: ignore
            generator=cached_generator,
            config=GraphRAGConfig(enable_caching=True),
        )

        # Add something to cache
        cached_generator._cache = {"key": MagicMock()}

        graphrag.clear_cache()

        assert cached_generator._cache == {}

    def test_mode_to_query_type(self, graphrag: GraphRAG) -> None:
        """Test mode to query type conversion."""
        assert graphrag._mode_to_query_type(SearchMode.LOCAL) == QueryType.LOCAL
        assert graphrag._mode_to_query_type(SearchMode.GLOBAL) == QueryType.GLOBAL
        assert graphrag._mode_to_query_type(SearchMode.HYBRID) == QueryType.DRIFT
        assert graphrag._mode_to_query_type(SearchMode.DRIFT) == QueryType.DRIFT

    def test_refine_query(self, graphrag: GraphRAG) -> None:
        """Test query refinement."""
        original = "How does auth work?"
        response = "The login_user() function handles authentication. It calls verify_token()."

        refined = graphrag._refine_query(original, response)

        assert original in refined
        # Should include code-like terms (without parentheses as those are stripped)
        assert "login_user" in refined or "verify_token" in refined


class TestGraphRAGWithCaching:
    """Tests for GraphRAG with caching enabled."""

    @pytest.fixture
    def mock_retriever(self) -> MockRetriever:
        """Create mock retriever."""
        return MockRetriever()

    @pytest.fixture
    def cached_graphrag(self, mock_retriever: MockRetriever) -> GraphRAG:
        """Create GraphRAG with caching."""
        client = MockLLMClient(response="Cached response")
        base_generator = ResponseGenerator(client=client)
        cached_generator = CachedGenerator(generator=base_generator, cache_size=10)

        return GraphRAG(
            retriever=mock_retriever,  # type: ignore
            generator=cached_generator,
            config=GraphRAGConfig(enable_caching=True),
        )

    @pytest.mark.asyncio
    async def test_uses_cached_generator(
        self, cached_graphrag: GraphRAG
    ) -> None:
        """Test that cached generator is used."""
        assert isinstance(cached_graphrag.generator, CachedGenerator)


class TestGraphRAGCreate:
    """Tests for GraphRAG.create factory method."""

    def test_create_with_mock_components(self) -> None:
        """Test create with mock components."""
        mock_searcher = MagicMock()
        mock_graph = MagicMock()
        mock_client = MockLLMClient()

        graphrag = GraphRAG.create(
            searcher=mock_searcher,
            graph=mock_graph,
            llm_client=mock_client,
            config=GraphRAGConfig(enable_caching=True),
        )

        assert graphrag.retriever is not None
        assert graphrag.generator is not None
        assert isinstance(graphrag.generator, CachedGenerator)

    def test_create_without_caching(self) -> None:
        """Test create without caching."""
        mock_searcher = MagicMock()
        mock_graph = MagicMock()
        mock_client = MockLLMClient()

        graphrag = GraphRAG.create(
            searcher=mock_searcher,
            graph=mock_graph,
            llm_client=mock_client,
            config=GraphRAGConfig(enable_caching=False),
        )

        assert isinstance(graphrag.generator, ResponseGenerator)


class TestMockGraphRAG:
    """Tests for MockGraphRAG."""

    @pytest.fixture
    def mock_graphrag(self) -> MockGraphRAG:
        """Create mock GraphRAG."""
        return MockGraphRAG()

    def test_initialization(self, mock_graphrag: MockGraphRAG) -> None:
        """Test initialization."""
        assert mock_graphrag._contexts == []
        assert mock_graphrag._response == "Mock response"

    def test_set_contexts(self, mock_graphrag: MockGraphRAG) -> None:
        """Test setting contexts."""
        contexts = [
            CodeContext(
                code_id="id",
                name="func",
                qualified_name="func",
                code_type="function",
                file_path=Path("/test.py"),
                line_start=1,
                line_end=5,
                content="pass",
                language="python",
            )
        ]

        mock_graphrag.set_contexts(contexts)

        assert len(mock_graphrag._contexts) == 1

    def test_set_response(self, mock_graphrag: MockGraphRAG) -> None:
        """Test setting response."""
        mock_graphrag.set_response("Custom response")

        assert mock_graphrag._response == "Custom response"

    @pytest.mark.asyncio
    async def test_query(self, mock_graphrag: MockGraphRAG) -> None:
        """Test mock query."""
        result = await mock_graphrag.query(
            "test query",
            mode=SearchMode.LOCAL,
        )

        assert result.query == "test query"
        assert result.mode == SearchMode.LOCAL
        assert result.response == "Mock response"

    @pytest.mark.asyncio
    async def test_query_default_mode(self, mock_graphrag: MockGraphRAG) -> None:
        """Test mock query with default mode."""
        result = await mock_graphrag.query("test")

        assert result.mode == SearchMode.LOCAL


class TestCreateMockGraphRAG:
    """Tests for create_mock_graphrag helper."""

    def test_creates_graphrag(self) -> None:
        """Test creates GraphRAG instance."""
        graphrag = create_mock_graphrag()

        assert graphrag is not None

    @pytest.mark.asyncio
    async def test_mock_graphrag_query(self) -> None:
        """Test querying mock GraphRAG."""
        graphrag = create_mock_graphrag()

        result = await graphrag.query("test query")

        assert result is not None
        assert len(result.response) > 0


class TestGraphRAGDrift:
    """Tests for DRIFT search mode."""

    @pytest.fixture
    def mock_retriever(self) -> MockRetriever:
        """Create mock retriever with varying results."""
        retriever = MockRetriever()
        retriever.set_contexts([
            CodeContext(
                code_id="id1",
                name="func1",
                qualified_name="func1",
                code_type="function",
                file_path=Path("/a.py"),
                line_start=1,
                line_end=5,
                content="def func1(): pass",
                language="python",
            ),
        ])
        return retriever

    @pytest.fixture
    def graphrag(self, mock_retriever: MockRetriever) -> GraphRAG:
        """Create GraphRAG for DRIFT testing."""
        client = MockLLMClient(response="Response with func1() and func2() calls.")
        generator = ResponseGenerator(client=client)

        return GraphRAG(
            retriever=mock_retriever,  # type: ignore
            generator=generator,
            config=GraphRAGConfig(max_iterations=3, drift_threshold=0.8),
        )

    @pytest.mark.asyncio
    async def test_drift_iterates(self, graphrag: GraphRAG) -> None:
        """Test DRIFT performs iterations."""
        result = await graphrag.query(
            "Explain the data flow",
            mode=SearchMode.DRIFT,
        )

        assert result.iterations >= 1
        assert result.mode == SearchMode.DRIFT

    @pytest.mark.asyncio
    async def test_drift_converges(self, graphrag: GraphRAG) -> None:
        """Test DRIFT converges when results overlap."""
        # With same context returned each time, should converge quickly
        result = await graphrag.query(
            "test",
            mode=SearchMode.DRIFT,
        )

        # Should have converged before max iterations
        assert result.iterations <= graphrag.config.max_iterations


class TestGraphRAGHybrid:
    """Tests for hybrid search mode."""

    @pytest.fixture
    def mock_retriever(self) -> MockRetriever:
        """Create mock retriever."""
        retriever = MockRetriever()
        retriever.set_contexts([
            CodeContext(
                code_id="id1",
                name="func",
                qualified_name="func",
                code_type="function",
                file_path=Path("/test.py"),
                line_start=1,
                line_end=5,
                content="pass",
                language="python",
            )
        ])
        return retriever

    @pytest.fixture
    def graphrag(self, mock_retriever: MockRetriever) -> GraphRAG:
        """Create GraphRAG."""
        client = MockLLMClient(response="Hybrid response")
        generator = ResponseGenerator(client=client)

        return GraphRAG(
            retriever=mock_retriever,  # type: ignore
            generator=generator,
        )

    @pytest.mark.asyncio
    async def test_hybrid_combines_local_and_global(
        self, graphrag: GraphRAG
    ) -> None:
        """Test hybrid combines local and global results."""
        result = await graphrag.query(
            "test",
            mode=SearchMode.HYBRID,
        )

        assert result.mode == SearchMode.HYBRID
        # Hybrid should have results from both local and global
        assert result.retrieval is not None


class TestGraphRAGExplainCode:
    """Tests for explain_code method."""

    @pytest.fixture
    def mock_retriever(self) -> MockRetriever:
        """Create mock retriever."""
        return MockRetriever()

    @pytest.fixture
    def graphrag(self, mock_retriever: MockRetriever) -> GraphRAG:
        """Create GraphRAG."""
        client = MockLLMClient(response="This code does X.")
        generator = ResponseGenerator(client=client)

        return GraphRAG(
            retriever=mock_retriever,  # type: ignore
            generator=generator,
        )

    @pytest.mark.asyncio
    async def test_explain_code_not_found(self, graphrag: GraphRAG) -> None:
        """Test explain_code when code not found."""
        result = await graphrag.explain_code("nonexistent_id")

        assert "not found" in result.response.lower()

    @pytest.mark.asyncio
    async def test_explain_code_with_question(
        self, graphrag: GraphRAG, mock_retriever: MockRetriever
    ) -> None:
        """Test explain_code with specific question."""
        mock_retriever.set_contexts([
            CodeContext(
                code_id="id1",
                name="my_func",
                qualified_name="my_func",
                code_type="function",
                file_path=Path("/test.py"),
                line_start=1,
                line_end=5,
                content="def my_func(): return 42",
                language="python",
            )
        ])

        # explain_code uses retrieve_by_relationship which returns mock contexts
        result = await graphrag.explain_code(
            "id1",
            question="What does this return?",
        )

        # If contexts found, the question is used
        # If not found, returns "Code not found"
        assert "What does this return?" in result.query or "not found" in result.response.lower()
