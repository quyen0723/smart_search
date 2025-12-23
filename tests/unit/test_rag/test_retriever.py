"""Tests for RAG retriever module."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smart_search.rag.prompts import (
    CodeContext,
    CommunitySummary,
    QueryType,
    RelationshipPath,
)
from smart_search.rag.retriever import (
    ContextRetriever,
    MockRetriever,
    RetrievalResult,
    RetrieverConfig,
)


class TestRetrieverConfig:
    """Tests for RetrieverConfig."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = RetrieverConfig()

        assert config.max_vector_results == 20
        assert config.max_graph_depth == 2
        assert config.max_context_items == 10
        assert config.include_callers is True
        assert config.include_callees is True
        assert config.include_siblings is False
        assert config.semantic_threshold == 0.5

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = RetrieverConfig(
            max_vector_results=50,
            max_graph_depth=3,
            max_context_items=20,
            include_callers=False,
            include_callees=True,
            include_siblings=True,
            semantic_threshold=0.7,
        )

        assert config.max_vector_results == 50
        assert config.max_graph_depth == 3
        assert config.max_context_items == 20
        assert config.include_callers is False
        assert config.include_siblings is True
        assert config.semantic_threshold == 0.7


class TestRetrievalResult:
    """Tests for RetrievalResult."""

    @pytest.fixture
    def sample_context(self) -> CodeContext:
        """Create sample code context."""
        return CodeContext(
            code_id="test_id",
            name="test_func",
            qualified_name="mod.test_func",
            code_type="function",
            file_path=Path("/test.py"),
            line_start=1,
            line_end=10,
            content="def test_func(): pass",
            language="python",
        )

    def test_creation(self, sample_context: CodeContext) -> None:
        """Test creating retrieval result."""
        result = RetrievalResult(
            query="How does test_func work?",
            query_type=QueryType.LOCAL,
            contexts=[sample_context],
        )

        assert result.query == "How does test_func work?"
        assert result.query_type == QueryType.LOCAL
        assert len(result.contexts) == 1
        assert result.processing_time_ms == 0.0

    def test_default_values(self) -> None:
        """Test default values."""
        result = RetrievalResult(
            query="test",
            query_type=QueryType.LOCAL,
        )

        assert result.contexts == []
        assert result.relationships == []
        assert result.communities == []
        assert result.processing_time_ms == 0.0

    def test_has_results_true(self, sample_context: CodeContext) -> None:
        """Test has_results when results exist."""
        result = RetrievalResult(
            query="test",
            query_type=QueryType.LOCAL,
            contexts=[sample_context],
        )

        assert result.has_results is True

    def test_has_results_false(self) -> None:
        """Test has_results when no results."""
        result = RetrievalResult(
            query="test",
            query_type=QueryType.LOCAL,
        )

        assert result.has_results is False

    def test_has_results_with_relationships(self) -> None:
        """Test has_results with relationships only."""
        result = RetrievalResult(
            query="test",
            query_type=QueryType.LOCAL,
            relationships=[
                RelationshipPath(source="a", target="b", path=[])
            ],
        )

        assert result.has_results is True

    def test_has_results_with_communities(self) -> None:
        """Test has_results with communities only."""
        result = RetrievalResult(
            query="test",
            query_type=QueryType.GLOBAL,
            communities=[
                CommunitySummary(
                    community_id="c1",
                    name="Test",
                    description="Test community",
                )
            ],
        )

        assert result.has_results is True

    def test_to_prompt_context(self, sample_context: CodeContext) -> None:
        """Test conversion to PromptContext."""
        relationship = RelationshipPath(
            source="a",
            target="b",
            path=[("a", "CALLS")],
        )
        community = CommunitySummary(
            community_id="c1",
            name="Test",
            description="Test",
        )

        result = RetrievalResult(
            query="How does it work?",
            query_type=QueryType.DRIFT,
            contexts=[sample_context],
            relationships=[relationship],
            communities=[community],
        )

        prompt_context = result.to_prompt_context(max_tokens=2000)

        assert prompt_context.query == "How does it work?"
        assert prompt_context.query_type == QueryType.DRIFT
        assert len(prompt_context.code_contexts) == 1
        assert len(prompt_context.relationships) == 1
        assert len(prompt_context.communities) == 1
        assert prompt_context.max_tokens == 2000


class TestContextRetriever:
    """Tests for ContextRetriever."""

    @pytest.fixture
    def mock_searcher(self) -> MagicMock:
        """Create mock searcher."""
        searcher = MagicMock()
        searcher.search = AsyncMock()
        searcher.search_similar = AsyncMock()
        return searcher

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        """Create mock graph."""
        graph = MagicMock()
        graph.get_callers = MagicMock(return_value=[])
        graph.get_callees = MagicMock(return_value=[])
        graph.get_ancestors = MagicMock(return_value=[])
        graph.get_descendants = MagicMock(return_value=[])
        graph.get_children = MagicMock(return_value=[])
        graph.get_node = MagicMock(return_value=None)
        graph.get_node_community = MagicMock(return_value=None)
        graph.get_community_members = MagicMock(return_value=[])
        return graph

    @pytest.fixture
    def retriever(
        self, mock_searcher: MagicMock, mock_graph: MagicMock
    ) -> ContextRetriever:
        """Create context retriever."""
        return ContextRetriever(
            searcher=mock_searcher,
            graph=mock_graph,
            config=RetrieverConfig(),
        )

    def test_initialization(
        self, retriever: ContextRetriever, mock_searcher: MagicMock
    ) -> None:
        """Test retriever initialization."""
        assert retriever.searcher == mock_searcher
        assert retriever.config is not None
        assert retriever._community_cache == {}

    @pytest.mark.asyncio
    async def test_retrieve_local(
        self, retriever: ContextRetriever, mock_searcher: MagicMock
    ) -> None:
        """Test local retrieval."""
        # Setup mock search result
        mock_hit = MagicMock()
        mock_hit.id = "test_id"
        mock_hit.name = "test_func"
        mock_hit.qualified_name = "mod.test_func"
        mock_hit.code_type = "function"
        mock_hit.file_path = "/test.py"
        mock_hit.line_start = 1
        mock_hit.line_end = 10
        mock_hit.content = "def test_func(): pass"
        mock_hit.language = "python"
        mock_hit.score = 0.9
        mock_hit.metadata = {"docstring": "A test function"}

        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_searcher.search.return_value = mock_result

        result = await retriever.retrieve(
            "How does test_func work?",
            QueryType.LOCAL,
        )

        assert result.query == "How does test_func work?"
        assert result.query_type == QueryType.LOCAL
        assert len(result.contexts) == 1
        assert result.contexts[0].name == "test_func"
        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_retrieve_global(
        self, retriever: ContextRetriever, mock_searcher: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test global retrieval."""
        # Setup mock
        mock_result = MagicMock()
        mock_result.hits = []
        mock_searcher.search.return_value = mock_result

        result = await retriever.retrieve(
            "What is the architecture?",
            QueryType.GLOBAL,
        )

        assert result.query_type == QueryType.GLOBAL
        # Communities should be retrieved for global queries
        assert isinstance(result.communities, list)

    @pytest.mark.asyncio
    async def test_retrieve_drift(
        self, retriever: ContextRetriever, mock_searcher: MagicMock
    ) -> None:
        """Test drift retrieval."""
        mock_result = MagicMock()
        mock_result.hits = []
        mock_searcher.search.return_value = mock_result

        result = await retriever.retrieve(
            "How does authentication work?",
            QueryType.DRIFT,
        )

        assert result.query_type == QueryType.DRIFT
        # DRIFT includes both relationships and communities
        assert isinstance(result.relationships, list)
        assert isinstance(result.communities, list)

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(
        self, retriever: ContextRetriever, mock_searcher: MagicMock
    ) -> None:
        """Test retrieval with filters."""
        mock_result = MagicMock()
        mock_result.hits = []
        mock_searcher.search.return_value = mock_result

        filters = {"language": "python", "file_path": "/src"}

        await retriever.retrieve("test", QueryType.LOCAL, filters=filters)

        # Verify search was called
        mock_searcher.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_similar(
        self, retriever: ContextRetriever, mock_searcher: MagicMock
    ) -> None:
        """Test similar code retrieval."""
        mock_hit = MagicMock()
        mock_hit.id = "similar_id"
        mock_hit.name = "similar_func"
        mock_hit.qualified_name = "similar_func"
        mock_hit.code_type = "function"
        mock_hit.file_path = "/similar.py"
        mock_hit.line_start = 1
        mock_hit.line_end = 5
        mock_hit.content = "def similar_func(): pass"
        mock_hit.language = "python"
        mock_hit.score = 0.85
        mock_hit.metadata = {}

        mock_result = MagicMock()
        mock_result.hits = [mock_hit]
        mock_searcher.search_similar.return_value = mock_result

        results = await retriever.retrieve_similar(
            "def my_func(): return 42",
            limit=5,
        )

        assert len(results) == 1
        assert results[0].name == "similar_func"
        mock_searcher.search_similar.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_by_relationship_callers(
        self, retriever: ContextRetriever, mock_graph: MagicMock
    ) -> None:
        """Test retrieving by caller relationship."""
        mock_node = MagicMock()
        mock_node.id = "caller_id"
        mock_node.name = "caller_func"
        mock_node.qualified_name = "caller_func"
        mock_node.type = MagicMock(value="function")
        mock_node.file_path = "/caller.py"
        mock_node.line_start = 1
        mock_node.line_end = 5
        mock_node.content = "def caller_func(): pass"
        mock_node.language = "python"

        mock_graph.get_callers.return_value = ["caller_id"]
        mock_graph.get_node.return_value = mock_node

        results = await retriever.retrieve_by_relationship(
            "target_id",
            "callers",
            depth=1,
        )

        assert len(results) == 1
        assert results[0].name == "caller_func"
        mock_graph.get_callers.assert_called_with("target_id", 1)

    @pytest.mark.asyncio
    async def test_retrieve_by_relationship_callees(
        self, retriever: ContextRetriever, mock_graph: MagicMock
    ) -> None:
        """Test retrieving by callee relationship."""
        mock_graph.get_callees.return_value = []

        results = await retriever.retrieve_by_relationship(
            "source_id",
            "callees",
            depth=2,
        )

        mock_graph.get_callees.assert_called_with("source_id", 2)

    @pytest.mark.asyncio
    async def test_retrieve_by_relationship_ancestors(
        self, retriever: ContextRetriever, mock_graph: MagicMock
    ) -> None:
        """Test retrieving by ancestor relationship."""
        mock_graph.get_ancestors.return_value = []

        await retriever.retrieve_by_relationship(
            "node_id",
            "ancestors",
            depth=3,
        )

        mock_graph.get_ancestors.assert_called_with("node_id", 3)

    @pytest.mark.asyncio
    async def test_retrieve_by_relationship_descendants(
        self, retriever: ContextRetriever, mock_graph: MagicMock
    ) -> None:
        """Test retrieving by descendant relationship."""
        mock_graph.get_descendants.return_value = []

        await retriever.retrieve_by_relationship(
            "node_id",
            "descendants",
            depth=2,
        )

        mock_graph.get_descendants.assert_called_with("node_id", 2)

    def test_search_to_contexts(self, retriever: ContextRetriever) -> None:
        """Test converting search results to contexts."""
        mock_hit = MagicMock()
        mock_hit.id = "id1"
        mock_hit.name = "func"
        mock_hit.qualified_name = "mod.func"
        mock_hit.code_type = "function"
        mock_hit.file_path = "/test.py"
        mock_hit.line_start = 1
        mock_hit.line_end = 10
        mock_hit.content = "def func(): pass"
        mock_hit.language = "python"
        mock_hit.score = 0.8
        mock_hit.metadata = {"docstring": "A function"}

        mock_result = MagicMock()
        mock_result.hits = [mock_hit]

        contexts = retriever._search_to_contexts(mock_result)

        assert len(contexts) == 1
        assert contexts[0].code_id == "id1"
        assert contexts[0].name == "func"
        assert contexts[0].relevance_score == 0.8

    def test_search_to_contexts_filters_low_score(
        self, retriever: ContextRetriever
    ) -> None:
        """Test that low score results are filtered."""
        mock_hit = MagicMock()
        mock_hit.score = 0.3  # Below threshold of 0.5

        mock_result = MagicMock()
        mock_result.hits = [mock_hit]

        contexts = retriever._search_to_contexts(mock_result)

        assert len(contexts) == 0

    def test_get_siblings(
        self, retriever: ContextRetriever, mock_graph: MagicMock
    ) -> None:
        """Test getting sibling nodes."""
        mock_node = MagicMock()
        mock_node.parent_id = "parent_id"

        mock_graph.get_node.return_value = mock_node
        mock_graph.get_children.return_value = ["sibling1", "sibling2", "self_id"]

        siblings = retriever._get_siblings("self_id")

        assert "sibling1" in siblings
        assert "sibling2" in siblings
        assert "self_id" not in siblings  # Self excluded

    def test_get_siblings_no_parent(
        self, retriever: ContextRetriever, mock_graph: MagicMock
    ) -> None:
        """Test getting siblings when no parent."""
        mock_node = MagicMock()
        mock_node.parent_id = None

        mock_graph.get_node.return_value = mock_node

        siblings = retriever._get_siblings("orphan_id")

        assert siblings == []

    def test_clear_community_cache(self, retriever: ContextRetriever) -> None:
        """Test clearing community cache."""
        retriever._community_cache = {"c1": MagicMock()}

        retriever.clear_community_cache()

        assert retriever._community_cache == {}


class TestMockRetriever:
    """Tests for MockRetriever."""

    @pytest.fixture
    def mock_retriever(self) -> MockRetriever:
        """Create mock retriever."""
        return MockRetriever()

    @pytest.fixture
    def sample_context(self) -> CodeContext:
        """Create sample context."""
        return CodeContext(
            code_id="id",
            name="func",
            qualified_name="func",
            code_type="function",
            file_path=Path("/test.py"),
            line_start=1,
            line_end=5,
            content="def func(): pass",
            language="python",
        )

    def test_initialization(self, mock_retriever: MockRetriever) -> None:
        """Test mock retriever initialization."""
        assert mock_retriever._contexts == []
        assert mock_retriever._relationships == []
        assert mock_retriever._communities == []

    def test_set_contexts(
        self, mock_retriever: MockRetriever, sample_context: CodeContext
    ) -> None:
        """Test setting mock contexts."""
        mock_retriever.set_contexts([sample_context])

        assert len(mock_retriever._contexts) == 1
        assert mock_retriever._contexts[0] == sample_context

    def test_set_relationships(self, mock_retriever: MockRetriever) -> None:
        """Test setting mock relationships."""
        relationships = [
            RelationshipPath(source="a", target="b", path=[])
        ]

        mock_retriever.set_relationships(relationships)

        assert len(mock_retriever._relationships) == 1

    def test_set_communities(self, mock_retriever: MockRetriever) -> None:
        """Test setting mock communities."""
        communities = [
            CommunitySummary(community_id="c1", name="Test", description="Test")
        ]

        mock_retriever.set_communities(communities)

        assert len(mock_retriever._communities) == 1

    @pytest.mark.asyncio
    async def test_retrieve(
        self, mock_retriever: MockRetriever, sample_context: CodeContext
    ) -> None:
        """Test mock retrieve."""
        mock_retriever.set_contexts([sample_context])

        result = await mock_retriever.retrieve(
            "test query",
            QueryType.LOCAL,
        )

        assert result.query == "test query"
        assert result.query_type == QueryType.LOCAL
        assert len(result.contexts) == 1
        assert result.processing_time_ms == 1.0

    @pytest.mark.asyncio
    async def test_retrieve_similar(
        self, mock_retriever: MockRetriever, sample_context: CodeContext
    ) -> None:
        """Test mock retrieve_similar."""
        mock_retriever.set_contexts([sample_context])

        results = await mock_retriever.retrieve_similar(
            "code content",
            limit=5,
        )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_similar_respects_limit(
        self, mock_retriever: MockRetriever
    ) -> None:
        """Test retrieve_similar respects limit."""
        contexts = [
            CodeContext(
                code_id=f"id{i}",
                name=f"func{i}",
                qualified_name=f"func{i}",
                code_type="function",
                file_path=Path(f"/test{i}.py"),
                line_start=1,
                line_end=5,
                content="pass",
                language="python",
            )
            for i in range(10)
        ]

        mock_retriever.set_contexts(contexts)

        results = await mock_retriever.retrieve_similar("code", limit=3)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_by_relationship(
        self, mock_retriever: MockRetriever, sample_context: CodeContext
    ) -> None:
        """Test mock retrieve_by_relationship."""
        mock_retriever.set_contexts([sample_context])

        results = await mock_retriever.retrieve_by_relationship(
            "code_id",
            "callers",
            depth=1,
        )

        assert len(results) == 1


class TestContextRetrieverExpansion:
    """Tests for context expansion in retriever."""

    @pytest.fixture
    def mock_searcher(self) -> MagicMock:
        """Create mock searcher."""
        searcher = MagicMock()
        searcher.search = AsyncMock()
        return searcher

    @pytest.fixture
    def mock_graph(self) -> MagicMock:
        """Create mock graph."""
        graph = MagicMock()
        return graph

    @pytest.mark.asyncio
    async def test_expand_local_with_callers(
        self, mock_searcher: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test local expansion includes callers."""
        config = RetrieverConfig(include_callers=True, include_callees=False)
        retriever = ContextRetriever(mock_searcher, mock_graph, config)

        mock_graph.get_callers.return_value = ["caller1", "caller2"]

        contexts = [
            CodeContext(
                code_id="target",
                name="target_func",
                qualified_name="target_func",
                code_type="function",
                file_path=Path("/test.py"),
                line_start=1,
                line_end=5,
                content="",
                language="python",
            )
        ]

        relationships = await retriever._expand_local(contexts)

        assert len(relationships) == 2
        mock_graph.get_callers.assert_called()

    @pytest.mark.asyncio
    async def test_expand_local_with_callees(
        self, mock_searcher: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test local expansion includes callees."""
        config = RetrieverConfig(include_callers=False, include_callees=True)
        retriever = ContextRetriever(mock_searcher, mock_graph, config)

        mock_graph.get_callees.return_value = ["callee1"]

        contexts = [
            CodeContext(
                code_id="source",
                name="source_func",
                qualified_name="source_func",
                code_type="function",
                file_path=Path("/test.py"),
                line_start=1,
                line_end=5,
                content="",
                language="python",
            )
        ]

        relationships = await retriever._expand_local(contexts)

        assert len(relationships) == 1
        mock_graph.get_callees.assert_called()

    @pytest.mark.asyncio
    async def test_expand_local_deduplicates(
        self, mock_searcher: MagicMock, mock_graph: MagicMock
    ) -> None:
        """Test expansion deduplicates relationships."""
        config = RetrieverConfig(include_callers=True, include_callees=True)
        retriever = ContextRetriever(mock_searcher, mock_graph, config)

        # Same relationship from both directions
        mock_graph.get_callers.return_value = []
        mock_graph.get_callees.return_value = ["common"]

        contexts = [
            CodeContext(
                code_id="a",
                name="a",
                qualified_name="a",
                code_type="function",
                file_path=Path("/a.py"),
                line_start=1,
                line_end=5,
                content="",
                language="python",
            ),
            CodeContext(
                code_id="b",
                name="b",
                qualified_name="b",
                code_type="function",
                file_path=Path("/b.py"),
                line_start=1,
                line_end=5,
                content="",
                language="python",
            ),
        ]

        relationships = await retriever._expand_local(contexts)

        # Should have deduplicated
        keys = {(r.source, r.target) for r in relationships}
        assert len(keys) == len(relationships)
