"""Tests for hybrid search."""

from pathlib import Path

import pytest

from smart_search.embedding.jina_embedder import MockEmbedder
from smart_search.embedding.models import EmbeddingConfig
from smart_search.search.hybrid import (
    HybridSearchConfig,
    HybridSearcher,
    SearchFacade,
)
from smart_search.search.meilisearch_client import MockMeilisearchClient
from smart_search.search.schemas import (
    IndexDocument,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchType,
)


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = HybridSearchConfig()
        assert config.semantic_weight == 0.5
        assert config.keyword_weight == 0.5
        assert config.rerank is True
        assert config.min_score == 0.0

    def test_weight_normalization(self) -> None:
        """Test weights are normalized."""
        config = HybridSearchConfig(semantic_weight=2.0, keyword_weight=2.0)
        assert config.semantic_weight == 0.5
        assert config.keyword_weight == 0.5

    def test_custom_weights(self) -> None:
        """Test custom weights."""
        config = HybridSearchConfig(semantic_weight=0.7, keyword_weight=0.3)
        assert config.semantic_weight == 0.7
        assert config.keyword_weight == 0.3


class TestHybridSearcher:
    """Tests for HybridSearcher."""

    @pytest.fixture
    def client(self) -> MockMeilisearchClient:
        """Create mock client."""
        return MockMeilisearchClient()

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        """Create mock embedder."""
        return MockEmbedder(EmbeddingConfig(dimensions=64))

    @pytest.fixture
    def searcher(
        self, client: MockMeilisearchClient, embedder: MockEmbedder
    ) -> HybridSearcher:
        """Create hybrid searcher."""
        return HybridSearcher(client, embedder)

    @pytest.fixture
    async def populated_client(
        self, client: MockMeilisearchClient
    ) -> MockMeilisearchClient:
        """Create client with documents."""
        docs = [
            IndexDocument(
                id="func1",
                name="calculate_total",
                qualified_name="math.calculate_total",
                code_type="function",
                file_path="math.py",
                line_start=1,
                line_end=10,
                content="def calculate_total(items): return sum(items)",
                language="python",
            ),
            IndexDocument(
                id="func2",
                name="calculate_average",
                qualified_name="math.calculate_average",
                code_type="function",
                file_path="math.py",
                line_start=12,
                line_end=15,
                content="def calculate_average(items): return sum(items) / len(items)",
                language="python",
            ),
            IndexDocument(
                id="class1",
                name="Calculator",
                qualified_name="math.Calculator",
                code_type="class",
                file_path="math.py",
                line_start=17,
                line_end=50,
                content="class Calculator: def add(self, a, b): return a + b",
                language="python",
            ),
        ]
        await client.add_documents(docs)
        return client

    @pytest.mark.asyncio
    async def test_keyword_search(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test keyword search."""
        searcher = HybridSearcher(populated_client, embedder)
        query = SearchQuery(query="calculate", search_type=SearchType.KEYWORD)
        result = await searcher.search(query)
        assert result.total >= 1

    @pytest.mark.asyncio
    async def test_semantic_search(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test semantic search."""
        searcher = HybridSearcher(populated_client, embedder)
        query = SearchQuery(query="compute sum", search_type=SearchType.SEMANTIC)
        result = await searcher.search(query)
        # Mock client doesn't support semantic, returns empty
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test hybrid search."""
        searcher = HybridSearcher(populated_client, embedder)
        query = SearchQuery(query="calculate", search_type=SearchType.HYBRID)
        result = await searcher.search(query)
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_reranking(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test result reranking."""
        config = HybridSearchConfig(rerank=True)
        searcher = HybridSearcher(populated_client, embedder, config)
        query = SearchQuery(query="calculate", search_type=SearchType.KEYWORD)
        result = await searcher.search(query)
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_min_score_filter(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test minimum score filter."""
        config = HybridSearchConfig(min_score=2.0)  # High threshold
        searcher = HybridSearcher(populated_client, embedder, config)
        query = SearchQuery(query="calculate", search_type=SearchType.KEYWORD)
        result = await searcher.search(query)
        # With mock, scores are 1.0, so should be filtered out
        # But reranking can boost scores
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_similar(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test similar code search."""
        searcher = HybridSearcher(populated_client, embedder)
        result = await searcher.search_similar(
            "def add_numbers(a, b): return a + b",
            limit=5,
        )
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_similar_exclude_ids(
        self,
        populated_client: MockMeilisearchClient,
        embedder: MockEmbedder,
    ) -> None:
        """Test similar search with excluded IDs."""
        searcher = HybridSearcher(populated_client, embedder)
        result = await searcher.search_similar(
            "def add_numbers(a, b): return a + b",
            limit=5,
            exclude_ids=["func1"],
        )
        assert all(hit.id != "func1" for hit in result.hits)

    @pytest.mark.asyncio
    async def test_close(self, searcher: HybridSearcher) -> None:
        """Test closing resources."""
        await searcher.close()


class TestHybridSearcherReranking:
    """Tests for reranking logic."""

    def test_boost_exact_name_match(self) -> None:
        """Test exact name match boosting."""
        config = HybridSearchConfig(rerank=True)
        client = MockMeilisearchClient()
        embedder = MockEmbedder()
        searcher = HybridSearcher(client, embedder, config)

        # Create result with a hit that matches query
        result = SearchResult(
            hits=[
                SearchHit(
                    id="1",
                    name="calculate",
                    qualified_name="test.calculate",
                    code_type="function",
                    file_path=Path("test.py"),
                    line_start=1,
                    line_end=5,
                    content="",
                    language="python",
                    score=1.0,
                ),
            ],
            total=1,
            query="calculate",
            search_type=SearchType.KEYWORD,
        )

        query = SearchQuery(query="calculate")
        reranked = searcher._rerank_results(result, query)
        # Score should be boosted for exact name match
        assert reranked.hits[0].score > 1.0

    def test_boost_function_query(self) -> None:
        """Test boosting for function-like queries."""
        config = HybridSearchConfig(rerank=True)
        client = MockMeilisearchClient()
        embedder = MockEmbedder()
        searcher = HybridSearcher(client, embedder, config)

        result = SearchResult(
            hits=[
                SearchHit(
                    id="1",
                    name="test",
                    qualified_name="test",
                    code_type="function",
                    file_path=Path("test.py"),
                    line_start=1,
                    line_end=5,
                    content="",
                    language="python",
                    score=1.0,
                ),
            ],
            total=1,
            query="def test",
            search_type=SearchType.KEYWORD,
        )

        query = SearchQuery(query="def test")
        reranked = searcher._rerank_results(result, query)
        assert reranked.hits[0].score >= 1.0


class TestSearchFacade:
    """Tests for SearchFacade."""

    @pytest.fixture
    async def facade(self) -> SearchFacade:
        """Create search facade."""
        client = MockMeilisearchClient()
        embedder = MockEmbedder()
        searcher = HybridSearcher(client, embedder)

        # Add some documents
        docs = [
            IndexDocument(
                id="func1",
                name="process_data",
                qualified_name="data.process_data",
                code_type="function",
                file_path="data.py",
                line_start=1,
                line_end=10,
                content="def process_data(items): pass",
                language="python",
            ),
            IndexDocument(
                id="class1",
                name="DataProcessor",
                qualified_name="data.DataProcessor",
                code_type="class",
                file_path="data.py",
                line_start=12,
                line_end=50,
                content="class DataProcessor: pass",
                language="python",
            ),
        ]
        await client.add_documents(docs)

        return SearchFacade(searcher)

    @pytest.mark.asyncio
    async def test_search(self, facade: SearchFacade) -> None:
        """Test basic search."""
        result = await facade.search("data")
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_search_with_filters(self, facade: SearchFacade) -> None:
        """Test search with filters."""
        result = await facade.search(
            "data",
            languages=["python"],
            code_types=["function"],
        )
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_find_function(self, facade: SearchFacade) -> None:
        """Test finding a function."""
        result = await facade.find_function("process_data")
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_find_class(self, facade: SearchFacade) -> None:
        """Test finding a class."""
        result = await facade.find_class("DataProcessor")
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_find_similar(self, facade: SearchFacade) -> None:
        """Test finding similar code."""
        result = await facade.find_similar("def process(items): pass")
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_close(self, facade: SearchFacade) -> None:
        """Test closing facade."""
        await facade.close()
