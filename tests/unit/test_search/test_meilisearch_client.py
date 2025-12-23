"""Tests for Meilisearch client."""

from pathlib import Path

import pytest

from smart_search.search.meilisearch_client import (
    MeilisearchClient,
    MockMeilisearchClient,
)
from smart_search.search.schemas import (
    IndexDocument,
    SearchQuery,
    SearchResult,
    SearchType,
)


class TestMockMeilisearchClient:
    """Tests for MockMeilisearchClient."""

    @pytest.fixture
    def client(self) -> MockMeilisearchClient:
        """Create mock client."""
        return MockMeilisearchClient()

    @pytest.fixture
    def sample_docs(self) -> list[IndexDocument]:
        """Create sample documents."""
        return [
            IndexDocument(
                id="test::func1",
                name="calculate_sum",
                qualified_name="test.calculate_sum",
                code_type="function",
                file_path="test.py",
                line_start=1,
                line_end=5,
                content="def calculate_sum(a, b): return a + b",
                language="python",
            ),
            IndexDocument(
                id="test::func2",
                name="calculate_diff",
                qualified_name="test.calculate_diff",
                code_type="function",
                file_path="test.py",
                line_start=7,
                line_end=10,
                content="def calculate_diff(a, b): return a - b",
                language="python",
            ),
            IndexDocument(
                id="test::class1",
                name="Calculator",
                qualified_name="test.Calculator",
                code_type="class",
                file_path="test.py",
                line_start=12,
                line_end=30,
                content="class Calculator: pass",
                language="python",
            ),
        ]

    @pytest.mark.asyncio
    async def test_initialize_index(self, client: MockMeilisearchClient) -> None:
        """Test index initialization."""
        await client.initialize_index()
        assert client._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_recreate(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test index recreation."""
        await client.add_documents(sample_docs)
        assert len(client._documents) == 3

        await client.initialize_index(recreate=True)
        assert len(client._documents) == 0

    @pytest.mark.asyncio
    async def test_add_documents(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test adding documents."""
        count = await client.add_documents(sample_docs)
        assert count == 3

        stats = await client.get_stats()
        assert stats.total_documents == 3

    @pytest.mark.asyncio
    async def test_update_documents(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test updating documents."""
        await client.add_documents(sample_docs)

        # Update a document
        updated = IndexDocument(
            id="test::func1",
            name="calculate_sum_updated",
            qualified_name="test.calculate_sum",
            code_type="function",
            file_path="test.py",
            line_start=1,
            line_end=5,
            content="def calculate_sum(a, b, c): return a + b + c",
            language="python",
        )
        count = await client.update_documents([updated])
        assert count == 1

        doc = await client.get_document("test::func1")
        assert doc is not None
        assert doc["name"] == "calculate_sum_updated"

    @pytest.mark.asyncio
    async def test_delete_documents(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test deleting documents."""
        await client.add_documents(sample_docs)
        count = await client.delete_documents(["test::func1", "test::func2"])
        assert count == 2

        stats = await client.get_stats()
        assert stats.total_documents == 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, client: MockMeilisearchClient) -> None:
        """Test deleting nonexistent documents."""
        count = await client.delete_documents(["nonexistent"])
        assert count == 0

    @pytest.mark.asyncio
    async def test_search_keyword(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test keyword search."""
        await client.add_documents(sample_docs)

        query = SearchQuery(query="calculate", search_type=SearchType.KEYWORD)
        result = await client.search(query)

        assert result.total == 2
        assert all("calculate" in hit.name.lower() for hit in result.hits)

    @pytest.mark.asyncio
    async def test_search_case_insensitive(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test case-insensitive search."""
        await client.add_documents(sample_docs)

        query = SearchQuery(query="CALCULATE", search_type=SearchType.KEYWORD)
        result = await client.search(query)
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_search_pagination(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test search pagination."""
        await client.add_documents(sample_docs)

        query = SearchQuery(query="test", limit=1, offset=0)
        result = await client.search(query)
        assert len(result.hits) <= 1

    @pytest.mark.asyncio
    async def test_search_no_results(self, client: MockMeilisearchClient) -> None:
        """Test search with no results."""
        query = SearchQuery(query="nonexistent")
        result = await client.search(query)
        assert result.total == 0
        assert len(result.hits) == 0

    @pytest.mark.asyncio
    async def test_get_document(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test getting a document."""
        await client.add_documents(sample_docs)

        doc = await client.get_document("test::func1")
        assert doc is not None
        assert doc["name"] == "calculate_sum"

    @pytest.mark.asyncio
    async def test_get_document_not_found(
        self, client: MockMeilisearchClient
    ) -> None:
        """Test getting nonexistent document."""
        doc = await client.get_document("nonexistent")
        assert doc is None

    @pytest.mark.asyncio
    async def test_health_check(self, client: MockMeilisearchClient) -> None:
        """Test health check."""
        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with MockMeilisearchClient() as client:
            await client.add_documents(
                [
                    IndexDocument(
                        id="test",
                        name="test",
                        qualified_name="test",
                        code_type="function",
                        file_path="test.py",
                        line_start=1,
                        line_end=5,
                        content="test",
                        language="python",
                    )
                ]
            )
            stats = await client.get_stats()
            assert stats.total_documents == 1


class TestMeilisearchClientInit:
    """Tests for MeilisearchClient initialization."""

    def test_default_index_name(self) -> None:
        """Test default index name."""
        client = MeilisearchClient("http://localhost:7700")
        assert client.index_name == "code_units"

    def test_custom_index_name(self) -> None:
        """Test custom index name."""
        client = MeilisearchClient(
            "http://localhost:7700", index_name="custom_index"
        )
        assert client.index_name == "custom_index"

    def test_not_initialized(self) -> None:
        """Test client is not initialized initially."""
        client = MeilisearchClient("http://localhost:7700")
        assert client._initialized is False
        assert client._client is None

    def test_with_api_key(self) -> None:
        """Test client with API key."""
        client = MeilisearchClient(
            "http://localhost:7700",
            api_key="test_key",
        )
        assert client.api_key == "test_key"

    def test_custom_embedding_dimensions(self) -> None:
        """Test custom embedding dimensions."""
        client = MeilisearchClient(
            "http://localhost:7700",
            embedding_dimensions=512,
        )
        assert client.embedding_dimensions == 512


class TestMockMeilisearchClientAdvanced:
    """Advanced tests for MockMeilisearchClient."""

    @pytest.fixture
    def client(self) -> MockMeilisearchClient:
        """Create mock client."""
        return MockMeilisearchClient()

    @pytest.fixture
    def sample_docs(self) -> list[IndexDocument]:
        """Create sample documents."""
        return [
            IndexDocument(
                id="test::func1",
                name="process_data",
                qualified_name="test.process_data",
                code_type="function",
                file_path="test.py",
                line_start=1,
                line_end=5,
                content="def process_data(items): return items",
                language="python",
            ),
            IndexDocument(
                id="test::func2",
                name="filter_data",
                qualified_name="test.filter_data",
                code_type="function",
                file_path="test.py",
                line_start=7,
                line_end=10,
                content="def filter_data(items): return [x for x in items]",
                language="python",
            ),
        ]

    @pytest.mark.asyncio
    async def test_search_by_content(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test search matches content."""
        await client.add_documents(sample_docs)
        query = SearchQuery(query="items", search_type=SearchType.KEYWORD)
        result = await client.search(query)
        assert result.total == 2

    @pytest.mark.asyncio
    async def test_search_hybrid_type(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test hybrid search type."""
        await client.add_documents(sample_docs)
        query = SearchQuery(query="process", search_type=SearchType.HYBRID)
        result = await client.search(query)
        assert result.search_type == SearchType.HYBRID

    @pytest.mark.asyncio
    async def test_search_semantic_type(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test semantic search type."""
        await client.add_documents(sample_docs)
        query = SearchQuery(query="process", search_type=SearchType.SEMANTIC)
        result = await client.search(query)
        assert result.search_type == SearchType.SEMANTIC

    @pytest.mark.asyncio
    async def test_delete_by_filter(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test delete by filter (no-op in mock)."""
        await client.add_documents(sample_docs)
        await client.delete_by_filter('file_path = "test.py"')
        # Mock doesn't actually filter, just ensure no error

    @pytest.mark.asyncio
    async def test_add_empty_documents(self, client: MockMeilisearchClient) -> None:
        """Test adding empty document list."""
        count = await client.add_documents([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_update_empty_documents(self, client: MockMeilisearchClient) -> None:
        """Test updating empty document list."""
        count = await client.update_documents([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_delete_empty_list(self, client: MockMeilisearchClient) -> None:
        """Test deleting empty ID list."""
        count = await client.delete_documents([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_search_with_embedding(
        self, client: MockMeilisearchClient, sample_docs: list[IndexDocument]
    ) -> None:
        """Test search with embedding."""
        await client.add_documents(sample_docs)
        query = SearchQuery(query="process", search_type=SearchType.HYBRID)
        embedding = [0.1] * 64
        result = await client.search(query, embedding=embedding)
        assert isinstance(result, SearchResult)

    @pytest.mark.asyncio
    async def test_close_multiple_times(self, client: MockMeilisearchClient) -> None:
        """Test closing client multiple times."""
        await client.close()
        await client.close()  # Should not raise
