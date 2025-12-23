"""Tests for search schemas."""

from pathlib import Path

import pytest

from smart_search.search.schemas import (
    IndexDocument,
    IndexStats,
    SearchFilter,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchType,
    SortField,
    SortOrder,
)


class TestSearchFilter:
    """Tests for SearchFilter."""

    def test_empty_filter(self) -> None:
        """Test empty filter."""
        filter_ = SearchFilter()
        assert filter_.to_meilisearch_filter() is None

    def test_language_filter(self) -> None:
        """Test language filter."""
        filter_ = SearchFilter(languages=["python", "javascript"])
        result = filter_.to_meilisearch_filter()
        assert result is not None
        assert 'language = "python"' in result
        assert 'language = "javascript"' in result
        assert "OR" in result

    def test_code_type_filter(self) -> None:
        """Test code type filter."""
        filter_ = SearchFilter(code_types=["function", "class"])
        result = filter_.to_meilisearch_filter()
        assert result is not None
        assert 'code_type = "function"' in result
        assert 'code_type = "class"' in result

    def test_line_range_filter(self) -> None:
        """Test line range filter."""
        filter_ = SearchFilter(min_line=10, max_line=100)
        result = filter_.to_meilisearch_filter()
        assert result is not None
        assert "line_start >= 10" in result
        assert "line_end <= 100" in result

    def test_combined_filters(self) -> None:
        """Test combined filters."""
        filter_ = SearchFilter(
            languages=["python"],
            code_types=["function"],
            min_line=1,
        )
        result = filter_.to_meilisearch_filter()
        assert result is not None
        assert "AND" in result


class TestSearchQuery:
    """Tests for SearchQuery."""

    def test_default_query(self) -> None:
        """Test default query parameters."""
        query = SearchQuery(query="test")
        assert query.query == "test"
        assert query.search_type == SearchType.HYBRID
        assert query.limit == 20
        assert query.offset == 0
        assert query.highlight is True

    def test_custom_query(self) -> None:
        """Test custom query parameters."""
        query = SearchQuery(
            query="search term",
            search_type=SearchType.KEYWORD,
            limit=50,
            offset=10,
            highlight=False,
        )
        assert query.search_type == SearchType.KEYWORD
        assert query.limit == 50
        assert query.offset == 10
        assert query.highlight is False

    def test_limit_validation(self) -> None:
        """Test limit is validated."""
        query = SearchQuery(query="test", limit=0)
        assert query.limit == 1  # Minimum

        query = SearchQuery(query="test", limit=200)
        assert query.limit == 100  # Maximum

    def test_offset_validation(self) -> None:
        """Test offset is validated."""
        query = SearchQuery(query="test", offset=-5)
        assert query.offset == 0

    def test_with_filters(self) -> None:
        """Test query with filters."""
        filter_ = SearchFilter(languages=["python"])
        query = SearchQuery(query="test", filters=filter_)
        assert query.filters is not None
        assert query.filters.languages == ["python"]


class TestSearchHit:
    """Tests for SearchHit."""

    @pytest.fixture
    def sample_hit(self) -> SearchHit:
        """Create a sample search hit."""
        return SearchHit(
            id="test::func",
            name="func",
            qualified_name="test.func",
            code_type="function",
            file_path=Path("test.py"),
            line_start=10,
            line_end=20,
            content="def func(): pass",
            language="python",
            score=0.95,
            highlights={"content": ["def <mark>func</mark>(): pass"]},
            metadata={"decorators": []},
        )

    def test_to_dict(self, sample_hit: SearchHit) -> None:
        """Test conversion to dict."""
        d = sample_hit.to_dict()
        assert d["id"] == "test::func"
        assert d["name"] == "func"
        assert d["file_path"] == "test.py"
        assert d["score"] == 0.95

    def test_from_meilisearch(self) -> None:
        """Test creation from Meilisearch document."""
        doc = {
            "id": "test::func",
            "name": "func",
            "qualified_name": "test.func",
            "code_type": "function",
            "file_path": "test.py",
            "line_start": 10,
            "line_end": 20,
            "content": "def func(): pass",
            "language": "python",
            "_rankingScore": 0.85,
            "_formatted": {
                "content": "def <mark>func</mark>(): pass",
            },
        }
        hit = SearchHit.from_meilisearch(doc)
        assert hit.id == "test::func"
        assert hit.score == 0.85
        assert "content" in hit.highlights

    def test_from_meilisearch_minimal(self) -> None:
        """Test creation from minimal Meilisearch document."""
        doc = {"id": "test"}
        hit = SearchHit.from_meilisearch(doc)
        assert hit.id == "test"
        assert hit.name == ""
        assert hit.score == 0.0


class TestSearchResult:
    """Tests for SearchResult."""

    def test_empty_result(self) -> None:
        """Test empty search result."""
        result = SearchResult(
            hits=[],
            total=0,
            query="test",
            search_type=SearchType.KEYWORD,
        )
        assert result.count == 0
        assert not result.has_more

    def test_result_with_hits(self) -> None:
        """Test result with hits."""
        hits = [
            SearchHit(
                id=f"hit{i}",
                name=f"hit{i}",
                qualified_name=f"hit{i}",
                code_type="function",
                file_path=Path("test.py"),
                line_start=i,
                line_end=i + 10,
                content="",
                language="python",
            )
            for i in range(10)
        ]
        result = SearchResult(
            hits=hits,
            total=100,
            query="test",
            search_type=SearchType.HYBRID,
            offset=0,
            limit=10,
        )
        assert result.count == 10
        assert result.has_more

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        result = SearchResult(
            hits=[],
            total=0,
            query="test",
            search_type=SearchType.KEYWORD,
            processing_time_ms=5.0,
        )
        d = result.to_dict()
        assert d["query"] == "test"
        assert d["search_type"] == "keyword"
        assert d["processing_time_ms"] == 5.0


class TestIndexDocument:
    """Tests for IndexDocument."""

    def test_basic_document(self) -> None:
        """Test basic document creation."""
        doc = IndexDocument(
            id="test::func",
            name="func",
            qualified_name="test.func",
            code_type="function",
            file_path="test.py",
            line_start=10,
            line_end=20,
            content="def func(): pass",
            language="python",
        )
        assert doc.id == "test::func"
        assert doc.embedding is None

    def test_document_with_embedding(self) -> None:
        """Test document with embedding."""
        doc = IndexDocument(
            id="test",
            name="test",
            qualified_name="test",
            code_type="function",
            file_path="test.py",
            line_start=1,
            line_end=5,
            content="",
            language="python",
            embedding=[0.1, 0.2, 0.3],
        )
        assert doc.embedding == [0.1, 0.2, 0.3]

    def test_to_dict_without_embedding(self) -> None:
        """Test to_dict without embedding."""
        doc = IndexDocument(
            id="test",
            name="test",
            qualified_name="test",
            code_type="function",
            file_path="test.py",
            line_start=1,
            line_end=5,
            content="",
            language="python",
        )
        d = doc.to_dict()
        assert "_vectors" not in d

    def test_to_dict_with_embedding(self) -> None:
        """Test to_dict with embedding."""
        doc = IndexDocument(
            id="test",
            name="test",
            qualified_name="test",
            code_type="function",
            file_path="test.py",
            line_start=1,
            line_end=5,
            content="",
            language="python",
            embedding=[0.1, 0.2, 0.3],
        )
        d = doc.to_dict()
        assert "_vectors" in d
        assert d["_vectors"]["default"] == [0.1, 0.2, 0.3]


class TestIndexStats:
    """Tests for IndexStats."""

    def test_from_meilisearch(self) -> None:
        """Test creation from Meilisearch stats."""
        stats_data = {
            "numberOfDocuments": 100,
            "isIndexing": False,
            "fieldDistribution": {"content": 100, "name": 100},
        }
        stats = IndexStats.from_meilisearch(stats_data)
        assert stats.total_documents == 100
        assert stats.is_indexing is False
        assert stats.field_distribution == {"content": 100, "name": 100}

    def test_from_meilisearch_minimal(self) -> None:
        """Test creation from minimal stats."""
        stats = IndexStats.from_meilisearch({})
        assert stats.total_documents == 0
        assert stats.is_indexing is False


class TestEnums:
    """Tests for enum classes."""

    def test_search_type_values(self) -> None:
        """Test SearchType values."""
        assert SearchType.KEYWORD.value == "keyword"
        assert SearchType.SEMANTIC.value == "semantic"
        assert SearchType.HYBRID.value == "hybrid"

    def test_sort_order_values(self) -> None:
        """Test SortOrder values."""
        assert SortOrder.ASC.value == "asc"
        assert SortOrder.DESC.value == "desc"

    def test_sort_field_values(self) -> None:
        """Test SortField values."""
        assert SortField.RELEVANCE.value == "relevance"
        assert SortField.FILE_PATH.value == "file_path"
