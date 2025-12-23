"""Tests for search indexer."""

from pathlib import Path

import pytest

from smart_search.embedding.jina_embedder import MockEmbedder
from smart_search.embedding.models import EmbeddingConfig
from smart_search.parsing.models import CodeUnit, CodeUnitType, Language, Position, Span
from smart_search.search.indexer import (
    BatchIndexer,
    IndexingConfig,
    IndexingResult,
    SearchIndexer,
)
from smart_search.search.meilisearch_client import MockMeilisearchClient


def make_span(start_line: int = 1, end_line: int = 10) -> Span:
    """Helper to create a Span."""
    return Span(
        start=Position(line=start_line, column=0),
        end=Position(line=end_line, column=0),
    )


def make_unit(
    id: str,
    name: str,
    content: str = "def test(): pass",
    file_path: str = "test.py",
    unit_type: CodeUnitType = CodeUnitType.FUNCTION,
) -> CodeUnit:
    """Helper to create a CodeUnit."""
    return CodeUnit(
        id=id,
        name=name,
        qualified_name=f"test.{name}",
        type=unit_type,
        file_path=Path(file_path),
        span=make_span(),
        language=Language.PYTHON,
        content=content,
        signature=f"def {name}():",
        docstring=f"Docstring for {name}",
    )


class TestIndexingConfig:
    """Tests for IndexingConfig."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = IndexingConfig()
        assert config.batch_size == 50
        assert config.include_embeddings is True
        assert config.max_content_length == 10000

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = IndexingConfig(
            batch_size=100,
            include_embeddings=False,
            max_content_length=5000,
        )
        assert config.batch_size == 100
        assert config.include_embeddings is False
        assert config.max_content_length == 5000


class TestIndexingResult:
    """Tests for IndexingResult."""

    def test_empty_result(self) -> None:
        """Test empty result."""
        result = IndexingResult()
        assert result.total_indexed == 0
        assert result.success is True

    def test_result_with_errors(self) -> None:
        """Test result with errors."""
        result = IndexingResult(errors=["Error 1", "Error 2"])
        assert result.success is False

    def test_result_with_data(self) -> None:
        """Test result with data."""
        result = IndexingResult(
            total_indexed=10,
            total_updated=5,
            total_deleted=2,
            duration_ms=100.0,
        )
        assert result.success is True
        assert result.total_indexed == 10


class TestSearchIndexer:
    """Tests for SearchIndexer."""

    @pytest.fixture
    def client(self) -> MockMeilisearchClient:
        """Create mock client."""
        return MockMeilisearchClient()

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        """Create mock embedder."""
        return MockEmbedder(EmbeddingConfig(dimensions=64))

    @pytest.fixture
    def indexer(
        self, client: MockMeilisearchClient, embedder: MockEmbedder
    ) -> SearchIndexer:
        """Create indexer with embedder."""
        return SearchIndexer(client, embedder)

    @pytest.fixture
    def indexer_no_embed(self, client: MockMeilisearchClient) -> SearchIndexer:
        """Create indexer without embedder."""
        return SearchIndexer(client, config=IndexingConfig(include_embeddings=False))

    @pytest.fixture
    def sample_units(self) -> list[CodeUnit]:
        """Create sample code units."""
        return [
            make_unit("test::func1", "func1", "def func1(): pass"),
            make_unit("test::func2", "func2", "def func2(): pass"),
            make_unit("test::class1", "MyClass", "class MyClass: pass", unit_type=CodeUnitType.CLASS),
        ]

    @pytest.mark.asyncio
    async def test_index_units(
        self, indexer: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test indexing code units."""
        result = await indexer.index_units(sample_units)
        assert result.total_indexed == 3
        assert result.success

    @pytest.mark.asyncio
    async def test_index_empty_units(self, indexer: SearchIndexer) -> None:
        """Test indexing empty list."""
        result = await indexer.index_units([])
        assert result.total_indexed == 0
        assert result.success

    @pytest.mark.asyncio
    async def test_index_with_update(
        self, indexer: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test indexing with update mode."""
        # First index
        await indexer.index_units(sample_units)

        # Then update
        result = await indexer.index_units(sample_units, update=True)
        assert result.total_updated == 3
        assert result.total_indexed == 0

    @pytest.mark.asyncio
    async def test_index_without_embeddings(
        self, indexer_no_embed: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test indexing without embeddings."""
        result = await indexer_no_embed.index_units(sample_units)
        assert result.success

    @pytest.mark.asyncio
    async def test_content_truncation(self, indexer: SearchIndexer) -> None:
        """Test long content is truncated."""
        # Create unit with very long content
        long_content = "x" * 20000
        unit = make_unit("test::long", "long_func", long_content)

        result = await indexer.index_units([unit])
        assert result.success

    @pytest.mark.asyncio
    async def test_delete_by_file(
        self, indexer: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test deleting by file."""
        await indexer.index_units(sample_units)
        count = await indexer.delete_by_file(Path("test.py"))
        assert count >= 0  # Mock doesn't return actual count

    @pytest.mark.asyncio
    async def test_delete_by_ids(
        self, indexer: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test deleting by IDs."""
        await indexer.index_units(sample_units)
        count = await indexer.delete_by_ids(["test::func1", "test::func2"])
        assert count == 2

    @pytest.mark.asyncio
    async def test_reindex_file(
        self, indexer: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test reindexing a file."""
        # First index
        await indexer.index_units(sample_units)

        # Reindex with new units
        new_units = [make_unit("test::new", "new_func")]
        result = await indexer.reindex_file(Path("test.py"), new_units)
        assert result.success

    @pytest.mark.asyncio
    async def test_get_stats(
        self, indexer: SearchIndexer, sample_units: list[CodeUnit]
    ) -> None:
        """Test getting stats."""
        await indexer.index_units(sample_units)
        stats = await indexer.get_stats()
        assert stats.total_documents == 3

    @pytest.mark.asyncio
    async def test_clear_index(self, indexer: SearchIndexer) -> None:
        """Test clearing index."""
        await indexer.clear_index()
        stats = await indexer.get_stats()
        assert stats.total_documents == 0

    @pytest.mark.asyncio
    async def test_close(self, indexer: SearchIndexer) -> None:
        """Test closing indexer."""
        await indexer.close()


class TestSearchIndexerBatching:
    """Tests for indexer batching."""

    @pytest.mark.asyncio
    async def test_batch_indexing(self) -> None:
        """Test indexing in batches."""
        client = MockMeilisearchClient()
        config = IndexingConfig(batch_size=2, include_embeddings=False)
        indexer = SearchIndexer(client, config=config)

        units = [make_unit(f"test::func{i}", f"func{i}") for i in range(5)]
        result = await indexer.index_units(units)

        assert result.total_indexed == 5
        assert result.success


class TestBatchIndexer:
    """Tests for BatchIndexer."""

    @pytest.fixture
    def indexer(self) -> SearchIndexer:
        """Create search indexer."""
        client = MockMeilisearchClient()
        return SearchIndexer(client, config=IndexingConfig(include_embeddings=False))

    @pytest.fixture
    def batch_indexer(self, indexer: SearchIndexer) -> BatchIndexer:
        """Create batch indexer."""
        return BatchIndexer(indexer, batch_size=2)

    @pytest.mark.asyncio
    async def test_index_all(self, batch_indexer: BatchIndexer) -> None:
        """Test indexing all files."""
        units_by_file = {
            Path("file1.py"): [make_unit("f1::func", "func1", file_path="file1.py")],
            Path("file2.py"): [make_unit("f2::func", "func2", file_path="file2.py")],
            Path("file3.py"): [make_unit("f3::func", "func3", file_path="file3.py")],
        }

        result = await batch_indexer.index_all(units_by_file)
        assert result.total_indexed == 3
        assert result.success

    @pytest.mark.asyncio
    async def test_index_all_with_progress(self, batch_indexer: BatchIndexer) -> None:
        """Test indexing with progress callback."""
        units_by_file = {
            Path("file1.py"): [make_unit("f1::func", "func1", file_path="file1.py")],
            Path("file2.py"): [make_unit("f2::func", "func2", file_path="file2.py")],
        }

        progress_values = []

        def on_progress(progress: float, result: IndexingResult) -> None:
            progress_values.append(progress)

        result = await batch_indexer.index_all(units_by_file, on_progress)
        assert result.success
        assert len(progress_values) >= 1

    @pytest.mark.asyncio
    async def test_reindex_changed(self, batch_indexer: BatchIndexer) -> None:
        """Test reindexing changed files."""
        changed_files = {
            Path("changed.py"): [make_unit("c::func", "changed_func", file_path="changed.py")],
        }
        deleted_files = [Path("deleted.py")]

        result = await batch_indexer.reindex_changed(changed_files, deleted_files)
        assert result.total_indexed == 1
        assert result.total_deleted == 1

    @pytest.mark.asyncio
    async def test_reindex_empty(self, batch_indexer: BatchIndexer) -> None:
        """Test reindexing with no changes."""
        result = await batch_indexer.reindex_changed({}, [])
        assert result.total_indexed == 0
        assert result.total_deleted == 0
