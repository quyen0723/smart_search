"""Tests for embedding pipeline."""

from pathlib import Path

import pytest

from smart_search.embedding.jina_embedder import MockEmbedder
from smart_search.embedding.models import EmbeddingConfig
from smart_search.embedding.pipeline import (
    ChunkingConfig,
    CodeChunk,
    CodeChunker,
    EmbeddingPipeline,
    PipelineResult,
)
from smart_search.parsing.models import CodeUnit, CodeUnitType, Language, Position, Span


def make_span(start_line: int = 1, end_line: int = 10) -> Span:
    """Helper to create a Span."""
    return Span(
        start=Position(line=start_line, column=0),
        end=Position(line=end_line, column=0),
    )


def make_unit(
    id: str,
    name: str,
    content: str,
    unit_type: CodeUnitType = CodeUnitType.FUNCTION,
    start_line: int = 1,
    end_line: int = 10,
) -> CodeUnit:
    """Helper to create a CodeUnit."""
    return CodeUnit(
        id=id,
        name=name,
        qualified_name=f"test.{name}",
        type=unit_type,
        file_path=Path("test.py"),
        span=make_span(start_line, end_line),
        language=Language.PYTHON,
        content=content,
        signature=f"def {name}():",
        docstring=f"Docstring for {name}",
    )


class TestChunkingConfig:
    """Tests for ChunkingConfig."""

    def test_default_config(self) -> None:
        """Test default chunking configuration."""
        config = ChunkingConfig()
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 100
        assert config.overlap == 200
        assert config.include_context is True

    def test_custom_config(self) -> None:
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            max_chunk_size=1000,
            min_chunk_size=50,
            overlap=100,
            include_context=False,
        )
        assert config.max_chunk_size == 1000
        assert config.min_chunk_size == 50
        assert config.overlap == 100
        assert config.include_context is False


class TestCodeChunk:
    """Tests for CodeChunk."""

    def test_create_chunk(self) -> None:
        """Test creating a code chunk."""
        chunk = CodeChunk(
            id="test::chunk_0",
            unit_id="test",
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        assert chunk.id == "test::chunk_0"
        assert chunk.unit_id == "test"
        assert chunk.content == "def foo(): pass"
        assert chunk.start_line == 1
        assert chunk.end_line == 1

    def test_content_hash(self) -> None:
        """Test content hash generation."""
        chunk = CodeChunk(
            id="test",
            unit_id="unit",
            content="test content",
            start_line=1,
            end_line=1,
        )
        hash1 = chunk.content_hash
        assert len(hash1) == 16
        # Same content should have same hash
        chunk2 = CodeChunk(
            id="other",
            unit_id="unit",
            content="test content",
            start_line=1,
            end_line=1,
        )
        assert chunk2.content_hash == hash1

    def test_full_content_without_context(self) -> None:
        """Test full content without context."""
        chunk = CodeChunk(
            id="test",
            unit_id="unit",
            content="def foo(): pass",
            start_line=1,
            end_line=1,
        )
        assert chunk.full_content == "def foo(): pass"

    def test_full_content_with_context(self) -> None:
        """Test full content with context."""
        chunk = CodeChunk(
            id="test",
            unit_id="unit",
            content="def foo(): pass",
            start_line=1,
            end_line=1,
            context="Function: foo",
        )
        assert chunk.full_content == "Function: foo\n\ndef foo(): pass"


class TestCodeChunker:
    """Tests for CodeChunker."""

    @pytest.fixture
    def chunker(self) -> CodeChunker:
        """Create a code chunker."""
        return CodeChunker()

    def test_chunk_small_unit(self, chunker: CodeChunker) -> None:
        """Test chunking a small unit."""
        unit = make_unit("test", "foo", "def foo(): pass")
        chunks = chunker.chunk_unit(unit)
        assert len(chunks) == 1
        assert chunks[0].content == "def foo(): pass"
        assert chunks[0].unit_id == "test"

    def test_chunk_large_unit(self) -> None:
        """Test chunking a large unit."""
        chunker = CodeChunker(ChunkingConfig(max_chunk_size=100, overlap=20))
        # Create content larger than max_chunk_size
        lines = [f"line {i}" for i in range(50)]
        content = "\n".join(lines)
        unit = make_unit("test", "large", content, end_line=50)

        chunks = chunker.chunk_unit(unit)
        assert len(chunks) > 1
        # All chunks should have content
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_chunk_includes_metadata(self, chunker: CodeChunker) -> None:
        """Test chunk metadata."""
        unit = make_unit("test", "foo", "def foo(): pass")
        chunks = chunker.chunk_unit(unit)
        assert chunks[0].metadata["name"] == "foo"
        assert chunks[0].metadata["type"] == "function"
        assert chunks[0].metadata["file"] == "test.py"

    def test_chunk_includes_context(self, chunker: CodeChunker) -> None:
        """Test chunk context."""
        unit = make_unit("test", "foo", "def foo(): pass")
        chunks = chunker.chunk_unit(unit)
        context = chunks[0].context
        assert "Signature:" in context
        assert "Name:" in context
        assert "Type:" in context

    def test_chunk_without_context(self) -> None:
        """Test chunking without context."""
        chunker = CodeChunker(ChunkingConfig(include_context=False))
        unit = make_unit("test", "foo", "def foo(): pass")
        chunks = chunker.chunk_unit(unit)
        assert chunks[0].context == ""

    def test_chunk_multiple_units(self, chunker: CodeChunker) -> None:
        """Test chunking multiple units."""
        units = [
            make_unit("test1", "foo", "def foo(): pass"),
            make_unit("test2", "bar", "def bar(): pass"),
        ]
        chunks = chunker.chunk_units(units)
        assert len(chunks) == 2
        assert chunks[0].unit_id == "test1"
        assert chunks[1].unit_id == "test2"

    def test_chunk_line_numbers(self, chunker: CodeChunker) -> None:
        """Test chunk line numbers."""
        unit = make_unit("test", "foo", "def foo():\n    pass", start_line=5, end_line=6)
        chunks = chunker.chunk_unit(unit)
        assert chunks[0].start_line == 5
        assert chunks[0].end_line == 6

    def test_chunk_empty_content(self, chunker: CodeChunker) -> None:
        """Test chunking unit with minimal content."""
        unit = make_unit("test", "foo", "x")
        chunks = chunker.chunk_unit(unit)
        assert len(chunks) == 1

    def test_chunk_id_format(self, chunker: CodeChunker) -> None:
        """Test chunk ID format."""
        unit = make_unit("my_unit", "foo", "def foo(): pass")
        chunks = chunker.chunk_unit(unit)
        assert chunks[0].id == "my_unit::chunk_0"


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_empty_result(self) -> None:
        """Test empty pipeline result."""
        result = PipelineResult(
            embeddings=[],
            total_chunks=0,
            total_tokens=0,
        )
        assert len(result.embeddings) == 0
        assert result.total_chunks == 0
        assert result.skipped == 0
        assert result.errors == []

    def test_result_with_data(self) -> None:
        """Test pipeline result with data."""
        result = PipelineResult(
            embeddings=[],
            total_chunks=10,
            total_tokens=500,
            skipped=3,
            errors=["error1"],
        )
        assert result.total_chunks == 10
        assert result.total_tokens == 500
        assert result.skipped == 3
        assert result.errors == ["error1"]


class TestEmbeddingPipeline:
    """Tests for EmbeddingPipeline."""

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        """Create a mock embedder."""
        return MockEmbedder(EmbeddingConfig(dimensions=64))

    @pytest.fixture
    def pipeline(self, embedder: MockEmbedder) -> EmbeddingPipeline:
        """Create an embedding pipeline."""
        return EmbeddingPipeline(embedder)

    @pytest.mark.asyncio
    async def test_embed_units(self, pipeline: EmbeddingPipeline) -> None:
        """Test embedding code units."""
        units = [
            make_unit("test1", "foo", "def foo(): pass"),
            make_unit("test2", "bar", "def bar(): pass"),
        ]
        result = await pipeline.embed_units(units)
        assert len(result.embeddings) == 2
        assert result.total_chunks == 2
        assert result.skipped == 0

    @pytest.mark.asyncio
    async def test_embed_units_with_cached(
        self, pipeline: EmbeddingPipeline
    ) -> None:
        """Test embedding with cached hashes."""
        unit = make_unit("test", "foo", "def foo(): pass")
        # Get the chunk hash
        chunks = pipeline.chunker.chunk_unit(unit)
        cached = {chunks[0].content_hash}

        result = await pipeline.embed_units([unit], cached_hashes=cached)
        assert len(result.embeddings) == 0
        assert result.skipped == 1

    @pytest.mark.asyncio
    async def test_embed_chunks(self, pipeline: EmbeddingPipeline) -> None:
        """Test embedding code chunks directly."""
        chunks = [
            CodeChunk(
                id="test1",
                unit_id="unit1",
                content="def foo(): pass",
                start_line=1,
                end_line=1,
            ),
            CodeChunk(
                id="test2",
                unit_id="unit2",
                content="def bar(): pass",
                start_line=1,
                end_line=1,
            ),
        ]
        result = await pipeline.embed_chunks(chunks)
        assert len(result.embeddings) == 2

    @pytest.mark.asyncio
    async def test_embed_chunks_empty(self, pipeline: EmbeddingPipeline) -> None:
        """Test embedding empty chunks list."""
        result = await pipeline.embed_chunks([])
        assert len(result.embeddings) == 0
        assert result.total_chunks == 0

    @pytest.mark.asyncio
    async def test_embed_text(self, pipeline: EmbeddingPipeline) -> None:
        """Test embedding arbitrary text."""
        result = await pipeline.embed_text("hello world")
        assert result.text == "hello world"
        assert len(result.embedding) == 64

    @pytest.mark.asyncio
    async def test_embed_texts(self, pipeline: EmbeddingPipeline) -> None:
        """Test embedding multiple texts."""
        texts = ["hello", "world"]
        results = await pipeline.embed_texts(texts)
        assert len(results) == 2
        assert results[0].text == "hello"
        assert results[1].text == "world"

    @pytest.mark.asyncio
    async def test_close(self, pipeline: EmbeddingPipeline) -> None:
        """Test pipeline close."""
        await pipeline.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_chunk_embedding_metadata(
        self, pipeline: EmbeddingPipeline
    ) -> None:
        """Test chunk embedding metadata."""
        unit = make_unit("test", "foo", "def foo(): pass")
        result = await pipeline.embed_units([unit])
        embedding = result.embeddings[0]
        assert "start_line" in embedding.metadata
        assert "end_line" in embedding.metadata
        assert "token_count" in embedding.metadata

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_chunker(
        self, embedder: MockEmbedder
    ) -> None:
        """Test pipeline with custom chunker."""
        chunker = CodeChunker(ChunkingConfig(include_context=False))
        pipeline = EmbeddingPipeline(embedder, chunker)
        unit = make_unit("test", "foo", "def foo(): pass")
        result = await pipeline.embed_units([unit])
        assert len(result.embeddings) == 1
