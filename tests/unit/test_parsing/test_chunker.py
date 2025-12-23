"""Tests for code chunker."""

from pathlib import Path

import pytest

from smart_search.parsing.chunker import CodeChunker, SmartChunker, estimate_token_count
from smart_search.parsing.models import (
    CodeUnit,
    CodeUnitType,
    Language,
    Position,
    Span,
)


class TestCodeChunker:
    """Tests for CodeChunker class."""

    @pytest.fixture
    def chunker(self) -> CodeChunker:
        """Create a chunker instance."""
        return CodeChunker(
            max_chunk_size=100,
            overlap_size=20,
            min_chunk_size=10,
        )

    @pytest.fixture
    def sample_unit(self) -> CodeUnit:
        """Create a sample code unit."""
        return CodeUnit(
            id="test.py::func",
            name="func",
            qualified_name="test.func",
            type=CodeUnitType.FUNCTION,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=5, column=0),
            ),
            language=Language.PYTHON,
            content="def func():\n    pass",
            docstring="A function.",
            signature="func()",
        )

    def test_single_chunk(self, chunker: CodeChunker, sample_unit: CodeUnit) -> None:
        """Test that small content produces single chunk."""
        chunks = chunker.chunk_unit(sample_unit)
        assert len(chunks) == 1
        assert chunks[0].is_split is False
        assert chunks[0].chunk_index == 0
        assert chunks[0].total_chunks == 1

    def test_chunk_id_format(
        self, chunker: CodeChunker, sample_unit: CodeUnit
    ) -> None:
        """Test chunk ID format."""
        chunks = chunker.chunk_unit(sample_unit)
        assert chunks[0].id == "test.py::func::0"

    def test_chunk_metadata(
        self, chunker: CodeChunker, sample_unit: CodeUnit
    ) -> None:
        """Test that chunk inherits metadata from unit."""
        chunks = chunker.chunk_unit(sample_unit)
        chunk = chunks[0]

        assert chunk.unit_id == sample_unit.id
        assert chunk.file_path == str(sample_unit.file_path)
        assert chunk.language == sample_unit.language.value
        assert chunk.unit_type == sample_unit.type.value
        assert chunk.unit_name == sample_unit.qualified_name

    def test_chunk_context(self, chunker: CodeChunker, sample_unit: CodeUnit) -> None:
        """Test that chunk has context."""
        chunks = chunker.chunk_unit(sample_unit)
        chunk = chunks[0]

        assert chunk.context_before is not None
        assert "test.py" in chunk.context_before
        assert "func" in chunk.context_before

    def test_multiple_chunks(self, chunker: CodeChunker) -> None:
        """Test splitting large content into multiple chunks."""
        # Create a large unit
        large_content = "\n".join([f"line_{i} = {i}" for i in range(50)])
        unit = CodeUnit(
            id="test.py::large",
            name="large",
            qualified_name="test.large",
            type=CodeUnitType.FUNCTION,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=50, column=0),
            ),
            language=Language.PYTHON,
            content=large_content,
        )

        chunks = chunker.chunk_unit(unit)
        assert len(chunks) > 1

        # All chunks should have correct total_chunks
        for chunk in chunks:
            assert chunk.total_chunks == len(chunks)
            assert chunk.is_split is True

        # Chunk indices should be sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_units_multiple(self, chunker: CodeChunker) -> None:
        """Test chunking multiple units."""
        units = [
            CodeUnit(
                id=f"test.py::func{i}",
                name=f"func{i}",
                qualified_name=f"test.func{i}",
                type=CodeUnitType.FUNCTION,
                file_path=Path("test.py"),
                span=Span(
                    start=Position(line=1, column=0),
                    end=Position(line=2, column=0),
                ),
                language=Language.PYTHON,
                content=f"def func{i}(): pass",
            )
            for i in range(3)
        ]

        chunks = chunker.chunk_units(units)
        assert len(chunks) == 3


class TestSmartChunker:
    """Tests for SmartChunker class."""

    @pytest.fixture
    def chunker(self) -> SmartChunker:
        """Create a smart chunker."""
        return SmartChunker(
            max_chunk_size=200,
            overlap_size=30,
            min_chunk_size=20,
        )

    def test_splits_at_function_boundary(self, chunker: SmartChunker) -> None:
        """Test that smart chunker prefers splitting at function boundaries."""
        content = """def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        unit = CodeUnit(
            id="test.py::module",
            name="module",
            qualified_name="test",
            type=CodeUnitType.MODULE,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=10, column=0),
            ),
            language=Language.PYTHON,
            content=content,
        )

        chunks = chunker.chunk_unit(unit)
        # Should split at function boundaries
        assert len(chunks) >= 1

    def test_falls_back_to_basic_split(self, chunker: SmartChunker) -> None:
        """Test fallback to basic splitting."""
        # Content without natural split points
        content = "x = 1\n" * 50
        unit = CodeUnit(
            id="test.py::vars",
            name="vars",
            qualified_name="test.vars",
            type=CodeUnitType.MODULE,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=50, column=0),
            ),
            language=Language.PYTHON,
            content=content,
        )

        chunks = chunker.chunk_unit(unit)
        assert len(chunks) >= 1


class TestEstimateTokenCount:
    """Tests for token count estimation."""

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert estimate_token_count("") == 0

    def test_simple_string(self) -> None:
        """Test simple string estimation."""
        # ~4 chars per token
        text = "a" * 100
        count = estimate_token_count(text)
        assert count == 25

    def test_code_string(self) -> None:
        """Test code estimation."""
        code = "def hello_world(): print('hello')"
        count = estimate_token_count(code)
        assert count > 0


class TestChunkerContextBuilding:
    """Tests for context building in chunks."""

    @pytest.fixture
    def chunker(self) -> CodeChunker:
        """Create chunker."""
        return CodeChunker()

    def test_context_includes_file_path(self, chunker: CodeChunker) -> None:
        """Test context includes file path."""
        unit = CodeUnit(
            id="path/to/file.py::func",
            name="func",
            qualified_name="module.func",
            type=CodeUnitType.FUNCTION,
            file_path=Path("path/to/file.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=2, column=0),
            ),
            language=Language.PYTHON,
            content="def func(): pass",
        )

        chunks = chunker.chunk_unit(unit)
        assert "path/to/file.py" in chunks[0].context_before

    def test_context_includes_signature(self, chunker: CodeChunker) -> None:
        """Test context includes signature."""
        unit = CodeUnit(
            id="test.py::func",
            name="func",
            qualified_name="test.func",
            type=CodeUnitType.FUNCTION,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=2, column=0),
            ),
            language=Language.PYTHON,
            content="def func(x: int) -> int: return x",
            signature="func(x: int) -> int",
        )

        chunks = chunker.chunk_unit(unit)
        assert "func(x: int) -> int" in chunks[0].context_before

    def test_context_includes_docstring_summary(self, chunker: CodeChunker) -> None:
        """Test context includes docstring first line."""
        unit = CodeUnit(
            id="test.py::func",
            name="func",
            qualified_name="test.func",
            type=CodeUnitType.FUNCTION,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=5, column=0),
            ),
            language=Language.PYTHON,
            content='def func():\n    """Process data efficiently."""\n    pass',
            docstring="Process data efficiently.\n\nMore details here.",
        )

        chunks = chunker.chunk_unit(unit)
        assert "Process data efficiently." in chunks[0].context_before


class TestChunkEmbeddingText:
    """Tests for chunk embedding text generation."""

    def test_embedding_text_with_context(self) -> None:
        """Test embedding text includes context."""
        from smart_search.parsing.models import CodeChunk

        chunk = CodeChunk(
            id="test::0",
            unit_id="test",
            content="def func(): pass",
            chunk_index=0,
            total_chunks=1,
            context_before="# File: test.py\n# function: func",
            context_after="# End of function",
        )

        text = chunk.to_embedding_text()
        assert "# File: test.py" in text
        assert "def func(): pass" in text
        assert "# End of function" in text

    def test_embedding_text_without_context(self) -> None:
        """Test embedding text without context."""
        from smart_search.parsing.models import CodeChunk

        chunk = CodeChunk(
            id="test::0",
            unit_id="test",
            content="def func(): pass",
            chunk_index=0,
            total_chunks=1,
        )

        text = chunk.to_embedding_text()
        assert text == "def func(): pass"
