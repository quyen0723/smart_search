"""Tests for parsing models."""

from pathlib import Path

import pytest

from smart_search.parsing.models import (
    CodeChunk,
    CodeUnit,
    CodeUnitType,
    Language,
    Parameter,
    ParsedFile,
    Position,
    Span,
)


class TestLanguage:
    """Tests for Language enum."""

    def test_from_extension_python(self) -> None:
        """Test Python extension detection."""
        assert Language.from_extension("py") == Language.PYTHON
        assert Language.from_extension(".py") == Language.PYTHON

    def test_from_extension_javascript(self) -> None:
        """Test JavaScript extension detection."""
        assert Language.from_extension("js") == Language.JAVASCRIPT
        assert Language.from_extension("jsx") == Language.JAVASCRIPT

    def test_from_extension_typescript(self) -> None:
        """Test TypeScript extension detection."""
        assert Language.from_extension("ts") == Language.TYPESCRIPT
        assert Language.from_extension("tsx") == Language.TYPESCRIPT

    def test_from_extension_java(self) -> None:
        """Test Java extension detection."""
        assert Language.from_extension("java") == Language.JAVA

    def test_from_extension_go(self) -> None:
        """Test Go extension detection."""
        assert Language.from_extension("go") == Language.GO

    def test_from_extension_unsupported(self) -> None:
        """Test unsupported extension returns None."""
        assert Language.from_extension("xyz") is None
        assert Language.from_extension("") is None

    def test_from_extension_case_insensitive(self) -> None:
        """Test case insensitive extension matching."""
        assert Language.from_extension("PY") == Language.PYTHON
        assert Language.from_extension("Js") == Language.JAVASCRIPT


class TestPosition:
    """Tests for Position dataclass."""

    def test_valid_position(self) -> None:
        """Test creating valid position."""
        pos = Position(line=1, column=0)
        assert pos.line == 1
        assert pos.column == 0

    def test_invalid_line(self) -> None:
        """Test that line must be >= 1."""
        with pytest.raises(ValueError, match="Line must be >= 1"):
            Position(line=0, column=0)

    def test_invalid_column(self) -> None:
        """Test that column must be >= 0."""
        with pytest.raises(ValueError, match="Column must be >= 0"):
            Position(line=1, column=-1)


class TestSpan:
    """Tests for Span dataclass."""

    def test_span_properties(self) -> None:
        """Test Span properties."""
        span = Span(
            start=Position(line=1, column=0),
            end=Position(line=5, column=10),
        )
        assert span.start_line == 1
        assert span.end_line == 5
        assert span.line_count == 5


class TestParameter:
    """Tests for Parameter dataclass."""

    def test_simple_parameter(self) -> None:
        """Test simple parameter."""
        param = Parameter(name="x")
        assert param.name == "x"
        assert param.type_annotation is None
        assert param.default_value is None
        assert param.is_variadic is False
        assert param.is_keyword is False

    def test_typed_parameter(self) -> None:
        """Test parameter with type."""
        param = Parameter(name="x", type_annotation="int")
        assert param.type_annotation == "int"

    def test_default_parameter(self) -> None:
        """Test parameter with default."""
        param = Parameter(name="x", default_value="10")
        assert param.default_value == "10"

    def test_variadic_parameter(self) -> None:
        """Test *args parameter."""
        param = Parameter(name="args", is_variadic=True)
        assert param.is_variadic is True

    def test_keyword_parameter(self) -> None:
        """Test **kwargs parameter."""
        param = Parameter(name="kwargs", is_keyword=True)
        assert param.is_keyword is True


class TestCodeUnit:
    """Tests for CodeUnit dataclass."""

    @pytest.fixture
    def sample_unit(self) -> CodeUnit:
        """Create a sample code unit."""
        return CodeUnit(
            id="test.py::my_func",
            name="my_func",
            qualified_name="test.my_func",
            type=CodeUnitType.FUNCTION,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=5, column=0),
            ),
            language=Language.PYTHON,
            content="def my_func():\n    pass",
            docstring="A test function.",
            signature="my_func()",
        )

    def test_unit_properties(self, sample_unit: CodeUnit) -> None:
        """Test CodeUnit properties."""
        assert sample_unit.start_line == 1
        assert sample_unit.end_line == 5
        assert sample_unit.name == "my_func"

    def test_to_dict(self, sample_unit: CodeUnit) -> None:
        """Test conversion to dictionary."""
        d = sample_unit.to_dict()
        assert d["id"] == "test.py::my_func"
        assert d["name"] == "my_func"
        assert d["type"] == "function"
        assert d["language"] == "python"
        assert d["docstring"] == "A test function."

    def test_unit_with_parameters(self) -> None:
        """Test unit with parameters."""
        unit = CodeUnit(
            id="test.py::add",
            name="add",
            qualified_name="test.add",
            type=CodeUnitType.FUNCTION,
            file_path=Path("test.py"),
            span=Span(
                start=Position(line=1, column=0),
                end=Position(line=2, column=0),
            ),
            language=Language.PYTHON,
            content="def add(a, b): return a + b",
            parameters=[
                Parameter(name="a", type_annotation="int"),
                Parameter(name="b", type_annotation="int"),
            ],
            return_type="int",
        )
        d = unit.to_dict()
        assert len(d["parameters"]) == 2
        assert d["parameters"][0]["name"] == "a"
        assert d["return_type"] == "int"


class TestParsedFile:
    """Tests for ParsedFile dataclass."""

    def test_parsed_file_properties(self) -> None:
        """Test ParsedFile properties."""
        pf = ParsedFile(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            content="# test",
            content_hash="abc123",
        )
        assert pf.has_errors is False
        assert pf.unit_count == 0

    def test_parsed_file_with_errors(self) -> None:
        """Test ParsedFile with errors."""
        pf = ParsedFile(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            content="# test",
            content_hash="abc123",
            parse_errors=["Syntax error at line 1"],
        )
        assert pf.has_errors is True

    def test_parsed_file_with_units(self) -> None:
        """Test ParsedFile with units."""
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
            content="def func(): pass",
        )
        pf = ParsedFile(
            file_path=Path("test.py"),
            language=Language.PYTHON,
            content="def func(): pass",
            content_hash="abc123",
            units=[unit],
        )
        assert pf.unit_count == 1


class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_single_chunk(self) -> None:
        """Test non-split chunk."""
        chunk = CodeChunk(
            id="unit::0",
            unit_id="unit",
            content="def func(): pass",
            chunk_index=0,
            total_chunks=1,
        )
        assert chunk.is_split is False

    def test_split_chunk(self) -> None:
        """Test split chunk."""
        chunk = CodeChunk(
            id="unit::1",
            unit_id="unit",
            content="    return x",
            chunk_index=1,
            total_chunks=3,
        )
        assert chunk.is_split is True

    def test_to_embedding_text(self) -> None:
        """Test embedding text generation."""
        chunk = CodeChunk(
            id="unit::0",
            unit_id="unit",
            content="def func(): pass",
            chunk_index=0,
            total_chunks=1,
            context_before="# Function definition",
        )
        text = chunk.to_embedding_text()
        assert "# Function definition" in text
        assert "def func(): pass" in text


class TestCodeUnitType:
    """Tests for CodeUnitType enum."""

    def test_all_types_exist(self) -> None:
        """Test all expected types exist."""
        assert CodeUnitType.MODULE.value == "module"
        assert CodeUnitType.CLASS.value == "class"
        assert CodeUnitType.FUNCTION.value == "function"
        assert CodeUnitType.METHOD.value == "method"
        assert CodeUnitType.PROPERTY.value == "property"
        assert CodeUnitType.VARIABLE.value == "variable"
        assert CodeUnitType.IMPORT.value == "import"
        assert CodeUnitType.DECORATOR.value == "decorator"
