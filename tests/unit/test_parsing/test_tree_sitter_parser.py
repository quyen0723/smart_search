"""Tests for tree-sitter parser."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from smart_search.core.exceptions import SyntaxParseError, UnsupportedLanguageError
from smart_search.parsing.models import Language
from smart_search.parsing.tree_sitter_parser import (
    TreeSitterParser,
    create_span_from_node,
    get_node_text,
)


class TestTreeSitterParser:
    """Tests for TreeSitterParser class."""

    @pytest.fixture
    def parser(self) -> TreeSitterParser:
        """Create a parser instance."""
        return TreeSitterParser()

    def test_initialization(self, parser: TreeSitterParser) -> None:
        """Test parser initialization."""
        assert parser is not None
        assert Language.PYTHON in parser.supported_languages

    def test_is_supported(self, parser: TreeSitterParser) -> None:
        """Test language support checking."""
        assert parser.is_supported(Language.PYTHON) is True

    def test_parse_source_python(self, parser: TreeSitterParser) -> None:
        """Test parsing Python source code."""
        source = "def hello(): pass"
        tree = parser.parse_source(source, Language.PYTHON)
        assert tree is not None
        assert tree.root_node is not None
        assert tree.root_node.type == "module"

    def test_parse_source_bytes(self, parser: TreeSitterParser) -> None:
        """Test parsing bytes input."""
        source = b"def hello(): pass"
        tree = parser.parse_source(source, Language.PYTHON)
        assert tree is not None

    def test_parse_source_unsupported_language(self, parser: TreeSitterParser) -> None:
        """Test parsing unsupported language raises error."""
        # Create a language that's not in supported list
        with pytest.raises(UnsupportedLanguageError):
            parser.parse_source("code", Language.JAVA)

    def test_parse_source_with_syntax_error(self, parser: TreeSitterParser) -> None:
        """Test parsing code with syntax errors."""
        source = "def hello( pass"  # Missing closing paren
        tree = parser.parse_source(source, Language.PYTHON)
        # Tree-sitter still returns a tree, but with error nodes
        assert tree.root_node.has_error is True

    def test_parse_file(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test parsing a file."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    pass\n")

        result = parser.parse_file(test_file)

        assert result is not None
        assert result.file_path == test_file
        assert result.language == Language.PYTHON
        assert result.content_hash is not None
        assert result.has_errors is False

    def test_parse_file_unsupported_extension(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test parsing file with unsupported extension."""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("some content")

        with pytest.raises(UnsupportedLanguageError):
            parser.parse_file(test_file)

    def test_parse_file_with_errors(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test parsing file with syntax errors."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def broken( pass")

        result = parser.parse_file(test_file)
        assert result.has_errors is True
        assert len(result.parse_errors) > 0

    def test_register_extractor(self, parser: TreeSitterParser) -> None:
        """Test registering a language extractor."""
        mock_extractor = MagicMock()
        parser.register_extractor(Language.PYTHON, mock_extractor)
        assert Language.PYTHON in parser._extractors

    def test_parse_file_with_extractor(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test parsing file with registered extractor."""
        from smart_search.parsing.extractors.python import PythonExtractor

        parser.register_extractor(Language.PYTHON, PythonExtractor())

        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''def hello():
    """Say hello."""
    print("Hello")
'''
        )

        result = parser.parse_file(test_file)
        assert result.unit_count >= 1

    def test_extract_imports(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting imports from file."""
        from smart_search.parsing.extractors.python import PythonExtractor

        parser.register_extractor(Language.PYTHON, PythonExtractor())

        test_file = tmp_path / "test.py"
        test_file.write_text(
            """import os
from pathlib import Path
import sys

def func():
    pass
"""
        )

        result = parser.parse_file(test_file)
        assert "os" in result.imports
        assert "pathlib" in result.imports
        assert "sys" in result.imports

    def test_extract_module_docstring(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting module docstring."""
        from smart_search.parsing.extractors.python import PythonExtractor

        parser.register_extractor(Language.PYTHON, PythonExtractor())

        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''"""This is a module docstring."""

def func():
    pass
'''
        )

        result = parser.parse_file(test_file)
        assert result.module_docstring == "This is a module docstring."


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_node_text(self) -> None:
        """Test get_node_text function."""
        parser = TreeSitterParser()
        source = b"def hello(): pass"
        tree = parser.parse_source(source, Language.PYTHON)

        # Get the function definition node
        func_node = tree.root_node.children[0]
        text = get_node_text(func_node, source)
        assert text == "def hello(): pass"

    def test_create_span_from_node(self) -> None:
        """Test create_span_from_node function."""
        parser = TreeSitterParser()
        source = b"def hello():\n    pass"
        tree = parser.parse_source(source, Language.PYTHON)

        func_node = tree.root_node.children[0]
        span = create_span_from_node(func_node)

        assert span.start_line == 1
        assert span.start.column == 0
        assert span.end_line >= 1


class TestParserEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def parser(self) -> TreeSitterParser:
        """Create parser with Python extractor."""
        p = TreeSitterParser()
        from smart_search.parsing.extractors.python import PythonExtractor

        p.register_extractor(Language.PYTHON, PythonExtractor())
        return p

    def test_empty_file(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test parsing empty file."""
        test_file = tmp_path / "empty.py"
        test_file.write_text("")

        result = parser.parse_file(test_file)
        assert result.unit_count == 0
        assert result.has_errors is False

    def test_comments_only(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test file with only comments."""
        test_file = tmp_path / "comments.py"
        test_file.write_text("# Just a comment\n# Another comment\n")

        result = parser.parse_file(test_file)
        assert result.unit_count == 0

    def test_complex_file(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test parsing complex Python file."""
        test_file = tmp_path / "complex.py"
        test_file.write_text(
            '''"""Module docstring."""

import os
from typing import List

class MyClass:
    """A class."""

    def __init__(self, value: int):
        """Initialize."""
        self.value = value

    def get_value(self) -> int:
        """Get the value."""
        return self.value

def standalone_func(x: int, y: int = 10) -> int:
    """Add two numbers."""
    return x + y

@decorator
def decorated_func():
    pass
'''
        )

        result = parser.parse_file(test_file)
        assert result.module_docstring == "Module docstring."
        assert result.unit_count >= 4  # class, __init__, get_value, standalone_func, decorated_func
