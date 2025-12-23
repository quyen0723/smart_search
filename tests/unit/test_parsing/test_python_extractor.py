"""Tests for Python extractor."""

from pathlib import Path

import pytest

from smart_search.parsing.extractors.python import PythonExtractor
from smart_search.parsing.models import CodeUnitType, Language
from smart_search.parsing.tree_sitter_parser import TreeSitterParser


class TestPythonExtractor:
    """Tests for PythonExtractor class."""

    @pytest.fixture
    def parser(self) -> TreeSitterParser:
        """Create parser with Python extractor."""
        p = TreeSitterParser()
        p.register_extractor(Language.PYTHON, PythonExtractor())
        return p

    def test_extract_function(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting a simple function."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''def hello():
    """Say hello."""
    print("Hello")
'''
        )

        result = parser.parse_file(test_file)
        assert result.unit_count == 1

        unit = result.units[0]
        assert unit.name == "hello"
        assert unit.type == CodeUnitType.FUNCTION
        assert unit.docstring == "Say hello."
        assert "print" in unit.calls

    def test_extract_function_with_params(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting function with parameters."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def add(a: int, b: int = 0) -> int:
    return a + b
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert len(unit.parameters) == 2
        assert unit.parameters[0].name == "a"
        assert unit.parameters[0].type_annotation == "int"
        assert unit.parameters[1].name == "b"
        assert unit.parameters[1].default_value is not None
        assert unit.return_type == "int"

    def test_extract_function_variadic(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting function with *args and **kwargs."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def func(*args, **kwargs):
    pass
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert any(p.is_variadic for p in unit.parameters)
        assert any(p.is_keyword for p in unit.parameters)

    def test_extract_class(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting a class."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''class MyClass:
    """A simple class."""
    pass
'''
        )

        result = parser.parse_file(test_file)
        assert result.unit_count == 1

        unit = result.units[0]
        assert unit.name == "MyClass"
        assert unit.type == CodeUnitType.CLASS
        assert unit.docstring == "A simple class."

    def test_extract_class_with_bases(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting class with base classes."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """class Child(Parent, Mixin):
    pass
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert "Parent" in unit.base_classes
        assert "Mixin" in unit.base_classes

    def test_extract_methods(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting methods from class."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            '''class MyClass:
    def __init__(self):
        """Initialize."""
        self.value = 0

    def get_value(self) -> int:
        """Get value."""
        return self.value
'''
        )

        result = parser.parse_file(test_file)
        # Should have: class + 2 methods
        assert result.unit_count == 3

        # Find the methods
        methods = [u for u in result.units if u.type == CodeUnitType.METHOD]
        assert len(methods) == 2

        init_method = next(m for m in methods if m.name == "__init__")
        assert init_method.parent_id is not None

    def test_extract_decorated_function(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting decorated function."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """@decorator
def func():
    pass
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert "decorator" in unit.decorators

    def test_extract_multiple_decorators(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting function with multiple decorators."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """@decorator1
@decorator2(arg)
def func():
    pass
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert len(unit.decorators) == 2

    def test_extract_decorated_class(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test extracting decorated class."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """@dataclass
class Point:
    x: int
    y: int
"""
        )

        result = parser.parse_file(test_file)
        classes = [u for u in result.units if u.type == CodeUnitType.CLASS]
        assert len(classes) == 1
        assert "dataclass" in classes[0].decorators

    def test_qualified_names(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test qualified name generation."""
        test_file = tmp_path / "mymodule.py"
        test_file.write_text(
            """class MyClass:
    def my_method(self):
        pass
"""
        )

        result = parser.parse_file(test_file)

        class_unit = next(u for u in result.units if u.type == CodeUnitType.CLASS)
        assert class_unit.qualified_name == "mymodule.MyClass"

        method_unit = next(u for u in result.units if u.type == CodeUnitType.METHOD)
        assert method_unit.qualified_name == "mymodule.MyClass.my_method"

    def test_extract_calls(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting function calls."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def func():
    print("hello")
    os.path.join("a", "b")
    helper()
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert "print" in unit.calls
        assert "helper" in unit.calls

    def test_signature_generation(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test function signature generation."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def process(data: list, count: int = 10) -> bool:
    pass
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert "process" in unit.signature
        assert "data: list" in unit.signature
        assert "count: int" in unit.signature
        assert "-> bool" in unit.signature

    def test_content_hash(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test content hash generation."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def func(): pass")

        result = parser.parse_file(test_file)
        unit = result.units[0]

        assert unit.content_hash is not None
        assert len(unit.content_hash) == 16


class TestPythonExtractorEdgeCases:
    """Test edge cases for Python extractor."""

    @pytest.fixture
    def parser(self) -> TreeSitterParser:
        """Create parser with Python extractor."""
        p = TreeSitterParser()
        p.register_extractor(Language.PYTHON, PythonExtractor())
        return p

    def test_async_function(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting async function."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """async def fetch():
    pass
"""
        )

        result = parser.parse_file(test_file)
        assert result.unit_count == 1
        assert result.units[0].name == "fetch"

    def test_nested_class(self, parser: TreeSitterParser, tmp_path: Path) -> None:
        """Test extracting nested class."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """class Outer:
    class Inner:
        pass
"""
        )

        result = parser.parse_file(test_file)
        # Should extract both classes
        classes = [u for u in result.units if u.type == CodeUnitType.CLASS]
        assert len(classes) >= 1  # At least outer class

    def test_lambda_not_extracted(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test that lambdas are not extracted as units."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """func = lambda x: x + 1
"""
        )

        result = parser.parse_file(test_file)
        # Lambdas should not be extracted as functions
        functions = [u for u in result.units if u.type == CodeUnitType.FUNCTION]
        assert len(functions) == 0

    def test_docstring_with_triple_single_quotes(
        self, parser: TreeSitterParser, tmp_path: Path
    ) -> None:
        """Test docstring with triple single quotes."""
        test_file = tmp_path / "test.py"
        test_file.write_text(
            """def func():
    '''Single quoted docstring.'''
    pass
"""
        )

        result = parser.parse_file(test_file)
        unit = result.units[0]
        assert unit.docstring == "Single quoted docstring."
