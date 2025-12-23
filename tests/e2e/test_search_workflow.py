"""End-to-end tests for search workflows."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def sample_python_project(tmp_path):
    """Create a sample Python project for testing."""
    project = tmp_path / "sample_project"
    project.mkdir()

    # Create main.py
    (project / "main.py").write_text('''
"""Main application module."""

from auth import login_user, logout_user
from utils import helper_function


def main():
    """Main entry point."""
    user = login_user("admin", "password")
    if user:
        result = helper_function(user)
        logout_user(user)
        return result
    return None


if __name__ == "__main__":
    main()
''')

    # Create auth.py
    (project / "auth.py").write_text('''
"""Authentication module."""


class User:
    """User class."""

    def __init__(self, username: str):
        self.username = username
        self.is_authenticated = False


def login_user(username: str, password: str) -> User | None:
    """Authenticate and return user."""
    if username and password:
        user = User(username)
        user.is_authenticated = True
        return user
    return None


def logout_user(user: User) -> bool:
    """Logout user."""
    if user and user.is_authenticated:
        user.is_authenticated = False
        return True
    return False
''')

    # Create utils.py
    (project / "utils.py").write_text('''
"""Utility functions."""


def helper_function(user):
    """Helper function for processing."""
    if user:
        return f"Processed: {user.username}"
    return None


def format_output(data: str) -> str:
    """Format output data."""
    return f"Output: {data}"
''')

    return project


@pytest.mark.e2e
class TestSearchWorkflow:
    """E2E tests for search workflow."""

    def test_parse_project_structure(self, sample_python_project):
        """Test parsing a project and extracting code units."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor

        parser = TreeSitterParser()
        parser.register_extractor(
            __import__("smart_search.parsing.models", fromlist=["Language"]).Language.PYTHON,
            PythonExtractor(),
        )

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        # Should have extracted multiple code units
        assert len(all_units) > 0

        # Should have functions and classes
        unit_types = {u.type.value for u in all_units}
        assert "function" in unit_types or "method" in unit_types

    def test_build_code_graph(self, sample_python_project):
        """Test building a code graph from parsed code units."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor
        from smart_search.parsing.models import Language
        from smart_search.graph.builder import GraphBuilder

        parser = TreeSitterParser()
        parser.register_extractor(Language.PYTHON, PythonExtractor())

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        # Build graph using GraphBuilder
        builder = GraphBuilder()
        graph = builder.build_from_units(all_units)

        # Should have nodes in graph
        assert graph.node_count > 0

    def test_full_parsing_to_graph_workflow(self, sample_python_project):
        """Test full workflow from parsing to graph building."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor
        from smart_search.parsing.models import Language
        from smart_search.graph.builder import GraphBuilder

        parser = TreeSitterParser()
        parser.register_extractor(Language.PYTHON, PythonExtractor())

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        # Build graph from units
        builder = GraphBuilder()
        graph = builder.build_from_units(all_units)

        # Verify graph structure
        assert graph.node_count >= len(all_units)


@pytest.mark.e2e
class TestNavigationWorkflow:
    """E2E tests for code navigation workflow."""

    def test_find_function_by_name(self, sample_python_project):
        """Test finding a function by name in the codebase."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor
        from smart_search.parsing.models import Language

        parser = TreeSitterParser()
        parser.register_extractor(Language.PYTHON, PythonExtractor())

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        # Find login_user function
        login_functions = [u for u in all_units if u.name == "login_user"]
        assert len(login_functions) == 1
        assert "auth.py" in str(login_functions[0].file_path)

    def test_find_class_definition(self, sample_python_project):
        """Test finding a class definition."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor
        from smart_search.parsing.models import Language

        parser = TreeSitterParser()
        parser.register_extractor(Language.PYTHON, PythonExtractor())

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        # Find User class
        user_classes = [u for u in all_units if u.name == "User"]
        assert len(user_classes) == 1
        assert user_classes[0].type.value == "class"


@pytest.mark.e2e
class TestImpactAnalysisWorkflow:
    """E2E tests for impact analysis workflow."""

    def test_analyze_function_impact(self, sample_python_project):
        """Test analyzing impact of changing a function."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor
        from smart_search.parsing.models import Language
        from smart_search.graph.builder import GraphBuilder
        from smart_search.graph.algorithms import GraphAlgorithms

        parser = TreeSitterParser()
        parser.register_extractor(Language.PYTHON, PythonExtractor())

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        builder = GraphBuilder()
        graph = builder.build_from_units(all_units)

        # Analyze impact of login_user
        algorithms = GraphAlgorithms(graph)

        # Find login_user node
        login_node = None
        for unit in all_units:
            if unit.name == "login_user":
                login_node = graph.get_node(unit.id)
                break

        # Get dependencies
        if login_node:
            descendants = algorithms.find_descendants(login_node.data.id, max_depth=2)
            # Impact analysis should work even with empty dependencies
            assert isinstance(descendants, list)


@pytest.mark.e2e
class TestGraphRAGWorkflow:
    """E2E tests for GraphRAG workflow."""

    def test_build_context_for_query(self, sample_python_project):
        """Test building context for a GraphRAG query."""
        from smart_search.parsing.tree_sitter_parser import TreeSitterParser
        from smart_search.parsing.extractors.python import PythonExtractor
        from smart_search.parsing.models import Language
        from smart_search.rag.prompts import PromptBuilder, CodeContext, PromptContext, QueryType

        parser = TreeSitterParser()
        parser.register_extractor(Language.PYTHON, PythonExtractor())

        all_units = []
        for py_file in sample_python_project.glob("*.py"):
            parsed = parser.parse_file(py_file)
            all_units.extend(parsed.units)

        # Build code contexts
        contexts = []
        for unit in all_units[:5]:  # Take first 5 units
            contexts.append(CodeContext(
                code_id=unit.id,
                name=unit.name,
                qualified_name=unit.qualified_name,
                code_type=unit.type.value,
                file_path=unit.file_path,
                line_start=unit.span.start.line,  # Use span.start.line
                line_end=unit.span.end.line,  # Use span.end.line
                content=unit.content[:500] if unit.content else "",
                language="python",
                relevance_score=0.9,
            ))

        # Build prompt context
        prompt_context = PromptContext(
            query="How does authentication work?",
            query_type=QueryType.LOCAL,
            code_contexts=contexts,
        )

        # Build prompts using PromptBuilder
        builder = PromptBuilder()
        full_prompt = builder.build_full_prompt(prompt_context)

        assert prompt_context is not None
        assert prompt_context.query == "How does authentication work?"
        assert "system" in full_prompt
        assert "user" in full_prompt


@pytest.mark.e2e
class TestIncrementalIndexingWorkflow:
    """E2E tests for incremental indexing workflow."""

    def test_detect_file_changes(self, sample_python_project):
        """Test detecting file changes in project."""
        from smart_search.indexing.hasher import ContentHasher

        hasher = ContentHasher()

        # Get initial hashes
        initial_hashes = {}
        for py_file in sample_python_project.glob("*.py"):
            initial_hashes[py_file] = hasher.hash_file(py_file)

        # Modify a file
        main_py = sample_python_project / "main.py"
        content = main_py.read_text()
        main_py.write_text(content + "\n# Modified\n")

        # Get new hashes
        new_hashes = {}
        for py_file in sample_python_project.glob("*.py"):
            new_hashes[py_file] = hasher.hash_file(py_file)

        # main.py should have different hash (compare FileHash objects)
        assert initial_hashes[main_py].content_hash != new_hashes[main_py].content_hash

        # Other files should have same hash
        for file_path, old_file_hash in initial_hashes.items():
            if "main.py" not in str(file_path):
                assert old_file_hash.content_hash == new_hashes[file_path].content_hash

    def test_compute_change_set(self, sample_python_project):
        """Test computing change set for incremental indexing."""
        from smart_search.indexing.hasher import ContentHasher, HashStore

        hasher = ContentHasher()
        store = HashStore()

        # Index initial state - use update() method with FileHash object
        for py_file in sample_python_project.glob("*.py"):
            file_hash = hasher.hash_file(py_file)
            store.update(file_hash)  # HashStore.update() takes FileHash

        # Modify a file
        main_py = sample_python_project / "main.py"
        content = main_py.read_text()
        main_py.write_text(content + "\n# Modified\n")

        # Get current hashes
        current_hashes = {}
        for py_file in sample_python_project.glob("*.py"):
            current_hashes[py_file] = hasher.hash_file(py_file)

        # Use store.compare() to compute change set
        comparison = store.compare(current_hashes)

        # Should have one modified file
        assert len(comparison.modified) == 1
        assert "main.py" in str(comparison.modified[0])
