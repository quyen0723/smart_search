"""Tests for RAG prompts module."""

from pathlib import Path

import pytest

from smart_search.rag.prompts import (
    CodeContext,
    CommunitySummary,
    ExplanationBuilder,
    PromptBuilder,
    PromptContext,
    QueryType,
    RelationshipPath,
)


class TestQueryType:
    """Tests for QueryType enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert QueryType.LOCAL.value == "local"
        assert QueryType.GLOBAL.value == "global"
        assert QueryType.DRIFT.value == "drift"

    def test_string_enum(self) -> None:
        """Test QueryType is a string enum."""
        assert isinstance(QueryType.LOCAL, str)
        assert QueryType.LOCAL == "local"


class TestCodeContext:
    """Tests for CodeContext dataclass."""

    @pytest.fixture
    def sample_context(self) -> CodeContext:
        """Create sample code context."""
        return CodeContext(
            code_id="test_id",
            name="test_function",
            qualified_name="module.test_function",
            code_type="function",
            file_path=Path("/src/module.py"),
            line_start=10,
            line_end=20,
            content="def test_function():\n    pass",
            language="python",
            docstring="A test function.",
            relevance_score=0.95,
        )

    def test_creation(self, sample_context: CodeContext) -> None:
        """Test creating a code context."""
        assert sample_context.code_id == "test_id"
        assert sample_context.name == "test_function"
        assert sample_context.qualified_name == "module.test_function"
        assert sample_context.code_type == "function"
        assert sample_context.line_start == 10
        assert sample_context.line_end == 20
        assert sample_context.language == "python"
        assert sample_context.relevance_score == 0.95

    def test_default_values(self) -> None:
        """Test default values."""
        context = CodeContext(
            code_id="id",
            name="name",
            qualified_name="name",
            code_type="function",
            file_path=Path("/test.py"),
            line_start=1,
            line_end=10,
            content="",
            language="python",
        )
        assert context.docstring is None
        assert context.relevance_score == 0.0

    def test_to_dict(self, sample_context: CodeContext) -> None:
        """Test conversion to dict."""
        d = sample_context.to_dict()

        assert d["code_id"] == "test_id"
        assert d["name"] == "test_function"
        assert d["qualified_name"] == "module.test_function"
        assert d["code_type"] == "function"
        assert d["file_path"] == "/src/module.py"
        assert d["line_start"] == 10
        assert d["line_end"] == 20
        assert d["language"] == "python"
        assert d["docstring"] == "A test function."
        assert d["relevance_score"] == 0.95

    def test_line_count_property(self, sample_context: CodeContext) -> None:
        """Test line_count property."""
        assert sample_context.line_count == 11  # 20 - 10 + 1

    def test_formatted_property(self, sample_context: CodeContext) -> None:
        """Test formatted property."""
        formatted = sample_context.formatted
        assert "test_function" in formatted
        assert "module.py" in formatted
        assert "def test_function():" in formatted


class TestRelationshipPath:
    """Tests for RelationshipPath dataclass."""

    def test_creation(self) -> None:
        """Test creating a relationship path."""
        path = RelationshipPath(
            source="func_a",
            target="func_b",
            path=[("func_a", "CALLS"), ("func_b", "CALLS")],
            description="func_a calls func_b",
        )

        assert path.source == "func_a"
        assert path.target == "func_b"
        assert len(path.path) == 2
        assert path.description == "func_a calls func_b"

    def test_default_values(self) -> None:
        """Test default values."""
        path = RelationshipPath(
            source="a",
            target="b",
            path=[],
        )
        assert path.description is None

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        path = RelationshipPath(
            source="func_a",
            target="func_b",
            path=[("func_a", "CALLS")],
            description="calls",
        )

        d = path.to_dict()

        assert d["source"] == "func_a"
        assert d["target"] == "func_b"
        assert d["path"] == [("func_a", "CALLS")]
        assert d["description"] == "calls"

    def test_depth_property(self) -> None:
        """Test depth property."""
        path = RelationshipPath(
            source="a",
            target="c",
            path=[("a", "CALLS"), ("b", "CALLS"), ("c", "CALLS")],
        )
        assert path.depth == 3


class TestCommunitySummary:
    """Tests for CommunitySummary dataclass."""

    def test_creation(self) -> None:
        """Test creating a community summary."""
        summary = CommunitySummary(
            community_id="comm_1",
            name="Authentication Module",
            description="Handles user authentication",
            key_components=["login", "logout", "verify"],
            size=10,
        )

        assert summary.community_id == "comm_1"
        assert summary.name == "Authentication Module"
        assert len(summary.key_components) == 3
        assert summary.size == 10

    def test_default_values(self) -> None:
        """Test default values."""
        summary = CommunitySummary(
            community_id="id",
            name="name",
            description="desc",
        )
        assert summary.key_components == []
        assert summary.size == 0

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        summary = CommunitySummary(
            community_id="comm_1",
            name="Test",
            description="Description",
            key_components=["a", "b"],
            size=5,
        )

        d = summary.to_dict()

        assert d["community_id"] == "comm_1"
        assert d["name"] == "Test"
        assert d["key_components"] == ["a", "b"]
        assert d["size"] == 5


class TestPromptContext:
    """Tests for PromptContext dataclass."""

    @pytest.fixture
    def sample_context(self) -> PromptContext:
        """Create sample prompt context."""
        code_context = CodeContext(
            code_id="id1",
            name="func",
            qualified_name="mod.func",
            code_type="function",
            file_path=Path("/test.py"),
            line_start=1,
            line_end=10,
            content="def func(): pass",
            language="python",
        )

        return PromptContext(
            query="How does func work?",
            query_type=QueryType.LOCAL,
            code_contexts=[code_context],
            relationships=[],
            communities=[],
        )

    def test_creation(self, sample_context: PromptContext) -> None:
        """Test creating prompt context."""
        assert sample_context.query == "How does func work?"
        assert sample_context.query_type == QueryType.LOCAL
        assert len(sample_context.code_contexts) == 1
        assert sample_context.max_tokens == 4000

    def test_default_values(self) -> None:
        """Test default values."""
        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
        )
        assert context.code_contexts == []
        assert context.relationships == []
        assert context.communities == []
        assert context.max_tokens == 4000

    def test_has_context_property(self, sample_context: PromptContext) -> None:
        """Test has_context property."""
        assert sample_context.has_context

        empty_context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
        )
        assert not empty_context.has_context

    def test_context_count_property(self, sample_context: PromptContext) -> None:
        """Test context_count property."""
        assert sample_context.context_count == 1


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create prompt builder."""
        return PromptBuilder()

    @pytest.fixture
    def sample_context(self) -> PromptContext:
        """Create sample context."""
        code_context = CodeContext(
            code_id="id1",
            name="calculate_total",
            qualified_name="billing.calculate_total",
            code_type="function",
            file_path=Path("/src/billing.py"),
            line_start=10,
            line_end=25,
            content="def calculate_total(items):\n    return sum(i.price for i in items)",
            language="python",
            docstring="Calculate total price of items.",
        )

        relationship = RelationshipPath(
            source="process_order",
            target="calculate_total",
            path=[("process_order", "CALLS")],
            description="process_order calls calculate_total",
        )

        return PromptContext(
            query="How is the order total calculated?",
            query_type=QueryType.LOCAL,
            code_contexts=[code_context],
            relationships=[relationship],
        )

    def test_build_system_prompt(self, builder: PromptBuilder) -> None:
        """Test building system prompt."""
        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
        )

        prompt = builder.build_system_prompt(context)

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "code" in prompt.lower()

    def test_build_user_prompt(
        self, builder: PromptBuilder, sample_context: PromptContext
    ) -> None:
        """Test building user prompt."""
        prompt = builder.build_user_prompt(sample_context)

        assert isinstance(prompt, str)
        assert sample_context.query in prompt
        assert "calculate_total" in prompt
        assert "billing.py" in prompt

    def test_build_full_prompt(
        self, builder: PromptBuilder, sample_context: PromptContext
    ) -> None:
        """Test building full prompt."""
        prompt = builder.build_full_prompt(sample_context)

        assert "system" in prompt
        assert "user" in prompt
        assert isinstance(prompt["system"], str)
        assert isinstance(prompt["user"], str)

    def test_format_code_context(self, builder: PromptBuilder) -> None:
        """Test formatting code context."""
        context = CodeContext(
            code_id="id",
            name="test_func",
            qualified_name="test_func",
            code_type="function",
            file_path=Path("/test.py"),
            line_start=1,
            line_end=5,
            content="def test_func(): pass",
            language="python",
        )

        formatted = builder._format_code_context(context)

        assert "test_func" in formatted
        assert "test.py" in formatted
        assert "def test_func()" in formatted

    def test_format_relationships(self, builder: PromptBuilder) -> None:
        """Test formatting relationships."""
        relationships = [
            RelationshipPath(
                source="a",
                target="b",
                path=[("a", "CALLS")],
                description="a calls b",
            )
        ]

        formatted = builder._format_relationships(relationships)

        assert "a" in formatted
        assert "b" in formatted
        assert "calls" in formatted.lower()

    def test_format_communities(self, builder: PromptBuilder) -> None:
        """Test formatting communities."""
        communities = [
            CommunitySummary(
                community_id="c1",
                name="Auth Module",
                description="Authentication logic",
                key_components=["login", "logout"],
                size=5,
            )
        ]

        formatted = builder._format_communities(communities)

        assert "Auth Module" in formatted
        assert "login" in formatted

    def test_global_query_system_prompt(self, builder: PromptBuilder) -> None:
        """Test system prompt for global queries."""
        context = PromptContext(
            query="test",
            query_type=QueryType.GLOBAL,
        )

        prompt = builder.build_system_prompt(context)

        # Global queries should emphasize architecture
        assert "architecture" in prompt.lower() or "pattern" in prompt.lower()

    def test_drift_query_system_prompt(self, builder: PromptBuilder) -> None:
        """Test system prompt for drift queries."""
        context = PromptContext(
            query="test",
            query_type=QueryType.DRIFT,
        )

        prompt = builder.build_system_prompt(context)

        # Should include comprehensive analysis
        assert len(prompt) > 0


class TestExplanationBuilder:
    """Tests for ExplanationBuilder."""

    @pytest.fixture
    def builder(self) -> ExplanationBuilder:
        """Create explanation builder."""
        return ExplanationBuilder()

    def test_explain_path_calls(self, builder: ExplanationBuilder) -> None:
        """Test explaining a CALLS relationship."""
        path = RelationshipPath(
            source="process_order",
            target="calculate_total",
            path=[("calculate_total", "CALLS")],
            description="process_order calls calculate_total",
        )

        explanation = builder.explain_path(path)

        assert "process_order" in explanation
        assert "calculate_total" in explanation
        assert "call" in explanation.lower()

    def test_explain_path_contains(self, builder: ExplanationBuilder) -> None:
        """Test explaining a CONTAINS relationship."""
        path = RelationshipPath(
            source="MyClass",
            target="my_method",
            path=[("my_method", "CONTAINS")],
        )

        explanation = builder.explain_path(path)

        assert "MyClass" in explanation
        assert "my_method" in explanation

    def test_explain_code_single(self, builder: ExplanationBuilder) -> None:
        """Test explaining a single code context."""
        context = CodeContext(
            code_id="id",
            name="calculate_sum",
            qualified_name="math.calculate_sum",
            code_type="function",
            file_path=Path("/math.py"),
            line_start=1,
            line_end=5,
            content="def calculate_sum(a, b):\n    return a + b",
            language="python",
            docstring="Calculate sum of two numbers.",
        )

        explanation = builder.explain_code(context)

        assert "calculate_sum" in explanation
        assert "function" in explanation.lower()

    def test_explain_code_multiple(self, builder: ExplanationBuilder) -> None:
        """Test explaining multiple code contexts."""
        contexts = [
            CodeContext(
                code_id="id1",
                name="func1",
                qualified_name="func1",
                code_type="function",
                file_path=Path("/a.py"),
                line_start=1,
                line_end=5,
                content="def func1(): pass",
                language="python",
            ),
            CodeContext(
                code_id="id2",
                name="func2",
                qualified_name="func2",
                code_type="function",
                file_path=Path("/b.py"),
                line_start=1,
                line_end=5,
                content="def func2(): pass",
                language="python",
            ),
        ]

        explanation = builder.explain_code(contexts)

        assert "func1" in explanation
        assert "func2" in explanation

    def test_format_location(self, builder: ExplanationBuilder) -> None:
        """Test formatting file location."""
        context = CodeContext(
            code_id="id",
            name="test",
            qualified_name="test",
            code_type="function",
            file_path=Path("/path/to/file.py"),
            line_start=10,
            line_end=20,
            content="",
            language="python",
        )

        location = builder._format_location(context)

        assert "file.py" in location
        assert "10" in location


class TestPromptContextMethods:
    """Tests for PromptContext methods."""

    def test_estimate_tokens(self) -> None:
        """Test token estimation."""
        context = PromptContext(
            query="How does this work?",
            query_type=QueryType.LOCAL,
            code_contexts=[
                CodeContext(
                    code_id="id1",
                    name="func",
                    qualified_name="func",
                    code_type="function",
                    file_path=Path("/test.py"),
                    line_start=1,
                    line_end=10,
                    content="def func(): pass",
                    language="python",
                )
            ],
        )

        tokens = context.estimate_tokens()
        assert tokens > 0

    def test_context_count_multiple(self) -> None:
        """Test context_count with multiple items."""
        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
            code_contexts=[
                CodeContext(
                    code_id=f"id{i}",
                    name=f"func{i}",
                    qualified_name=f"func{i}",
                    code_type="function",
                    file_path=Path(f"/test{i}.py"),
                    line_start=1,
                    line_end=5,
                    content="pass",
                    language="python",
                )
                for i in range(3)
            ],
            relationships=[
                RelationshipPath(source="a", target="b", path=[])
            ],
        )

        # 3 code contexts + 1 relationship
        assert context.context_count == 4


class TestPromptBuilderTruncation:
    """Tests for truncation in PromptBuilder."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create prompt builder."""
        return PromptBuilder()

    def test_truncate_to_max_tokens(self, builder: PromptBuilder) -> None:
        """Test truncate_to_max_tokens method."""
        contexts = [
            CodeContext(
                code_id=f"id{i}",
                name=f"func{i}",
                qualified_name=f"func{i}",
                code_type="function",
                file_path=Path(f"/test{i}.py"),
                line_start=1,
                line_end=100,
                content="x = 1\n" * 100,
                language="python",
                relevance_score=0.5 + i * 0.1,
            )
            for i in range(10)
        ]

        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
            code_contexts=contexts,
            max_tokens=500,
        )

        truncated = builder.truncate_to_max_tokens(context)

        # Should have fewer contexts after truncation
        assert len(truncated.code_contexts) <= len(contexts)

    def test_truncate_preserves_high_relevance(self, builder: PromptBuilder) -> None:
        """Test truncation preserves high relevance items."""
        high_relevance = CodeContext(
            code_id="high",
            name="high_func",
            qualified_name="high_func",
            code_type="function",
            file_path=Path("/high.py"),
            line_start=1,
            line_end=10,
            content="def high(): pass",
            language="python",
            relevance_score=0.99,
        )

        low_relevance = CodeContext(
            code_id="low",
            name="low_func",
            qualified_name="low_func",
            code_type="function",
            file_path=Path("/low.py"),
            line_start=1,
            line_end=100,
            content="x = 1\n" * 100,
            language="python",
            relevance_score=0.1,
        )

        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
            code_contexts=[low_relevance, high_relevance],
            max_tokens=200,
        )

        truncated = builder.truncate_to_max_tokens(context)

        # High relevance should be preserved if any contexts remain
        if truncated.code_contexts:
            assert any(c.code_id == "high" for c in truncated.code_contexts)


class TestPromptBuilderTokenLimit:
    """Tests for token limiting in PromptBuilder."""

    @pytest.fixture
    def builder(self) -> PromptBuilder:
        """Create prompt builder."""
        return PromptBuilder()

    def test_respects_max_tokens(self, builder: PromptBuilder) -> None:
        """Test that prompt respects max_tokens."""
        # Create many contexts
        contexts = [
            CodeContext(
                code_id=f"id{i}",
                name=f"function_{i}",
                qualified_name=f"module.function_{i}",
                code_type="function",
                file_path=Path(f"/src/file{i}.py"),
                line_start=1,
                line_end=100,
                content="def function(): " + "x = 1\n" * 100,
                language="python",
            )
            for i in range(50)
        ]

        context = PromptContext(
            query="How does this work?",
            query_type=QueryType.LOCAL,
            code_contexts=contexts,
            max_tokens=1000,  # Small limit
        )

        prompt = builder.build_user_prompt(context)

        # Should not include all 50 contexts
        assert prompt.count("function_") < 50

    def test_prioritizes_high_relevance(self, builder: PromptBuilder) -> None:
        """Test that high relevance contexts are prioritized."""
        contexts = [
            CodeContext(
                code_id="low",
                name="low_relevance",
                qualified_name="low_relevance",
                code_type="function",
                file_path=Path("/low.py"),
                line_start=1,
                line_end=5,
                content="def low_relevance(): pass",
                language="python",
                relevance_score=0.1,
            ),
            CodeContext(
                code_id="high",
                name="high_relevance",
                qualified_name="high_relevance",
                code_type="function",
                file_path=Path("/high.py"),
                line_start=1,
                line_end=5,
                content="def high_relevance(): pass",
                language="python",
                relevance_score=0.9,
            ),
        ]

        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
            code_contexts=contexts,
            max_tokens=500,
        )

        prompt = builder.build_user_prompt(context)

        # High relevance should appear
        assert "high_relevance" in prompt
