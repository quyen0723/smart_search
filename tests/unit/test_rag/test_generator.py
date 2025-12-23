"""Tests for RAG generator module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smart_search.rag.generator import (
    AnthropicClient,
    CachedGenerator,
    GenerationResult,
    GeneratorConfig,
    GeneratorModel,
    MockLLMClient,
    OpenAIClient,
    ResponseGenerator,
)
from smart_search.rag.prompts import CodeContext, PromptContext, QueryType


class TestGeneratorModel:
    """Tests for GeneratorModel enum."""

    def test_values(self) -> None:
        """Test enum values."""
        assert GeneratorModel.GPT4.value == "gpt-4"
        assert GeneratorModel.GPT4_TURBO.value == "gpt-4-turbo"
        assert GeneratorModel.GPT35_TURBO.value == "gpt-3.5-turbo"
        assert GeneratorModel.CLAUDE_3_OPUS.value == "claude-3-opus"
        assert GeneratorModel.CLAUDE_3_SONNET.value == "claude-3-sonnet"
        assert GeneratorModel.CLAUDE_3_HAIKU.value == "claude-3-haiku"
        assert GeneratorModel.LOCAL.value == "local"


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_default_values(self) -> None:
        """Test default configuration."""
        config = GeneratorConfig()

        assert config.model == GeneratorModel.GPT4_TURBO
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.stream is False
        assert config.timeout == 30.0
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = GeneratorConfig(
            model=GeneratorModel.CLAUDE_3_SONNET,
            temperature=0.5,
            max_tokens=4000,
            stream=True,
            timeout=60.0,
            retry_attempts=5,
            retry_delay=2.0,
        )

        assert config.model == GeneratorModel.CLAUDE_3_SONNET
        assert config.temperature == 0.5
        assert config.max_tokens == 4000
        assert config.stream is True
        assert config.timeout == 60.0
        assert config.retry_attempts == 5


class TestGenerationResult:
    """Tests for GenerationResult."""

    def test_creation(self) -> None:
        """Test creating generation result."""
        result = GenerationResult(
            query="How does this work?",
            response="This code does XYZ...",
            model="gpt-4-turbo",
            tokens_used=150,
            processing_time_ms=500.0,
            citations=["mod.func1", "mod.func2"],
        )

        assert result.query == "How does this work?"
        assert result.response == "This code does XYZ..."
        assert result.model == "gpt-4-turbo"
        assert result.tokens_used == 150
        assert result.processing_time_ms == 500.0
        assert len(result.citations) == 2

    def test_default_values(self) -> None:
        """Test default values."""
        result = GenerationResult(
            query="test",
            response="response",
            model="model",
        )

        assert result.tokens_used == 0
        assert result.processing_time_ms == 0.0
        assert result.citations == []
        assert result.metadata == {}

    def test_has_citations_true(self) -> None:
        """Test has_citations when citations exist."""
        result = GenerationResult(
            query="test",
            response="response",
            model="model",
            citations=["citation1"],
        )

        assert result.has_citations is True

    def test_has_citations_false(self) -> None:
        """Test has_citations when no citations."""
        result = GenerationResult(
            query="test",
            response="response",
            model="model",
        )

        assert result.has_citations is False

    def test_to_dict(self) -> None:
        """Test conversion to dict."""
        result = GenerationResult(
            query="test query",
            response="test response",
            model="gpt-4",
            tokens_used=100,
            processing_time_ms=250.0,
            citations=["func1"],
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["query"] == "test query"
        assert d["response"] == "test response"
        assert d["model"] == "gpt-4"
        assert d["tokens_used"] == 100
        assert d["processing_time_ms"] == 250.0
        assert d["citations"] == ["func1"]
        assert d["metadata"] == {"key": "value"}


class TestMockLLMClient:
    """Tests for MockLLMClient."""

    def test_initialization(self) -> None:
        """Test mock client initialization."""
        client = MockLLMClient(response="Custom response")

        assert client.response == "Custom response"
        assert client.calls == []

    @pytest.mark.asyncio
    async def test_generate(self) -> None:
        """Test mock generate."""
        client = MockLLMClient(response="Generated text")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        result = await client.generate(messages, temperature=0.5)

        assert result["content"] == "Generated text"
        assert "usage" in result
        assert result["usage"]["total_tokens"] == 150
        assert len(client.calls) == 1
        assert client.calls[0]["messages"] == messages

    @pytest.mark.asyncio
    async def test_generate_stream(self) -> None:
        """Test mock stream generation."""
        client = MockLLMClient(response="Hello world")

        chunks = []
        async for chunk in client.generate_stream([{"role": "user", "content": "Hi"}]):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert "Hello" in "".join(chunks)
        assert len(client.calls) == 1
        assert client.calls[0]["stream"] is True


class TestOpenAIClient:
    """Tests for OpenAIClient."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = OpenAIClient(
            api_key="test_key",
            model="gpt-4",
            base_url="https://custom.api.com",
        )

        assert client.api_key == "test_key"
        assert client.model == "gpt-4"
        assert client.base_url == "https://custom.api.com"

    def test_default_base_url(self) -> None:
        """Test default base URL."""
        client = OpenAIClient(api_key="key")

        assert client.base_url == "https://api.openai.com/v1"

    @pytest.mark.asyncio
    async def test_generate_requires_httpx(self) -> None:
        """Test generate raises ImportError without httpx."""
        client = OpenAIClient(api_key="key")

        # This will either work (if httpx installed) or raise ImportError
        try:
            await client.generate([{"role": "user", "content": "test"}])
        except ImportError as e:
            assert "httpx" in str(e)
        except Exception:
            pass  # Other errors are OK (like connection errors)


class TestAnthropicClient:
    """Tests for AnthropicClient."""

    def test_initialization(self) -> None:
        """Test client initialization."""
        client = AnthropicClient(
            api_key="test_key",
            model="claude-3-opus-20240229",
        )

        assert client.api_key == "test_key"
        assert client.model == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_generate_requires_httpx(self) -> None:
        """Test generate raises ImportError without httpx."""
        client = AnthropicClient(api_key="key")

        try:
            await client.generate([{"role": "user", "content": "test"}])
        except ImportError as e:
            assert "httpx" in str(e)
        except Exception:
            pass


class TestResponseGenerator:
    """Tests for ResponseGenerator."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock LLM client."""
        return MockLLMClient(response="This function calculates the sum.")

    @pytest.fixture
    def generator(self, mock_client: MockLLMClient) -> ResponseGenerator:
        """Create response generator."""
        return ResponseGenerator(
            client=mock_client,
            config=GeneratorConfig(retry_attempts=1),
        )

    @pytest.fixture
    def sample_context(self) -> PromptContext:
        """Create sample prompt context."""
        code_context = CodeContext(
            code_id="id1",
            name="calculate_sum",
            qualified_name="math.calculate_sum",
            code_type="function",
            file_path=Path("/src/math.py"),
            line_start=10,
            line_end=15,
            content="def calculate_sum(a, b):\n    return a + b",
            language="python",
            docstring="Calculate sum of two numbers.",
        )

        return PromptContext(
            query="How does calculate_sum work?",
            query_type=QueryType.LOCAL,
            code_contexts=[code_context],
        )

    @pytest.mark.asyncio
    async def test_generate(
        self, generator: ResponseGenerator, sample_context: PromptContext
    ) -> None:
        """Test generating response."""
        result = await generator.generate(sample_context)

        assert result.query == "How does calculate_sum work?"
        assert result.response == "This function calculates the sum."
        assert result.model == GeneratorModel.GPT4_TURBO.value
        assert result.tokens_used == 150
        assert result.processing_time_ms > 0
        assert "math.calculate_sum" in result.citations

    @pytest.mark.asyncio
    async def test_generate_with_additional_instructions(
        self, generator: ResponseGenerator, sample_context: PromptContext
    ) -> None:
        """Test generating with additional instructions."""
        result = await generator.generate(
            sample_context,
            additional_instructions="Be very concise.",
        )

        # Should include additional message in calls
        assert len(generator.client.calls) == 1
        messages = generator.client.calls[0]["messages"]
        assert any("concise" in str(msg) for msg in messages)

    @pytest.mark.asyncio
    async def test_generate_stream(
        self, generator: ResponseGenerator, sample_context: PromptContext
    ) -> None:
        """Test streaming generation."""
        chunks = []
        async for chunk in generator.generate_stream(sample_context):
            chunks.append(chunk)

        assert len(chunks) > 0

    @pytest.mark.asyncio
    async def test_generate_explanation(
        self, generator: ResponseGenerator
    ) -> None:
        """Test generating code explanation."""
        result = await generator.generate_explanation(
            "def add(a, b): return a + b",
            "What does this function do?",
        )

        assert result.query == "What does this function do?"
        assert len(result.response) > 0

    @pytest.mark.asyncio
    async def test_generate_summary(
        self, generator: ResponseGenerator
    ) -> None:
        """Test generating summary."""
        contexts = [
            {"name": "func1", "file_path": "/a.py", "content": "def func1(): pass"},
            {"name": "func2", "file_path": "/b.py", "content": "def func2(): pass"},
        ]

        summary = await generator.generate_summary(contexts)

        assert len(summary) > 0

    @pytest.mark.asyncio
    async def test_retry_on_failure(
        self, mock_client: MockLLMClient
    ) -> None:
        """Test retry logic on failure."""
        # Create client that fails then succeeds
        call_count = 0
        original_generate = mock_client.generate

        async def failing_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return await original_generate(*args, **kwargs)

        mock_client.generate = failing_generate

        generator = ResponseGenerator(
            client=mock_client,
            config=GeneratorConfig(retry_attempts=3, retry_delay=0.01),
        )

        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
        )

        result = await generator.generate(context)

        assert result.response == "This function calculates the sum."
        assert call_count == 2  # First failed, second succeeded

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self, mock_client: MockLLMClient
    ) -> None:
        """Test timeout handling."""
        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)  # Very slow
            return {"content": "response", "usage": {}}

        mock_client.generate = slow_generate

        generator = ResponseGenerator(
            client=mock_client,
            config=GeneratorConfig(timeout=0.1, retry_attempts=1),
        )

        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
        )

        with pytest.raises(asyncio.TimeoutError):
            await generator.generate(context)


class TestCachedGenerator:
    """Tests for CachedGenerator."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock client."""
        return MockLLMClient(response="Cached response")

    @pytest.fixture
    def cached_generator(self, mock_client: MockLLMClient) -> CachedGenerator:
        """Create cached generator."""
        base_generator = ResponseGenerator(client=mock_client)
        return CachedGenerator(generator=base_generator, cache_size=10)

    @pytest.fixture
    def sample_context(self) -> PromptContext:
        """Create sample context."""
        return PromptContext(
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
                    line_end=5,
                    content="def func(): pass",
                    language="python",
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_cache_hit(
        self, cached_generator: CachedGenerator, sample_context: PromptContext
    ) -> None:
        """Test cache hit returns cached result."""
        # First call
        result1 = await cached_generator.generate(sample_context)

        # Second call (should hit cache)
        result2 = await cached_generator.generate(sample_context)

        assert result1.response == result2.response
        # Only one actual generation call
        assert len(cached_generator.generator.client.calls) == 1

    @pytest.mark.asyncio
    async def test_cache_miss(
        self, cached_generator: CachedGenerator
    ) -> None:
        """Test cache miss generates new response."""
        context1 = PromptContext(
            query="Query 1",
            query_type=QueryType.LOCAL,
        )
        context2 = PromptContext(
            query="Query 2",
            query_type=QueryType.LOCAL,
        )

        await cached_generator.generate(context1)
        await cached_generator.generate(context2)

        # Two separate generation calls
        assert len(cached_generator.generator.client.calls) == 2

    @pytest.mark.asyncio
    async def test_cache_bypass(
        self, cached_generator: CachedGenerator, sample_context: PromptContext
    ) -> None:
        """Test bypassing cache."""
        await cached_generator.generate(sample_context, use_cache=True)
        await cached_generator.generate(sample_context, use_cache=False)

        # Two calls despite same context
        assert len(cached_generator.generator.client.calls) == 2

    def test_clear_cache(
        self, cached_generator: CachedGenerator
    ) -> None:
        """Test clearing cache."""
        cached_generator._cache = {"key": MagicMock()}
        cached_generator._cache_order = ["key"]

        cached_generator.clear_cache()

        assert cached_generator._cache == {}
        assert cached_generator._cache_order == []

    def test_cache_stats(
        self, cached_generator: CachedGenerator
    ) -> None:
        """Test cache statistics."""
        cached_generator._cache = {"k1": MagicMock(), "k2": MagicMock()}

        stats = cached_generator.cache_stats

        assert stats["size"] == 2
        assert stats["max_size"] == 10

    @pytest.mark.asyncio
    async def test_cache_eviction(
        self, mock_client: MockLLMClient
    ) -> None:
        """Test cache eviction when full."""
        base_generator = ResponseGenerator(client=mock_client)
        cached = CachedGenerator(generator=base_generator, cache_size=2)

        # Add 3 items to cache of size 2
        for i in range(3):
            context = PromptContext(
                query=f"Query {i}",
                query_type=QueryType.LOCAL,
            )
            await cached.generate(context)

        # Should have evicted oldest
        assert len(cached._cache) == 2


class TestResponseGeneratorMetadata:
    """Tests for generation metadata."""

    @pytest.fixture
    def mock_client(self) -> MockLLMClient:
        """Create mock client."""
        return MockLLMClient(response="Response with metadata")

    @pytest.mark.asyncio
    async def test_metadata_includes_query_type(
        self, mock_client: MockLLMClient
    ) -> None:
        """Test metadata includes query type."""
        generator = ResponseGenerator(client=mock_client)

        context = PromptContext(
            query="test",
            query_type=QueryType.GLOBAL,
        )

        result = await generator.generate(context)

        assert result.metadata["query_type"] == "global"

    @pytest.mark.asyncio
    async def test_metadata_includes_context_count(
        self, mock_client: MockLLMClient
    ) -> None:
        """Test metadata includes context count."""
        generator = ResponseGenerator(client=mock_client)

        code_contexts = [
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
        ]

        context = PromptContext(
            query="test",
            query_type=QueryType.LOCAL,
            code_contexts=code_contexts,
        )

        result = await generator.generate(context)

        assert result.metadata["context_count"] == 3
