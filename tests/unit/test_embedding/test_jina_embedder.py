"""Tests for Jina embedder."""

import pytest

from smart_search.core.exceptions import EmbeddingGenerationError, ModelLoadError
from smart_search.embedding.jina_embedder import (
    BaseEmbedder,
    JinaEmbedder,
    MockEmbedder,
    create_embedder,
)
from smart_search.embedding.models import EmbeddingConfig


class TestMockEmbedder:
    """Tests for MockEmbedder."""

    @pytest.fixture
    def embedder(self) -> MockEmbedder:
        """Create a mock embedder."""
        return MockEmbedder()

    @pytest.fixture
    def config(self) -> EmbeddingConfig:
        """Create embedding config."""
        return EmbeddingConfig(dimensions=128)

    @pytest.mark.asyncio
    async def test_embed_single_text(self, embedder: MockEmbedder) -> None:
        """Test embedding a single text."""
        result = await embedder.embed("hello world")
        assert result.text == "hello world"
        assert len(result.embedding) == 1024  # Default dimensions
        assert result.model == "mock-embedder"
        assert result.token_count == 2  # Two words

    @pytest.mark.asyncio
    async def test_embed_deterministic(self, embedder: MockEmbedder) -> None:
        """Test that embeddings are deterministic."""
        result1 = await embedder.embed("test text")
        result2 = await embedder.embed("test text")
        assert result1.embedding == result2.embedding

    @pytest.mark.asyncio
    async def test_embed_different_texts(self, embedder: MockEmbedder) -> None:
        """Test different texts produce different embeddings."""
        result1 = await embedder.embed("hello")
        result2 = await embedder.embed("world")
        assert result1.embedding != result2.embedding

    @pytest.mark.asyncio
    async def test_embed_batch(self, embedder: MockEmbedder) -> None:
        """Test batch embedding."""
        texts = ["hello", "world", "test"]
        result = await embedder.embed_batch(texts)
        assert result.count == 3
        assert result.model == "mock-embedder"
        assert len(result.embeddings) == 3

    @pytest.mark.asyncio
    async def test_embed_empty_batch(self, embedder: MockEmbedder) -> None:
        """Test empty batch."""
        result = await embedder.embed_batch([])
        assert result.count == 0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_custom_dimensions(self, config: EmbeddingConfig) -> None:
        """Test custom dimensions."""
        embedder = MockEmbedder(config)
        result = await embedder.embed("test")
        assert len(result.embedding) == 128

    @pytest.mark.asyncio
    async def test_normalized_embeddings(self) -> None:
        """Test embeddings are normalized by default."""
        embedder = MockEmbedder(EmbeddingConfig(normalize=True))
        result = await embedder.embed("test")
        norm = sum(v**2 for v in result.embedding) ** 0.5
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_unnormalized_embeddings(self) -> None:
        """Test unnormalized embeddings."""
        embedder = MockEmbedder(EmbeddingConfig(normalize=False, dimensions=64))
        result = await embedder.embed("test")
        norm = sum(v**2 for v in result.embedding) ** 0.5
        # Unnormalized should not be exactly 1.0
        assert len(result.embedding) == 64

    @pytest.mark.asyncio
    async def test_call_count(self, embedder: MockEmbedder) -> None:
        """Test call count tracking."""
        assert embedder.call_count == 0
        await embedder.embed("test1")
        assert embedder.call_count == 1
        await embedder.embed("test2")
        assert embedder.call_count == 2

    @pytest.mark.asyncio
    async def test_batch_increments_call_count(self, embedder: MockEmbedder) -> None:
        """Test batch embedding increments call count per item."""
        await embedder.embed_batch(["a", "b", "c"])
        assert embedder.call_count == 3

    @pytest.mark.asyncio
    async def test_close(self, embedder: MockEmbedder) -> None:
        """Test close is no-op."""
        await embedder.close()  # Should not raise


class TestJinaEmbedder:
    """Tests for JinaEmbedder initialization."""

    def test_requires_api_key(self) -> None:
        """Test that API key is required."""
        with pytest.raises(ModelLoadError):
            JinaEmbedder("")

    def test_create_with_api_key(self) -> None:
        """Test creation with API key."""
        embedder = JinaEmbedder("test-api-key")
        assert embedder.api_key == "test-api-key"
        assert embedder.config is not None

    def test_create_with_custom_config(self) -> None:
        """Test creation with custom config."""
        config = EmbeddingConfig(dimensions=512)
        embedder = JinaEmbedder("test-key", config)
        assert embedder.config.dimensions == 512

    def test_not_initialized_initially(self) -> None:
        """Test embedder is not initialized until used."""
        embedder = JinaEmbedder("test-key")
        assert embedder._initialized is False
        assert embedder._client is None


class TestCreateEmbedder:
    """Tests for create_embedder factory."""

    def test_create_mock_embedder(self) -> None:
        """Test creating mock embedder."""
        embedder = create_embedder(use_mock=True)
        assert isinstance(embedder, MockEmbedder)

    def test_create_mock_with_config(self) -> None:
        """Test creating mock embedder with config."""
        config = EmbeddingConfig(dimensions=256)
        embedder = create_embedder(config=config, use_mock=True)
        assert isinstance(embedder, MockEmbedder)
        assert embedder.config.dimensions == 256

    def test_create_jina_embedder(self) -> None:
        """Test creating Jina embedder."""
        embedder = create_embedder(api_key="test-key")
        assert isinstance(embedder, JinaEmbedder)

    def test_requires_api_key_for_real_embedder(self) -> None:
        """Test API key required for real embedder."""
        with pytest.raises(ModelLoadError):
            create_embedder(api_key=None, use_mock=False)


class TestBaseEmbedder:
    """Tests for BaseEmbedder interface."""

    def test_is_abstract(self) -> None:
        """Test BaseEmbedder is abstract."""
        with pytest.raises(TypeError):
            BaseEmbedder()  # type: ignore


class TestJinaEmbedderAsync:
    """Async tests for JinaEmbedder without actual API calls."""

    @pytest.mark.asyncio
    async def test_embed_empty_text_raises(self) -> None:
        """Test embedding empty text raises error."""
        embedder = JinaEmbedder("test-key")
        with pytest.raises(EmbeddingGenerationError):
            await embedder.embed("")

    @pytest.mark.asyncio
    async def test_embed_whitespace_raises(self) -> None:
        """Test embedding whitespace raises error."""
        embedder = JinaEmbedder("test-key")
        with pytest.raises(EmbeddingGenerationError):
            await embedder.embed("   ")

    @pytest.mark.asyncio
    async def test_embed_batch_empty_returns_empty(self) -> None:
        """Test empty batch returns empty result."""
        embedder = JinaEmbedder("test-key")
        result = await embedder.embed_batch([])
        assert result.count == 0
        assert result.total_tokens == 0

    @pytest.mark.asyncio
    async def test_embed_batch_all_empty_raises(self) -> None:
        """Test batch of all empty texts raises error."""
        embedder = JinaEmbedder("test-key")
        with pytest.raises(EmbeddingGenerationError):
            await embedder.embed_batch(["", "   ", "\n"])

    @pytest.mark.asyncio
    async def test_close(self) -> None:
        """Test close without initialization."""
        embedder = JinaEmbedder("test-key")
        await embedder.close()
        assert embedder._client is None

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        """Test async context manager."""
        async with JinaEmbedder("test-key") as embedder:
            assert embedder._initialized is True
        assert embedder._client is None
