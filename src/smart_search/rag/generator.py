"""Response generator for GraphRAG.

Generates natural language responses from retrieved context using LLM.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from smart_search.rag.prompts import PromptBuilder, PromptContext, QueryType
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class GeneratorModel(str, Enum):
    """Supported LLM models."""

    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    GPT35_TURBO = "gpt-3.5-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    LOCAL = "local"


@dataclass
class GeneratorConfig:
    """Configuration for response generator.

    Attributes:
        model: LLM model to use.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum response tokens.
        stream: Whether to stream responses.
        timeout: Request timeout in seconds.
        retry_attempts: Number of retry attempts.
        retry_delay: Delay between retries in seconds.
    """

    model: GeneratorModel = GeneratorModel.GPT4_TURBO
    temperature: float = 0.7
    max_tokens: int = 2000
    stream: bool = False
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class GenerationResult:
    """Result of response generation.

    Attributes:
        query: Original query.
        response: Generated response text.
        model: Model used for generation.
        tokens_used: Number of tokens used.
        processing_time_ms: Time taken in milliseconds.
        citations: Referenced code contexts.
        metadata: Additional metadata.
    """

    query: str
    response: str
    model: str
    tokens_used: int = 0
    processing_time_ms: float = 0.0
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_citations(self) -> bool:
        """Whether response has citations."""
        return len(self.citations) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "processing_time_ms": self.processing_time_ms,
            "citations": self.citations,
            "metadata": self.metadata,
        }


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate response from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional generation parameters.

        Returns:
            Response dict with 'content' and 'usage'.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response from messages.

        Args:
            messages: List of message dicts.
            **kwargs: Additional parameters.

        Yields:
            Response chunks.
        """
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo",
        base_url: str | None = None,
    ) -> None:
        """Initialize OpenAI client.

        Args:
            api_key: OpenAI API key.
            model: Model to use.
            base_url: Optional custom base URL.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create async client."""
        if self._client is None:
            try:
                import httpx

                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=60.0,
                )
            except ImportError:
                raise ImportError("httpx is required for OpenAI client")
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate response using OpenAI API."""
        client = await self._get_client()

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
        }

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
        }

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response using OpenAI API."""
        client = await self._get_client()

        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "stream": True,
        }

        async with client.stream("POST", "/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json

                        chunk = json.loads(data)
                        if content := chunk["choices"][0]["delta"].get("content"):
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class AnthropicClient(LLMClient):
    """Anthropic Claude API client."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
    ) -> None:
        """Initialize Anthropic client.

        Args:
            api_key: Anthropic API key.
            model: Model to use.
        """
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    async def _get_client(self) -> Any:
        """Get or create async client."""
        if self._client is None:
            try:
                import httpx

                self._client = httpx.AsyncClient(
                    base_url="https://api.anthropic.com/v1",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    timeout=60.0,
                )
            except ImportError:
                raise ImportError("httpx is required for Anthropic client")
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate response using Anthropic API."""
        client = await self._get_client()

        # Convert messages format
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        payload = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "messages": user_messages,
        }
        if system_msg:
            payload["system"] = system_msg

        response = await client.post("/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        return {
            "content": data["content"][0]["text"],
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0)
                ),
            },
        }

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream response using Anthropic API."""
        client = await self._get_client()

        # Convert messages format
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)

        payload = {
            "model": kwargs.get("model", self.model),
            "max_tokens": kwargs.get("max_tokens", 2000),
            "messages": user_messages,
            "stream": True,
        }
        if system_msg:
            payload["system"] = system_msg

        async with client.stream("POST", "/messages", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        import json

                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            if text := data.get("delta", {}).get("text"):
                                yield text
                    except (json.JSONDecodeError, KeyError):
                        continue

    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, response: str = "Mock response") -> None:
        """Initialize mock client.

        Args:
            response: Response to return.
        """
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return mock response."""
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return {
            "content": self.response,
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream mock response."""
        self.calls.append({"messages": messages, "kwargs": kwargs, "stream": True})
        words = self.response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.01)


class ResponseGenerator:
    """Generates responses from retrieved context."""

    def __init__(
        self,
        client: LLMClient,
        config: GeneratorConfig | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        """Initialize response generator.

        Args:
            client: LLM client for generation.
            config: Generator configuration.
            prompt_builder: Prompt builder for formatting.
        """
        self.client = client
        self.config = config or GeneratorConfig()
        self.prompt_builder = prompt_builder or PromptBuilder()

    async def generate(
        self,
        context: PromptContext,
        additional_instructions: str | None = None,
    ) -> GenerationResult:
        """Generate response for context.

        Args:
            context: Prompt context with query and code.
            additional_instructions: Extra instructions to add.

        Returns:
            GenerationResult with response.
        """
        import time

        start = time.time()

        # Build prompt
        prompt = self.prompt_builder.build_full_prompt(context)

        # Prepare messages
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

        if additional_instructions:
            messages.append(
                {"role": "user", "content": f"Additional: {additional_instructions}"}
            )

        # Generate with retry
        response_data = await self._generate_with_retry(messages)

        # Extract citations from context
        citations = [ctx.qualified_name for ctx in context.code_contexts]

        processing_time = (time.time() - start) * 1000

        result = GenerationResult(
            query=context.query,
            response=response_data["content"],
            model=self.config.model.value,
            tokens_used=response_data.get("usage", {}).get("total_tokens", 0),
            processing_time_ms=processing_time,
            citations=citations,
            metadata={
                "query_type": context.query_type.value,
                "context_count": len(context.code_contexts),
                "relationship_count": len(context.relationships),
            },
        )

        logger.info(
            "Response generated",
            query=context.query[:50],
            model=self.config.model.value,
            tokens=result.tokens_used,
            time_ms=processing_time,
        )

        return result

    async def generate_stream(
        self,
        context: PromptContext,
    ) -> AsyncIterator[str]:
        """Stream response for context.

        Args:
            context: Prompt context.

        Yields:
            Response chunks.
        """
        prompt = self.prompt_builder.build_full_prompt(context)

        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user"]},
        ]

        async for chunk in self.client.generate_stream(
            messages,
            model=self.config.model.value,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        ):
            yield chunk

    async def generate_explanation(
        self,
        code_content: str,
        question: str,
    ) -> GenerationResult:
        """Generate explanation for code.

        Args:
            code_content: Code to explain.
            question: Question about the code.

        Returns:
            GenerationResult with explanation.
        """
        import time

        start = time.time()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a code expert. Explain the following code clearly "
                    "and concisely. Use markdown for formatting."
                ),
            },
            {
                "role": "user",
                "content": f"Code:\n```\n{code_content}\n```\n\nQuestion: {question}",
            },
        ]

        response_data = await self._generate_with_retry(messages)
        processing_time = (time.time() - start) * 1000

        return GenerationResult(
            query=question,
            response=response_data["content"],
            model=self.config.model.value,
            tokens_used=response_data.get("usage", {}).get("total_tokens", 0),
            processing_time_ms=processing_time,
        )

    async def generate_summary(
        self,
        contexts: list[dict[str, Any]],
    ) -> str:
        """Generate summary of multiple contexts.

        Args:
            contexts: List of context dicts.

        Returns:
            Summary text.
        """
        # Format contexts for summary
        context_text = "\n\n".join(
            f"### {ctx.get('name', 'Unknown')}\n"
            f"File: {ctx.get('file_path', 'Unknown')}\n"
            f"```\n{ctx.get('content', '')[:500]}\n```"
            for ctx in contexts[:10]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize the following code components. Identify their "
                    "purposes, relationships, and key patterns."
                ),
            },
            {"role": "user", "content": context_text},
        ]

        response_data = await self._generate_with_retry(messages)
        return response_data["content"]

    async def _generate_with_retry(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Generate with retry logic.

        Args:
            messages: Messages to send.

        Returns:
            Response data.

        Raises:
            Exception: If all retries fail.
        """
        last_error: Exception | None = None

        for attempt in range(self.config.retry_attempts):
            try:
                return await asyncio.wait_for(
                    self.client.generate(
                        messages,
                        model=self.config.model.value,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    ),
                    timeout=self.config.timeout,
                )
            except asyncio.TimeoutError:
                last_error = asyncio.TimeoutError(
                    f"Generation timed out after {self.config.timeout}s"
                )
                logger.warning(
                    "Generation timeout",
                    attempt=attempt + 1,
                    max_attempts=self.config.retry_attempts,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    "Generation error",
                    attempt=attempt + 1,
                    error=str(e),
                )

            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))

        raise last_error or Exception("Generation failed")


class CachedGenerator:
    """Generator with response caching."""

    def __init__(
        self,
        generator: ResponseGenerator,
        cache_size: int = 100,
    ) -> None:
        """Initialize cached generator.

        Args:
            generator: Underlying generator.
            cache_size: Maximum cache entries.
        """
        self.generator = generator
        self.cache_size = cache_size
        self._cache: dict[str, GenerationResult] = {}
        self._cache_order: list[str] = []

    def _make_key(self, context: PromptContext) -> str:
        """Create cache key from context."""
        import hashlib

        key_data = f"{context.query}:{context.query_type.value}:{len(context.code_contexts)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def generate(
        self,
        context: PromptContext,
        use_cache: bool = True,
    ) -> GenerationResult:
        """Generate with caching.

        Args:
            context: Prompt context.
            use_cache: Whether to use cache.

        Returns:
            GenerationResult.
        """
        key = self._make_key(context)

        # Check cache
        if use_cache and key in self._cache:
            logger.debug("Cache hit", key=key[:8])
            return self._cache[key]

        # Generate
        result = await self.generator.generate(context)

        # Cache result
        self._cache[key] = result
        self._cache_order.append(key)

        # Evict old entries
        while len(self._cache) > self.cache_size:
            old_key = self._cache_order.pop(0)
            self._cache.pop(old_key, None)

        return result

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._cache_order.clear()

    @property
    def cache_stats(self) -> dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.cache_size,
        }
