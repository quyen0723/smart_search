"""GraphRAG orchestrator.

Combines retrieval and generation for graph-enhanced RAG.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

from smart_search.graph.engine import CodeGraph
from smart_search.rag.generator import (
    CachedGenerator,
    GenerationResult,
    GeneratorConfig,
    LLMClient,
    MockLLMClient,
    ResponseGenerator,
)
from smart_search.rag.prompts import (
    CodeContext,
    PromptBuilder,
    PromptContext,
    QueryType,
)
from smart_search.rag.retriever import (
    ContextRetriever,
    MockRetriever,
    RetrievalResult,
    RetrieverConfig,
)
from smart_search.search.hybrid import HybridSearcher
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class SearchMode(str, Enum):
    """Search mode for GraphRAG."""

    LOCAL = "local"  # Local context search
    GLOBAL = "global"  # Global/community search
    HYBRID = "hybrid"  # Combined local + global
    DRIFT = "drift"  # Dynamic retrieval with iterative focusing


@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG.

    Attributes:
        retriever_config: Retriever configuration.
        generator_config: Generator configuration.
        default_mode: Default search mode.
        enable_caching: Enable response caching.
        cache_size: Cache size if enabled.
        max_iterations: Max iterations for DRIFT mode.
        drift_threshold: Similarity threshold for DRIFT stopping.
    """

    retriever_config: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)
    default_mode: SearchMode = SearchMode.HYBRID
    enable_caching: bool = True
    cache_size: int = 100
    max_iterations: int = 3
    drift_threshold: float = 0.8


@dataclass
class GraphRAGResult:
    """Result from GraphRAG query.

    Attributes:
        query: Original query.
        mode: Search mode used.
        retrieval: Retrieval results.
        generation: Generated response.
        iterations: Number of DRIFT iterations (if applicable).
        total_time_ms: Total processing time.
    """

    query: str
    mode: SearchMode
    retrieval: RetrievalResult
    generation: GenerationResult
    iterations: int = 1
    total_time_ms: float = 0.0

    @property
    def response(self) -> str:
        """Get generated response."""
        return self.generation.response

    @property
    def contexts(self) -> list[CodeContext]:
        """Get retrieved contexts."""
        return self.retrieval.contexts

    @property
    def has_results(self) -> bool:
        """Whether results were found."""
        return self.retrieval.has_results

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "mode": self.mode.value,
            "response": self.response,
            "context_count": len(self.contexts),
            "citations": self.generation.citations,
            "iterations": self.iterations,
            "total_time_ms": self.total_time_ms,
            "retrieval_time_ms": self.retrieval.processing_time_ms,
            "generation_time_ms": self.generation.processing_time_ms,
        }


class GraphRAG:
    """GraphRAG orchestrator combining retrieval and generation."""

    def __init__(
        self,
        retriever: ContextRetriever,
        generator: ResponseGenerator | CachedGenerator,
        config: GraphRAGConfig | None = None,
    ) -> None:
        """Initialize GraphRAG.

        Args:
            retriever: Context retriever.
            generator: Response generator.
            config: GraphRAG configuration.
        """
        self.retriever = retriever
        self.generator = generator
        self.config = config or GraphRAGConfig()

    @classmethod
    def create(
        cls,
        searcher: HybridSearcher,
        graph: CodeGraph,
        llm_client: LLMClient,
        config: GraphRAGConfig | None = None,
    ) -> "GraphRAG":
        """Create GraphRAG with components.

        Args:
            searcher: Hybrid searcher.
            graph: Code graph.
            llm_client: LLM client.
            config: Configuration.

        Returns:
            GraphRAG instance.
        """
        config = config or GraphRAGConfig()

        retriever = ContextRetriever(
            searcher=searcher,
            graph=graph,
            config=config.retriever_config,
        )

        base_generator = ResponseGenerator(
            client=llm_client,
            config=config.generator_config,
        )

        if config.enable_caching:
            generator = CachedGenerator(
                generator=base_generator,
                cache_size=config.cache_size,
            )
        else:
            generator = base_generator

        return cls(
            retriever=retriever,
            generator=generator,
            config=config,
        )

    async def query(
        self,
        query: str,
        mode: SearchMode | None = None,
        filters: dict[str, Any] | None = None,
    ) -> GraphRAGResult:
        """Execute GraphRAG query.

        Args:
            query: User query.
            mode: Search mode (uses default if not specified).
            filters: Optional search filters.

        Returns:
            GraphRAGResult with response.
        """
        import time

        start = time.time()
        mode = mode or self.config.default_mode

        logger.info("GraphRAG query", query=query[:50], mode=mode.value)

        # Execute based on mode
        if mode == SearchMode.LOCAL:
            result = await self._local_search(query, filters)
        elif mode == SearchMode.GLOBAL:
            result = await self._global_search(query, filters)
        elif mode == SearchMode.HYBRID:
            result = await self._hybrid_search(query, filters)
        elif mode == SearchMode.DRIFT:
            result = await self._drift_search(query, filters)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        result.total_time_ms = (time.time() - start) * 1000

        logger.info(
            "GraphRAG complete",
            mode=mode.value,
            contexts=len(result.contexts),
            time_ms=result.total_time_ms,
        )

        return result

    async def query_stream(
        self,
        query: str,
        mode: SearchMode | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream GraphRAG response.

        Args:
            query: User query.
            mode: Search mode.
            filters: Optional filters.

        Yields:
            Response chunks.
        """
        mode = mode or self.config.default_mode

        # Retrieve context
        query_type = self._mode_to_query_type(mode)
        retrieval = await self.retriever.retrieve(query, query_type, filters)

        # Build prompt context
        prompt_context = retrieval.to_prompt_context()

        # Get the base generator (unwrap CachedGenerator if needed)
        base_generator = (
            self.generator.generator
            if isinstance(self.generator, CachedGenerator)
            else self.generator
        )

        # Stream response
        async for chunk in base_generator.generate_stream(prompt_context):
            yield chunk

    async def explain_code(
        self,
        code_id: str,
        question: str | None = None,
    ) -> GenerationResult:
        """Explain a specific code unit.

        Args:
            code_id: Code unit ID.
            question: Optional specific question.

        Returns:
            GenerationResult with explanation.
        """
        # Get code context
        contexts = await self.retriever.retrieve_by_relationship(
            code_id, "self", depth=0
        )

        if not contexts:
            return GenerationResult(
                query=question or "Explain this code",
                response="Code not found.",
                model="",
            )

        context = contexts[0]
        question = question or f"What does {context.name} do?"

        # Get the base generator (unwrap CachedGenerator if needed)
        base_generator = (
            self.generator.generator
            if isinstance(self.generator, CachedGenerator)
            else self.generator
        )

        return await base_generator.generate_explanation(
            context.content,
            question,
        )

    async def find_similar(
        self,
        code_content: str,
        limit: int = 5,
    ) -> list[CodeContext]:
        """Find similar code.

        Args:
            code_content: Code to find similar items for.
            limit: Maximum results.

        Returns:
            List of similar code contexts.
        """
        return await self.retriever.retrieve_similar(code_content, limit)

    async def explore_relationships(
        self,
        code_id: str,
        relationship: str = "callees",
        depth: int = 1,
    ) -> list[CodeContext]:
        """Explore code relationships.

        Args:
            code_id: Starting code unit ID.
            relationship: Relationship type (callers, callees, etc).
            depth: Traversal depth.

        Returns:
            List of related code contexts.
        """
        return await self.retriever.retrieve_by_relationship(
            code_id, relationship, depth
        )

    async def _local_search(
        self,
        query: str,
        filters: dict[str, Any] | None,
    ) -> GraphRAGResult:
        """Execute local search.

        Local search focuses on specific code units and their
        immediate relationships.
        """
        retrieval = await self.retriever.retrieve(
            query, QueryType.LOCAL, filters
        )

        prompt_context = retrieval.to_prompt_context()
        generation = await self._generate(prompt_context)

        return GraphRAGResult(
            query=query,
            mode=SearchMode.LOCAL,
            retrieval=retrieval,
            generation=generation,
        )

    async def _global_search(
        self,
        query: str,
        filters: dict[str, Any] | None,
    ) -> GraphRAGResult:
        """Execute global search.

        Global search uses community summaries to provide
        high-level understanding.
        """
        retrieval = await self.retriever.retrieve(
            query, QueryType.GLOBAL, filters
        )

        prompt_context = retrieval.to_prompt_context()
        generation = await self._generate(prompt_context)

        return GraphRAGResult(
            query=query,
            mode=SearchMode.GLOBAL,
            retrieval=retrieval,
            generation=generation,
        )

    async def _hybrid_search(
        self,
        query: str,
        filters: dict[str, Any] | None,
    ) -> GraphRAGResult:
        """Execute hybrid search.

        Combines local and global search for comprehensive results.
        """
        # Run both searches in parallel
        local_task = self.retriever.retrieve(query, QueryType.LOCAL, filters)
        global_task = self.retriever.retrieve(query, QueryType.GLOBAL, filters)

        local_result, global_result = await asyncio.gather(local_task, global_task)

        # Merge results
        merged = RetrievalResult(
            query=query,
            query_type=QueryType.DRIFT,  # Use DRIFT to indicate hybrid
            contexts=local_result.contexts,
            relationships=local_result.relationships,
            communities=global_result.communities,
            processing_time_ms=max(
                local_result.processing_time_ms,
                global_result.processing_time_ms,
            ),
        )

        prompt_context = merged.to_prompt_context()
        generation = await self._generate(prompt_context)

        return GraphRAGResult(
            query=query,
            mode=SearchMode.HYBRID,
            retrieval=merged,
            generation=generation,
        )

    async def _drift_search(
        self,
        query: str,
        filters: dict[str, Any] | None,
    ) -> GraphRAGResult:
        """Execute DRIFT (Dynamic Retrieval with Iterative Focusing) search.

        Iteratively refines context based on generated responses.
        """
        best_retrieval: RetrievalResult | None = None
        best_generation: GenerationResult | None = None
        iterations = 0

        current_query = query
        seen_ids: set[str] = set()

        for i in range(self.config.max_iterations):
            iterations = i + 1

            # Retrieve context
            retrieval = await self.retriever.retrieve(
                current_query, QueryType.DRIFT, filters
            )

            # Track seen contexts
            new_ids = {ctx.code_id for ctx in retrieval.contexts}
            overlap = len(seen_ids & new_ids) / max(len(new_ids), 1)
            seen_ids.update(new_ids)

            # Generate response
            prompt_context = retrieval.to_prompt_context()
            generation = await self._generate(prompt_context)

            # Update best results
            if best_retrieval is None or len(retrieval.contexts) > len(
                best_retrieval.contexts
            ):
                best_retrieval = retrieval
                best_generation = generation

            # Check stopping condition
            if overlap >= self.config.drift_threshold:
                logger.debug(
                    "DRIFT converged",
                    iteration=iterations,
                    overlap=overlap,
                )
                break

            # Refine query for next iteration
            # Use key terms from response to focus search
            current_query = self._refine_query(query, generation.response)

        return GraphRAGResult(
            query=query,
            mode=SearchMode.DRIFT,
            retrieval=best_retrieval or RetrievalResult(query=query, query_type=QueryType.DRIFT),
            generation=best_generation or GenerationResult(query=query, response="", model=""),
            iterations=iterations,
        )

    def _refine_query(self, original: str, response: str) -> str:
        """Refine query based on response.

        Args:
            original: Original query.
            response: Generated response.

        Returns:
            Refined query.
        """
        # Extract key terms from response (simplified)
        # In production, this could use NLP for keyword extraction
        words = response.split()[:50]
        # Filter to code-like terms
        code_terms = [
            w.strip(".,()[]{}:;")
            for w in words
            if any(c in w for c in ["_", "(", ")", "."])
            and len(w) > 2
        ]

        if code_terms:
            return f"{original} {' '.join(code_terms[:5])}"
        return original

    async def _generate(self, context: PromptContext) -> GenerationResult:
        """Generate response from context.

        Args:
            context: Prompt context.

        Returns:
            GenerationResult.
        """
        if isinstance(self.generator, CachedGenerator):
            return await self.generator.generate(context)
        return await self.generator.generate(context)

    def _mode_to_query_type(self, mode: SearchMode) -> QueryType:
        """Convert SearchMode to QueryType."""
        mapping = {
            SearchMode.LOCAL: QueryType.LOCAL,
            SearchMode.GLOBAL: QueryType.GLOBAL,
            SearchMode.HYBRID: QueryType.DRIFT,
            SearchMode.DRIFT: QueryType.DRIFT,
        }
        return mapping.get(mode, QueryType.LOCAL)

    def clear_cache(self) -> None:
        """Clear generator cache if enabled."""
        if isinstance(self.generator, CachedGenerator):
            self.generator.clear_cache()


class MockGraphRAG:
    """Mock GraphRAG for testing."""

    def __init__(self) -> None:
        """Initialize mock GraphRAG."""
        self._contexts: list[CodeContext] = []
        self._response: str = "Mock response"

    def set_contexts(self, contexts: list[CodeContext]) -> None:
        """Set mock contexts."""
        self._contexts = contexts

    def set_response(self, response: str) -> None:
        """Set mock response."""
        self._response = response

    async def query(
        self,
        query: str,
        mode: SearchMode | None = None,
        filters: dict[str, Any] | None = None,
    ) -> GraphRAGResult:
        """Execute mock query."""
        retrieval = RetrievalResult(
            query=query,
            query_type=QueryType.LOCAL,
            contexts=self._contexts,
            processing_time_ms=1.0,
        )

        generation = GenerationResult(
            query=query,
            response=self._response,
            model="mock",
            processing_time_ms=1.0,
        )

        return GraphRAGResult(
            query=query,
            mode=mode or SearchMode.LOCAL,
            retrieval=retrieval,
            generation=generation,
        )


def create_mock_graphrag() -> GraphRAG:
    """Create GraphRAG with mock components for testing.

    Returns:
        GraphRAG instance with mock components.
    """
    mock_retriever = MockRetriever()
    mock_client = MockLLMClient(response="This is a mock response for testing.")

    generator = ResponseGenerator(
        client=mock_client,
        config=GeneratorConfig(),
    )

    # Create a minimal GraphRAG-like object
    # We need to cast/adapt MockRetriever to match ContextRetriever interface
    class AdaptedGraphRAG(GraphRAG):
        def __init__(self) -> None:
            self.retriever = mock_retriever  # type: ignore
            self.generator = generator
            self.config = GraphRAGConfig()

    return AdaptedGraphRAG()
