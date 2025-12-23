"""Hybrid search combining keyword and semantic search.

Provides unified search interface with reranking.
"""

from dataclasses import dataclass
from typing import Any

from smart_search.embedding.jina_embedder import BaseEmbedder
from smart_search.search.meilisearch_client import MeilisearchClient
from smart_search.search.schemas import (
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchType,
)
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search.

    Attributes:
        semantic_weight: Weight for semantic search (0-1).
        keyword_weight: Weight for keyword search (0-1).
        rerank: Whether to rerank results.
        min_score: Minimum score threshold.
        dedup: Whether to deduplicate results.
    """

    semantic_weight: float = 0.5
    keyword_weight: float = 0.5
    rerank: bool = True
    min_score: float = 0.0
    dedup: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize weights."""
        total = self.semantic_weight + self.keyword_weight
        if total > 0:
            self.semantic_weight /= total
            self.keyword_weight /= total


class HybridSearcher:
    """Hybrid search combining keyword and semantic search."""

    def __init__(
        self,
        client: MeilisearchClient,
        embedder: BaseEmbedder,
        config: HybridSearchConfig | None = None,
    ) -> None:
        """Initialize hybrid searcher.

        Args:
            client: Meilisearch client.
            embedder: Embedding provider.
            config: Search configuration.
        """
        self.client = client
        self.embedder = embedder
        self.config = config or HybridSearchConfig()

    async def search(self, query: SearchQuery) -> SearchResult:
        """Perform hybrid search.

        Args:
            query: Search query.

        Returns:
            SearchResult with combined results.
        """
        if query.search_type == SearchType.KEYWORD:
            return await self._keyword_search(query)
        elif query.search_type == SearchType.SEMANTIC:
            return await self._semantic_search(query)
        else:
            return await self._hybrid_search(query)

    async def _keyword_search(self, query: SearchQuery) -> SearchResult:
        """Perform keyword-only search.

        Args:
            query: Search query.

        Returns:
            SearchResult.
        """
        return await self.client.search(query)

    async def _semantic_search(self, query: SearchQuery) -> SearchResult:
        """Perform semantic-only search.

        Args:
            query: Search query.

        Returns:
            SearchResult.
        """
        # Get query embedding
        embed_result = await self.embedder.embed(query.query)
        embedding = embed_result.embedding

        return await self.client.search(query, embedding=embedding)

    async def _hybrid_search(self, query: SearchQuery) -> SearchResult:
        """Perform hybrid search.

        Args:
            query: Search query.

        Returns:
            SearchResult with combined and reranked results.
        """
        # Get query embedding
        embed_result = await self.embedder.embed(query.query)
        embedding = embed_result.embedding

        # Use Meilisearch's native hybrid search
        result = await self.client.search(query, embedding=embedding)

        # Apply additional reranking if configured
        if self.config.rerank and result.hits:
            result = self._rerank_results(result, query)

        # Filter by minimum score
        if self.config.min_score > 0:
            result = self._filter_by_score(result)

        return result

    def _rerank_results(
        self,
        result: SearchResult,
        query: SearchQuery,
    ) -> SearchResult:
        """Rerank search results.

        Args:
            result: Original search result.
            query: Search query.

        Returns:
            Reranked SearchResult.
        """
        hits = result.hits

        # Apply query-specific boosting
        boosted_hits = []
        query_terms = set(query.query.lower().split())

        for hit in hits:
            boost = 1.0

            # Boost exact name matches
            if hit.name.lower() in query_terms:
                boost *= 2.0

            # Boost if query appears in qualified name
            if query.query.lower() in hit.qualified_name.lower():
                boost *= 1.5

            # Boost functions/methods for function-like queries
            if any(term in query_terms for term in ["def", "function", "method"]):
                if hit.code_type in ("function", "method"):
                    boost *= 1.3

            # Boost classes for class-like queries
            if any(term in query_terms for term in ["class", "type"]):
                if hit.code_type == "class":
                    boost *= 1.3

            # Apply boost to score
            boosted_hit = SearchHit(
                id=hit.id,
                name=hit.name,
                qualified_name=hit.qualified_name,
                code_type=hit.code_type,
                file_path=hit.file_path,
                line_start=hit.line_start,
                line_end=hit.line_end,
                content=hit.content,
                language=hit.language,
                score=hit.score * boost,
                highlights=hit.highlights,
                metadata=hit.metadata,
            )
            boosted_hits.append(boosted_hit)

        # Sort by boosted score
        boosted_hits.sort(key=lambda h: h.score, reverse=True)

        return SearchResult(
            hits=boosted_hits,
            total=result.total,
            query=result.query,
            search_type=result.search_type,
            processing_time_ms=result.processing_time_ms,
            offset=result.offset,
            limit=result.limit,
        )

    def _filter_by_score(self, result: SearchResult) -> SearchResult:
        """Filter results by minimum score.

        Args:
            result: Search result.

        Returns:
            Filtered SearchResult.
        """
        filtered_hits = [
            hit for hit in result.hits if hit.score >= self.config.min_score
        ]

        return SearchResult(
            hits=filtered_hits,
            total=len(filtered_hits),
            query=result.query,
            search_type=result.search_type,
            processing_time_ms=result.processing_time_ms,
            offset=result.offset,
            limit=result.limit,
        )

    async def search_similar(
        self,
        code_content: str,
        limit: int = 10,
        exclude_ids: list[str] | None = None,
    ) -> SearchResult:
        """Find similar code snippets.

        Args:
            code_content: Code to find similar matches for.
            limit: Maximum results.
            exclude_ids: IDs to exclude.

        Returns:
            SearchResult with similar code.
        """
        # Get embedding for the code
        embed_result = await self.embedder.embed(code_content)
        embedding = embed_result.embedding

        # Search with semantic only
        query = SearchQuery(
            query="",
            search_type=SearchType.SEMANTIC,
            limit=limit + len(exclude_ids or []),
        )

        result = await self.client.search(query, embedding=embedding)

        # Filter excluded IDs
        if exclude_ids:
            exclude_set = set(exclude_ids)
            result = SearchResult(
                hits=[h for h in result.hits if h.id not in exclude_set][:limit],
                total=result.total,
                query=result.query,
                search_type=result.search_type,
                processing_time_ms=result.processing_time_ms,
                offset=0,
                limit=limit,
            )

        return result

    async def close(self) -> None:
        """Close resources."""
        await self.embedder.close()
        await self.client.close()


class SearchFacade:
    """Simplified search interface.

    Provides a simple API for common search operations.
    """

    def __init__(self, searcher: HybridSearcher) -> None:
        """Initialize facade.

        Args:
            searcher: Hybrid searcher instance.
        """
        self.searcher = searcher

    async def search(
        self,
        query: str,
        limit: int = 20,
        search_type: SearchType = SearchType.HYBRID,
        languages: list[str] | None = None,
        code_types: list[str] | None = None,
    ) -> SearchResult:
        """Search for code.

        Args:
            query: Search query.
            limit: Maximum results.
            search_type: Type of search.
            languages: Filter by languages.
            code_types: Filter by code types.

        Returns:
            SearchResult.
        """
        from smart_search.search.schemas import SearchFilter

        filters = None
        if languages or code_types:
            filters = SearchFilter(
                languages=languages or [],
                code_types=code_types or [],
            )

        search_query = SearchQuery(
            query=query,
            search_type=search_type,
            filters=filters,
            limit=limit,
        )

        return await self.searcher.search(search_query)

    async def find_function(
        self,
        name: str,
        language: str | None = None,
    ) -> SearchResult:
        """Find a function by name.

        Args:
            name: Function name.
            language: Optional language filter.

        Returns:
            SearchResult.
        """
        from smart_search.search.schemas import SearchFilter

        filters = SearchFilter(
            code_types=["function", "method"],
            languages=[language] if language else [],
        )

        query = SearchQuery(
            query=name,
            search_type=SearchType.KEYWORD,
            filters=filters,
            limit=10,
        )

        return await self.searcher.search(query)

    async def find_class(
        self,
        name: str,
        language: str | None = None,
    ) -> SearchResult:
        """Find a class by name.

        Args:
            name: Class name.
            language: Optional language filter.

        Returns:
            SearchResult.
        """
        from smart_search.search.schemas import SearchFilter

        filters = SearchFilter(
            code_types=["class"],
            languages=[language] if language else [],
        )

        query = SearchQuery(
            query=name,
            search_type=SearchType.KEYWORD,
            filters=filters,
            limit=10,
        )

        return await self.searcher.search(query)

    async def find_similar(
        self,
        code: str,
        limit: int = 10,
    ) -> SearchResult:
        """Find similar code.

        Args:
            code: Code snippet.
            limit: Maximum results.

        Returns:
            SearchResult.
        """
        return await self.searcher.search_similar(code, limit=limit)

    async def close(self) -> None:
        """Close resources."""
        await self.searcher.close()
