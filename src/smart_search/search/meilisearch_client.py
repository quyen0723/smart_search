"""Meilisearch client wrapper.

Provides async interface for Meilisearch operations.
"""

import asyncio
from typing import Any

from meilisearch_python_sdk import AsyncClient
from meilisearch_python_sdk.errors import MeilisearchApiError
from meilisearch_python_sdk.models.search import Hybrid
from meilisearch_python_sdk.models.settings import (
    Embedders,
    MeilisearchSettings,
    UserProvidedEmbedder,
)

from smart_search.core.exceptions import (
    SearchConnectionError,
    SearchIndexError,
    SearchQueryError,
)
from smart_search.search.schemas import (
    IndexDocument,
    IndexStats,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchType,
)
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class MeilisearchClient:
    """Async Meilisearch client wrapper.

    Handles connection, indexing, and search operations.
    """

    DEFAULT_INDEX = "code_units"

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        index_name: str | None = None,
        embedding_dimensions: int = 1024,
    ) -> None:
        """Initialize client.

        Args:
            url: Meilisearch server URL.
            api_key: Optional API key.
            index_name: Index name to use.
            embedding_dimensions: Dimensions of embeddings for vector search.
        """
        self.url = url
        self.api_key = api_key
        self.index_name = index_name or self.DEFAULT_INDEX
        self.embedding_dimensions = embedding_dimensions
        self._client: AsyncClient | None = None
        self._initialized = False

    async def _ensure_client(self) -> AsyncClient:
        """Ensure client is initialized.

        Returns:
            Meilisearch client.

        Raises:
            SearchConnectionError: If connection fails.
        """
        if self._client is None:
            try:
                self._client = AsyncClient(self.url, self.api_key)
                self._initialized = True
            except Exception as e:
                raise SearchConnectionError(self.url, e)
        return self._client

    async def initialize_index(self, recreate: bool = False) -> None:
        """Initialize the search index.

        Args:
            recreate: Whether to recreate existing index.

        Raises:
            SearchIndexError: If initialization fails.
        """
        client = await self._ensure_client()

        try:
            if recreate:
                try:
                    await client.delete_index_if_exists(self.index_name)
                    logger.info("Deleted existing index", index=self.index_name)
                except MeilisearchApiError:
                    pass  # Index doesn't exist

            # Create index with primary key
            index = await client.create_index(
                self.index_name,
                primary_key="id",
            )

            # Configure settings
            settings = MeilisearchSettings(
                searchable_attributes=[
                    "content",
                    "name",
                    "qualified_name",
                    "signature",
                    "docstring",
                ],
                filterable_attributes=[
                    "language",
                    "code_type",
                    "file_path",
                    "line_start",
                    "line_end",
                ],
                sortable_attributes=[
                    "name",
                    "file_path",
                    "line_start",
                ],
                ranking_rules=[
                    "words",
                    "typo",
                    "proximity",
                    "attribute",
                    "sort",
                    "exactness",
                ],
                embedders={
                    "default": UserProvidedEmbedder(
                        dimensions=self.embedding_dimensions
                    )
                },
            )

            await index.update_settings(settings)
            logger.info("Index initialized", index=self.index_name)

        except MeilisearchApiError as e:
            if "already exists" not in str(e):
                raise SearchIndexError(self.index_name, "create", e)
            logger.info("Index already exists", index=self.index_name)

    async def add_documents(
        self,
        documents: list[IndexDocument],
        batch_size: int = 100,
    ) -> int:
        """Add documents to the index.

        Args:
            documents: Documents to index.
            batch_size: Number of documents per batch.

        Returns:
            Number of documents indexed.

        Raises:
            SearchIndexError: If indexing fails.
        """
        if not documents:
            return 0

        client = await self._ensure_client()
        index = client.index(self.index_name)
        total = 0

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                docs = [doc.to_dict() for doc in batch]
                await index.add_documents(docs)
                total += len(batch)
                logger.debug("Indexed batch", count=len(batch), total=total)

            logger.info("Documents indexed", count=total)
            return total

        except MeilisearchApiError as e:
            raise SearchIndexError(self.index_name, "add_documents", e)

    async def update_documents(
        self,
        documents: list[IndexDocument],
    ) -> int:
        """Update existing documents.

        Args:
            documents: Documents to update.

        Returns:
            Number of documents updated.

        Raises:
            SearchIndexError: If update fails.
        """
        if not documents:
            return 0

        client = await self._ensure_client()
        index = client.index(self.index_name)

        try:
            docs = [doc.to_dict() for doc in documents]
            await index.update_documents(docs)
            logger.info("Documents updated", count=len(documents))
            return len(documents)

        except MeilisearchApiError as e:
            raise SearchIndexError(self.index_name, "update_documents", e)

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID.

        Args:
            ids: Document IDs to delete.

        Returns:
            Number of documents deleted.

        Raises:
            SearchIndexError: If deletion fails.
        """
        if not ids:
            return 0

        client = await self._ensure_client()
        index = client.index(self.index_name)

        try:
            await index.delete_documents(ids)
            logger.info("Documents deleted", count=len(ids))
            return len(ids)

        except MeilisearchApiError as e:
            raise SearchIndexError(self.index_name, "delete_documents", e)

    async def delete_by_filter(self, filter_str: str) -> None:
        """Delete documents matching filter.

        Args:
            filter_str: Meilisearch filter string.

        Raises:
            SearchIndexError: If deletion fails.
        """
        client = await self._ensure_client()
        index = client.index(self.index_name)

        try:
            await index.delete_documents_by_filter(filter_str)
            logger.info("Documents deleted by filter", filter=filter_str)

        except MeilisearchApiError as e:
            raise SearchIndexError(self.index_name, "delete_by_filter", e)

    async def search(
        self,
        query: SearchQuery,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        """Search the index.

        Args:
            query: Search query parameters.
            embedding: Optional query embedding for semantic search.

        Returns:
            SearchResult with hits.

        Raises:
            SearchQueryError: If search fails.
        """
        client = await self._ensure_client()
        index = client.index(self.index_name)

        try:
            search_params: dict[str, Any] = {
                "limit": query.limit,
                "offset": query.offset,
                "show_ranking_score": True,
            }

            # Add filter if present
            if query.filters:
                filter_str = query.filters.to_meilisearch_filter()
                if filter_str:
                    search_params["filter"] = filter_str

            # Add highlighting
            if query.highlight:
                search_params["attributes_to_highlight"] = [
                    "content",
                    "name",
                    "qualified_name",
                ]
                search_params["highlight_pre_tag"] = "<mark>"
                search_params["highlight_post_tag"] = "</mark>"

            # Handle different search types
            if query.search_type == SearchType.SEMANTIC and embedding:
                # Pure semantic search
                search_params["vector"] = embedding
                search_params["hybrid"] = Hybrid(semantic_ratio=1.0, embedder="default")
                result = await index.search(query.query, **search_params)
            elif query.search_type == SearchType.HYBRID and embedding:
                # Hybrid search
                search_params["vector"] = embedding
                search_params["hybrid"] = Hybrid(semantic_ratio=0.5, embedder="default")
                result = await index.search(query.query, **search_params)
            else:
                # Keyword search
                result = await index.search(query.query, **search_params)

            # Convert hits
            hits = [SearchHit.from_meilisearch(hit) for hit in result.hits]

            return SearchResult(
                hits=hits,
                total=result.estimated_total_hits or len(hits),
                query=query.query,
                search_type=query.search_type,
                processing_time_ms=result.processing_time_ms or 0.0,
                offset=query.offset,
                limit=query.limit,
            )

        except MeilisearchApiError as e:
            raise SearchQueryError(query.query, e)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get a document by ID.

        Args:
            doc_id: Document ID.

        Returns:
            Document or None if not found.
        """
        client = await self._ensure_client()
        index = client.index(self.index_name)

        try:
            return await index.get_document(doc_id)
        except MeilisearchApiError:
            return None

    async def get_stats(self) -> IndexStats:
        """Get index statistics.

        Returns:
            IndexStats object.

        Raises:
            SearchIndexError: If stats retrieval fails.
        """
        client = await self._ensure_client()
        index = client.index(self.index_name)

        try:
            stats = await index.get_stats()
            return IndexStats(
                total_documents=stats.number_of_documents,
                is_indexing=stats.is_indexing,
                field_distribution=stats.field_distribution or {},
            )

        except MeilisearchApiError as e:
            raise SearchIndexError(self.index_name, "get_stats", e)

    async def wait_for_task(self, task_uid: int, timeout_ms: int = 30000) -> None:
        """Wait for a task to complete.

        Args:
            task_uid: Task UID.
            timeout_ms: Timeout in milliseconds.
        """
        client = await self._ensure_client()
        await client.wait_for_task(task_uid, timeout_in_ms=timeout_ms)

    async def health_check(self) -> bool:
        """Check if Meilisearch is healthy.

        Returns:
            True if healthy.
        """
        try:
            client = await self._ensure_client()
            health = await client.health()
            return health.status == "available"
        except Exception:
            return False

    async def close(self) -> None:
        """Close the client connection."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            self._initialized = False

    async def __aenter__(self) -> "MeilisearchClient":
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit async context."""
        await self.close()


class MockMeilisearchClient:
    """Mock Meilisearch client for testing.

    Stores documents in memory.
    """

    def __init__(self, index_name: str = "test_index") -> None:
        """Initialize mock client.

        Args:
            index_name: Index name.
        """
        self.index_name = index_name
        self._documents: dict[str, dict[str, Any]] = {}
        self._initialized = False

    async def initialize_index(self, recreate: bool = False) -> None:
        """Initialize the index."""
        if recreate:
            self._documents.clear()
        self._initialized = True

    async def add_documents(
        self,
        documents: list[IndexDocument],
        batch_size: int = 100,
    ) -> int:
        """Add documents."""
        for doc in documents:
            self._documents[doc.id] = doc.to_dict()
        return len(documents)

    async def update_documents(
        self,
        documents: list[IndexDocument],
    ) -> int:
        """Update documents."""
        for doc in documents:
            self._documents[doc.id] = doc.to_dict()
        return len(documents)

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents."""
        count = 0
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                count += 1
        return count

    async def delete_by_filter(self, filter_str: str) -> None:
        """Delete by filter (simplified)."""
        pass

    async def search(
        self,
        query: SearchQuery,
        embedding: list[float] | None = None,
    ) -> SearchResult:
        """Simple text search."""
        hits = []
        query_lower = query.query.lower()

        for doc in self._documents.values():
            # Simple text matching
            content = doc.get("content", "").lower()
            name = doc.get("name", "").lower()

            if query_lower in content or query_lower in name:
                hit = SearchHit(
                    id=doc["id"],
                    name=doc.get("name", ""),
                    qualified_name=doc.get("qualified_name", ""),
                    code_type=doc.get("code_type", ""),
                    file_path=doc.get("file_path", ""),
                    line_start=doc.get("line_start", 0),
                    line_end=doc.get("line_end", 0),
                    content=doc.get("content", ""),
                    language=doc.get("language", ""),
                    score=1.0,
                )
                hits.append(hit)

        # Apply pagination
        total = len(hits)
        hits = hits[query.offset : query.offset + query.limit]

        return SearchResult(
            hits=hits,
            total=total,
            query=query.query,
            search_type=query.search_type,
            processing_time_ms=1.0,
            offset=query.offset,
            limit=query.limit,
        )

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Get document by ID."""
        return self._documents.get(doc_id)

    async def get_stats(self) -> IndexStats:
        """Get stats."""
        return IndexStats(
            total_documents=len(self._documents),
            is_indexing=False,
        )

    async def health_check(self) -> bool:
        """Health check."""
        return True

    async def close(self) -> None:
        """Close client."""
        pass

    async def __aenter__(self) -> "MockMeilisearchClient":
        """Enter context."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit context."""
        pass
