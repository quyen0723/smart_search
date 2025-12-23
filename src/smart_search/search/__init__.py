"""Search module for hybrid code search."""

from smart_search.search.hybrid import (
    HybridSearchConfig,
    HybridSearcher,
    SearchFacade,
)
from smart_search.search.indexer import (
    BatchIndexer,
    IndexingConfig,
    IndexingResult,
    SearchIndexer,
)
from smart_search.search.meilisearch_client import (
    MeilisearchClient,
    MockMeilisearchClient,
)
from smart_search.search.schemas import (
    IndexDocument,
    IndexStats,
    SearchFilter,
    SearchHit,
    SearchQuery,
    SearchResult,
    SearchType,
    SortField,
    SortOrder,
)

__all__ = [
    # Schemas
    "SearchType",
    "SortOrder",
    "SortField",
    "SearchFilter",
    "SearchQuery",
    "SearchHit",
    "SearchResult",
    "IndexDocument",
    "IndexStats",
    # Client
    "MeilisearchClient",
    "MockMeilisearchClient",
    # Hybrid
    "HybridSearchConfig",
    "HybridSearcher",
    "SearchFacade",
    # Indexer
    "IndexingConfig",
    "IndexingResult",
    "SearchIndexer",
    "BatchIndexer",
]
