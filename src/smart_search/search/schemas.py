"""Search schemas and models.

Defines data structures for search operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class SearchType(str, Enum):
    """Type of search to perform."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class SortOrder(str, Enum):
    """Sort order for results."""

    ASC = "asc"
    DESC = "desc"


class SortField(str, Enum):
    """Fields to sort by."""

    RELEVANCE = "relevance"
    FILE_PATH = "file_path"
    LINE_NUMBER = "line_number"
    NAME = "name"
    CREATED_AT = "created_at"
    UPDATED_AT = "updated_at"


@dataclass
class SearchFilter:
    """Filter criteria for search.

    Attributes:
        languages: Filter by programming languages.
        file_patterns: Filter by file path patterns.
        code_types: Filter by code unit types.
        exclude_patterns: Patterns to exclude.
        min_line: Minimum line number.
        max_line: Maximum line number.
    """

    languages: list[str] = field(default_factory=list)
    file_patterns: list[str] = field(default_factory=list)
    code_types: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    min_line: int | None = None
    max_line: int | None = None

    def to_meilisearch_filter(self) -> str | None:
        """Convert to Meilisearch filter string.

        Returns:
            Filter string or None if no filters.
        """
        filters = []

        if self.languages:
            lang_filter = " OR ".join(f'language = "{lang}"' for lang in self.languages)
            filters.append(f"({lang_filter})")

        if self.code_types:
            type_filter = " OR ".join(
                f'code_type = "{ct}"' for ct in self.code_types
            )
            filters.append(f"({type_filter})")

        if self.min_line is not None:
            filters.append(f"line_start >= {self.min_line}")

        if self.max_line is not None:
            filters.append(f"line_end <= {self.max_line}")

        return " AND ".join(filters) if filters else None


@dataclass
class SearchQuery:
    """Search query parameters.

    Attributes:
        query: The search query string.
        search_type: Type of search to perform.
        filters: Search filters.
        limit: Maximum number of results.
        offset: Number of results to skip.
        sort_by: Field to sort by.
        sort_order: Sort order.
        highlight: Whether to highlight matches.
        include_content: Whether to include full content.
    """

    query: str
    search_type: SearchType = SearchType.HYBRID
    filters: SearchFilter | None = None
    limit: int = 20
    offset: int = 0
    sort_by: SortField = SortField.RELEVANCE
    sort_order: SortOrder = SortOrder.DESC
    highlight: bool = True
    include_content: bool = True

    def __post_init__(self) -> None:
        """Validate query parameters."""
        if self.limit < 1:
            self.limit = 1
        if self.limit > 100:
            self.limit = 100
        if self.offset < 0:
            self.offset = 0


@dataclass
class SearchHit:
    """A single search result.

    Attributes:
        id: Document ID.
        name: Code unit name.
        qualified_name: Fully qualified name.
        code_type: Type of code unit.
        file_path: Path to the file.
        line_start: Starting line number.
        line_end: Ending line number.
        content: Code content.
        language: Programming language.
        score: Relevance score.
        highlights: Highlighted snippets.
        metadata: Additional metadata.
    """

    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: Path
    line_start: int
    line_end: int
    content: str
    language: str
    score: float = 0.0
    highlights: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "id": self.id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "code_type": self.code_type,
            "file_path": str(self.file_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content": self.content,
            "language": self.language,
            "score": self.score,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }

    @classmethod
    def from_meilisearch(cls, doc: dict[str, Any]) -> "SearchHit":
        """Create from Meilisearch document.

        Args:
            doc: Meilisearch document.

        Returns:
            SearchHit instance.
        """
        # Extract highlights from _formatted field
        formatted = doc.get("_formatted", {})
        highlights = {}
        for field_name in ["content", "name", "qualified_name"]:
            if field_name in formatted:
                highlights[field_name] = [formatted[field_name]]

        return cls(
            id=doc.get("id", ""),
            name=doc.get("name", ""),
            qualified_name=doc.get("qualified_name", ""),
            code_type=doc.get("code_type", ""),
            file_path=Path(doc.get("file_path", "")),
            line_start=doc.get("line_start", 0),
            line_end=doc.get("line_end", 0),
            content=doc.get("content", ""),
            language=doc.get("language", ""),
            score=doc.get("_rankingScore", 0.0),
            highlights=highlights,
            metadata=doc.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Search results container.

    Attributes:
        hits: List of search hits.
        total: Total number of matching documents.
        query: Original query string.
        search_type: Type of search performed.
        processing_time_ms: Time taken to process query.
        offset: Offset used.
        limit: Limit used.
    """

    hits: list[SearchHit]
    total: int
    query: str
    search_type: SearchType
    processing_time_ms: float = 0.0
    offset: int = 0
    limit: int = 20

    @property
    def count(self) -> int:
        """Number of hits returned."""
        return len(self.hits)

    @property
    def has_more(self) -> bool:
        """Whether there are more results."""
        return self.offset + self.count < self.total

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "hits": [hit.to_dict() for hit in self.hits],
            "total": self.total,
            "count": self.count,
            "query": self.query,
            "search_type": self.search_type.value,
            "processing_time_ms": self.processing_time_ms,
            "offset": self.offset,
            "limit": self.limit,
            "has_more": self.has_more,
        }


@dataclass
class IndexDocument:
    """Document to be indexed.

    Attributes:
        id: Unique document ID.
        name: Code unit name.
        qualified_name: Fully qualified name.
        code_type: Type of code unit.
        file_path: Path to the file.
        line_start: Starting line number.
        line_end: Ending line number.
        content: Code content.
        language: Programming language.
        signature: Function/method signature.
        docstring: Documentation string.
        embedding: Vector embedding.
        metadata: Additional metadata.
    """

    id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    language: str
    signature: str = ""
    docstring: str = ""
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for indexing.

        Returns:
            Dictionary representation.
        """
        doc = {
            "id": self.id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "code_type": self.code_type,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content": self.content,
            "language": self.language,
            "signature": self.signature,
            "docstring": self.docstring,
            "metadata": self.metadata,
        }
        if self.embedding is not None:
            doc["_vectors"] = {"default": self.embedding}
        return doc


@dataclass
class IndexStats:
    """Index statistics.

    Attributes:
        total_documents: Total number of indexed documents.
        is_indexing: Whether indexing is in progress.
        field_distribution: Distribution of fields.
        last_update: Last update timestamp.
    """

    total_documents: int
    is_indexing: bool
    field_distribution: dict[str, int] = field(default_factory=dict)
    last_update: str | None = None

    @classmethod
    def from_meilisearch(cls, stats: dict[str, Any]) -> "IndexStats":
        """Create from Meilisearch stats.

        Args:
            stats: Meilisearch index stats.

        Returns:
            IndexStats instance.
        """
        return cls(
            total_documents=stats.get("numberOfDocuments", 0),
            is_indexing=stats.get("isIndexing", False),
            field_distribution=stats.get("fieldDistribution", {}),
        )
