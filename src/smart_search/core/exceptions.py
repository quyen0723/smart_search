"""Custom exceptions for Smart Search system.

This module defines a hierarchy of exceptions used throughout the application
for consistent error handling and reporting.
"""

from typing import Any


class SmartSearchError(Exception):
    """Base exception for all Smart Search errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        result: dict[str, Any] = {
            "error": self.__class__.__name__,
            "message": self.message,
        }
        if self.details:
            result["details"] = self.details
        return result


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(SmartSearchError):
    """Error in application configuration."""

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration value is missing."""

    def __init__(self, config_key: str) -> None:
        super().__init__(
            message=f"Missing required configuration: {config_key}",
            details={"config_key": config_key},
        )


# =============================================================================
# Search Errors
# =============================================================================


class SearchError(SmartSearchError):
    """Base class for search-related errors."""

    pass


class SearchConnectionError(SearchError):
    """Failed to connect to search engine."""

    def __init__(self, url: str, cause: Exception | None = None) -> None:
        super().__init__(
            message=f"Failed to connect to search engine at {url}",
            details={"url": url},
            cause=cause,
        )


class SearchIndexError(SearchError):
    """Error during indexing operation."""

    def __init__(self, index_name: str, operation: str, cause: Exception | None = None) -> None:
        super().__init__(
            message=f"Index operation '{operation}' failed on '{index_name}'",
            details={"index_name": index_name, "operation": operation},
            cause=cause,
        )


class SearchQueryError(SearchError):
    """Error executing search query."""

    def __init__(self, query: str, cause: Exception | None = None) -> None:
        super().__init__(
            message="Search query execution failed",
            details={"query": query},
            cause=cause,
        )


class SearchTimeoutError(SearchError):
    """Search operation timed out."""

    def __init__(self, timeout_ms: int) -> None:
        super().__init__(
            message=f"Search timed out after {timeout_ms}ms",
            details={"timeout_ms": timeout_ms},
        )


# =============================================================================
# Graph Errors
# =============================================================================


class GraphError(SmartSearchError):
    """Base class for graph-related errors."""

    pass


class NodeNotFoundError(GraphError):
    """Requested node not found in graph."""

    def __init__(self, node_id: str) -> None:
        super().__init__(
            message=f"Node not found: {node_id}",
            details={"node_id": node_id},
        )


class EdgeNotFoundError(GraphError):
    """Requested edge not found in graph."""

    def __init__(self, source_id: str, target_id: str) -> None:
        super().__init__(
            message=f"Edge not found: {source_id} -> {target_id}",
            details={"source_id": source_id, "target_id": target_id},
        )


class GraphSerializationError(GraphError):
    """Error serializing/deserializing graph."""

    def __init__(self, operation: str, path: str, cause: Exception | None = None) -> None:
        super().__init__(
            message=f"Graph {operation} failed for {path}",
            details={"operation": operation, "path": path},
            cause=cause,
        )


class CyclicDependencyError(GraphError):
    """Cyclic dependency detected in graph."""

    def __init__(self, cycle: list[str]) -> None:
        super().__init__(
            message="Cyclic dependency detected",
            details={"cycle": cycle},
        )


# =============================================================================
# Parsing Errors
# =============================================================================


class ParsingError(SmartSearchError):
    """Base class for parsing-related errors."""

    pass


class UnsupportedLanguageError(ParsingError):
    """Language not supported for parsing."""

    def __init__(self, language: str, supported: list[str]) -> None:
        super().__init__(
            message=f"Language '{language}' is not supported",
            details={"language": language, "supported_languages": supported},
        )


class SyntaxParseError(ParsingError):
    """Failed to parse source code syntax."""

    def __init__(self, file_path: str, line: int | None = None, cause: Exception | None = None) -> None:
        super().__init__(
            message=f"Failed to parse {file_path}" + (f" at line {line}" if line else ""),
            details={"file_path": file_path, "line": line},
            cause=cause,
        )


# =============================================================================
# Embedding Errors
# =============================================================================


class EmbeddingError(SmartSearchError):
    """Base class for embedding-related errors."""

    pass


class ModelLoadError(EmbeddingError):
    """Failed to load embedding model."""

    def __init__(self, model_name: str, cause: Exception | None = None) -> None:
        super().__init__(
            message=f"Failed to load model: {model_name}",
            details={"model_name": model_name},
            cause=cause,
        )


class EmbeddingGenerationError(EmbeddingError):
    """Failed to generate embeddings."""

    def __init__(self, text_length: int, cause: Exception | None = None) -> None:
        super().__init__(
            message="Failed to generate embedding",
            details={"text_length": text_length},
            cause=cause,
        )


# =============================================================================
# Indexing Errors
# =============================================================================


class IndexingError(SmartSearchError):
    """Base class for indexing-related errors."""

    pass


class FileNotFoundError(IndexingError):
    """File not found during indexing."""

    def __init__(self, file_path: str) -> None:
        super().__init__(
            message=f"File not found: {file_path}",
            details={"file_path": file_path},
        )


class InvalidPathError(IndexingError):
    """Invalid path provided."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            message=f"Invalid path: {path}. {reason}",
            details={"path": path, "reason": reason},
        )


# =============================================================================
# RAG Errors
# =============================================================================


class RAGError(SmartSearchError):
    """Base class for RAG-related errors."""

    pass


class RetrievalError(RAGError):
    """Error during retrieval phase."""

    def __init__(self, query: str, cause: Exception | None = None) -> None:
        super().__init__(
            message="Retrieval failed",
            details={"query": query},
            cause=cause,
        )


class GenerationError(RAGError):
    """Error during generation phase."""

    def __init__(self, cause: Exception | None = None) -> None:
        super().__init__(
            message="Generation failed",
            cause=cause,
        )
