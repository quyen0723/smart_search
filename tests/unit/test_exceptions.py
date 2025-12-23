"""Unit tests for custom exceptions."""

import pytest

from smart_search.core.exceptions import (
    ConfigurationError,
    CyclicDependencyError,
    EdgeNotFoundError,
    EmbeddingError,
    EmbeddingGenerationError,
    FileNotFoundError,
    GenerationError,
    GraphError,
    GraphSerializationError,
    IndexingError,
    InvalidPathError,
    MissingConfigError,
    ModelLoadError,
    NodeNotFoundError,
    ParsingError,
    RAGError,
    RetrievalError,
    SearchConnectionError,
    SearchError,
    SearchIndexError,
    SearchQueryError,
    SearchTimeoutError,
    SmartSearchError,
    SyntaxParseError,
    UnsupportedLanguageError,
)


class TestSmartSearchError:
    """Tests for base SmartSearchError."""

    def test_basic_creation(self) -> None:
        """Test basic exception creation."""
        exc = SmartSearchError("Test error")
        assert exc.message == "Test error"
        assert exc.details == {}
        assert exc.cause is None

    def test_with_details(self) -> None:
        """Test exception with details."""
        exc = SmartSearchError("Test error", details={"key": "value"})
        assert exc.details == {"key": "value"}

    def test_with_cause(self) -> None:
        """Test exception with cause."""
        cause = ValueError("Original error")
        exc = SmartSearchError("Test error", cause=cause)
        assert exc.cause is cause

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        exc = SmartSearchError("Test error", details={"key": "value"})
        result = exc.to_dict()

        assert result["error"] == "SmartSearchError"
        assert result["message"] == "Test error"
        assert result["details"] == {"key": "value"}

    def test_to_dict_without_details(self) -> None:
        """Test to_dict without details."""
        exc = SmartSearchError("Test error")
        result = exc.to_dict()

        assert "error" in result
        assert "message" in result
        assert "details" not in result


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_configuration_error(self) -> None:
        """Test ConfigurationError."""
        exc = ConfigurationError("Config error")
        assert isinstance(exc, SmartSearchError)

    def test_missing_config_error(self) -> None:
        """Test MissingConfigError."""
        exc = MissingConfigError("API_KEY")
        assert "API_KEY" in exc.message
        assert exc.details["config_key"] == "API_KEY"


class TestSearchErrors:
    """Tests for search-related exceptions."""

    def test_search_error_base(self) -> None:
        """Test SearchError is subclass of SmartSearchError."""
        exc = SearchError("Search failed")
        assert isinstance(exc, SmartSearchError)

    def test_search_connection_error(self) -> None:
        """Test SearchConnectionError."""
        cause = ConnectionError("Connection refused")
        exc = SearchConnectionError("http://localhost:7700", cause=cause)

        assert "http://localhost:7700" in exc.message
        assert exc.details["url"] == "http://localhost:7700"
        assert exc.cause is cause

    def test_search_index_error(self) -> None:
        """Test SearchIndexError."""
        exc = SearchIndexError("code_units", "create")

        assert "code_units" in exc.message
        assert "create" in exc.message
        assert exc.details["index_name"] == "code_units"
        assert exc.details["operation"] == "create"

    def test_search_query_error(self) -> None:
        """Test SearchQueryError."""
        exc = SearchQueryError("SELECT * FROM")

        assert exc.details["query"] == "SELECT * FROM"

    def test_search_timeout_error(self) -> None:
        """Test SearchTimeoutError."""
        exc = SearchTimeoutError(5000)

        assert "5000ms" in exc.message
        assert exc.details["timeout_ms"] == 5000


class TestGraphErrors:
    """Tests for graph-related exceptions."""

    def test_graph_error_base(self) -> None:
        """Test GraphError is subclass of SmartSearchError."""
        exc = GraphError("Graph error")
        assert isinstance(exc, SmartSearchError)

    def test_node_not_found_error(self) -> None:
        """Test NodeNotFoundError."""
        exc = NodeNotFoundError("module::func")

        assert "module::func" in exc.message
        assert exc.details["node_id"] == "module::func"

    def test_edge_not_found_error(self) -> None:
        """Test EdgeNotFoundError."""
        exc = EdgeNotFoundError("source", "target")

        assert "source" in exc.message
        assert "target" in exc.message
        assert exc.details["source_id"] == "source"
        assert exc.details["target_id"] == "target"

    def test_graph_serialization_error(self) -> None:
        """Test GraphSerializationError."""
        exc = GraphSerializationError("load", "/path/to/graph.msgpack")

        assert "load" in exc.message
        assert exc.details["operation"] == "load"
        assert exc.details["path"] == "/path/to/graph.msgpack"

    def test_cyclic_dependency_error(self) -> None:
        """Test CyclicDependencyError."""
        cycle = ["a", "b", "c", "a"]
        exc = CyclicDependencyError(cycle)

        assert exc.details["cycle"] == cycle


class TestParsingErrors:
    """Tests for parsing-related exceptions."""

    def test_parsing_error_base(self) -> None:
        """Test ParsingError is subclass of SmartSearchError."""
        exc = ParsingError("Parse failed")
        assert isinstance(exc, SmartSearchError)

    def test_unsupported_language_error(self) -> None:
        """Test UnsupportedLanguageError."""
        exc = UnsupportedLanguageError("rust", ["python", "javascript"])

        assert "rust" in exc.message
        assert exc.details["language"] == "rust"
        assert exc.details["supported_languages"] == ["python", "javascript"]

    def test_syntax_parse_error(self) -> None:
        """Test SyntaxParseError."""
        exc = SyntaxParseError("/path/file.py", line=42)

        assert "/path/file.py" in exc.message
        assert "42" in exc.message
        assert exc.details["file_path"] == "/path/file.py"
        assert exc.details["line"] == 42

    def test_syntax_parse_error_without_line(self) -> None:
        """Test SyntaxParseError without line number."""
        exc = SyntaxParseError("/path/file.py")

        assert exc.details["line"] is None


class TestEmbeddingErrors:
    """Tests for embedding-related exceptions."""

    def test_embedding_error_base(self) -> None:
        """Test EmbeddingError is subclass of SmartSearchError."""
        exc = EmbeddingError("Embedding failed")
        assert isinstance(exc, SmartSearchError)

    def test_model_load_error(self) -> None:
        """Test ModelLoadError."""
        exc = ModelLoadError("bert-base")

        assert "bert-base" in exc.message
        assert exc.details["model_name"] == "bert-base"

    def test_embedding_generation_error(self) -> None:
        """Test EmbeddingGenerationError."""
        exc = EmbeddingGenerationError(10000)

        assert exc.details["text_length"] == 10000


class TestIndexingErrors:
    """Tests for indexing-related exceptions."""

    def test_indexing_error_base(self) -> None:
        """Test IndexingError is subclass of SmartSearchError."""
        exc = IndexingError("Index failed")
        assert isinstance(exc, SmartSearchError)

    def test_file_not_found_error(self) -> None:
        """Test FileNotFoundError."""
        exc = FileNotFoundError("/path/to/missing.py")

        assert "/path/to/missing.py" in exc.message
        assert exc.details["file_path"] == "/path/to/missing.py"

    def test_invalid_path_error(self) -> None:
        """Test InvalidPathError."""
        exc = InvalidPathError("/invalid/../path", "Path contains ..")

        assert "/invalid/../path" in exc.message
        assert exc.details["path"] == "/invalid/../path"
        assert exc.details["reason"] == "Path contains .."


class TestRAGErrors:
    """Tests for RAG-related exceptions."""

    def test_rag_error_base(self) -> None:
        """Test RAGError is subclass of SmartSearchError."""
        exc = RAGError("RAG failed")
        assert isinstance(exc, SmartSearchError)

    def test_retrieval_error(self) -> None:
        """Test RetrievalError."""
        exc = RetrievalError("What is this function?")

        assert exc.details["query"] == "What is this function?"

    def test_generation_error(self) -> None:
        """Test GenerationError."""
        cause = RuntimeError("LLM error")
        exc = GenerationError(cause=cause)

        assert exc.cause is cause


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_inherit_from_base(self) -> None:
        """Test all exceptions inherit from SmartSearchError."""
        exceptions = [
            ConfigurationError("test"),
            SearchError("test"),
            GraphError("test"),
            ParsingError("test"),
            EmbeddingError("test"),
            IndexingError("test"),
            RAGError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, SmartSearchError)

    def test_specific_exceptions_inherit_from_category(self) -> None:
        """Test specific exceptions inherit from category base."""
        assert isinstance(SearchConnectionError("url"), SearchError)
        assert isinstance(NodeNotFoundError("id"), GraphError)
        assert isinstance(UnsupportedLanguageError("lang", []), ParsingError)
        assert isinstance(ModelLoadError("model"), EmbeddingError)
        assert isinstance(InvalidPathError("path", "reason"), IndexingError)
        assert isinstance(RetrievalError("query"), RAGError)
