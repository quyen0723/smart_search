# Kế hoạch Triển khai: Hệ thống Định vị Mã nguồn Thông minh V2.1

## Tổng quan

Xây dựng hệ thống định vị mã nguồn thông minh với khả năng:
- **Hybrid Search**: Tìm kiếm kết hợp từ khóa + ngữ nghĩa (Meilisearch)
- **Graph Analysis**: Phân tích đồ thị phụ thuộc (RustworkX)
- **GraphRAG**: Trả lời câu hỏi ngôn ngữ tự nhiên về codebase
- **Incremental Indexing**: Đánh chỉ mục tăng trưởng thông minh

---

## 1. Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                        │
│   /search  /navigate  /analyze  /graph  /index  /health        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                          │
│   Query Router │ Response Aggregator │ Cache Manager            │
└─────────────────────────────────────────────────────────────────┘
          │                    │                    │
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│   Search    │      │    Graph    │      │   AI/RAG    │
│  Engine     │      │   Engine    │      │   Engine    │
│(Meilisearch)│      │ (RustworkX) │      │ (GraphRAG)  │
└─────────────┘      └─────────────┘      └─────────────┘
          │                    │                    │
┌─────────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                         │
│  AST Parser (Tree-sitter) │ Embedding Pipeline │ Change Detector│
└─────────────────────────────────────────────────────────────────┘
          │
┌─────────────────────────────────────────────────────────────────┐
│                      Storage Layer                              │
│  Meilisearch Index │ Graph Store (MessagePack) │ Embedding Cache│
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Cấu trúc Thư mục

```
smart_search/
├── pyproject.toml
├── docker-compose.yml
├── .env.example
├── README.md
├── pytest.ini                         # Pytest configuration
├── .coveragerc                        # Coverage configuration
│
├── src/
│   └── smart_search/
│       ├── __init__.py
│       ├── main.py                    # FastAPI entry point
│       ├── config.py                  # Pydantic Settings
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── router.py
│       │   └── endpoints/
│       │       ├── search.py
│       │       ├── navigate.py
│       │       ├── analyze.py
│       │       ├── graph.py
│       │       ├── index.py
│       │       └── health.py
│       │
│       ├── core/
│       │   ├── orchestrator.py        # Query orchestration
│       │   ├── cache.py
│       │   └── exceptions.py
│       │
│       ├── parsing/
│       │   ├── tree_sitter_parser.py  # AST parsing
│       │   ├── chunker.py             # Code chunking
│       │   ├── models.py
│       │   └── extractors/
│       │       ├── python.py
│       │       ├── javascript.py
│       │       ├── java.py
│       │       └── go.py
│       │
│       ├── graph/
│       │   ├── engine.py              # RustworkX wrapper
│       │   ├── builder.py             # Graph construction
│       │   ├── algorithms.py          # Cycles, impact analysis
│       │   ├── persistence.py         # Serialization
│       │   └── models.py
│       │
│       ├── embedding/
│       │   ├── pipeline.py            # Embedding generation
│       │   ├── jina_embedder.py
│       │   └── cache.py
│       │
│       ├── search/
│       │   ├── meilisearch_client.py  # Async client
│       │   ├── hybrid.py              # Hybrid search logic
│       │   ├── indexer.py
│       │   └── schemas.py
│       │
│       ├── indexing/
│       │   ├── incremental.py         # Smart indexing
│       │   ├── watcher.py             # File system watch
│       │   ├── hasher.py
│       │   └── scheduler.py
│       │
│       ├── rag/
│       │   ├── graphrag.py            # GraphRAG core
│       │   ├── retriever.py
│       │   ├── generator.py
│       │   └── prompts.py
│       │
│       └── utils/
│           ├── logging.py
│           ├── git.py
│           └── file_utils.py
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures
│   │
│   ├── unit/                          # Unit tests (isolated, fast)
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── parsing/
│   │   │   ├── test_tree_sitter_parser.py
│   │   │   ├── test_chunker.py
│   │   │   ├── test_models.py
│   │   │   └── extractors/
│   │   │       ├── test_python_extractor.py
│   │   │       ├── test_javascript_extractor.py
│   │   │       └── test_java_extractor.py
│   │   ├── graph/
│   │   │   ├── test_engine.py
│   │   │   ├── test_builder.py
│   │   │   ├── test_algorithms.py
│   │   │   ├── test_persistence.py
│   │   │   └── test_models.py
│   │   ├── embedding/
│   │   │   ├── test_pipeline.py
│   │   │   ├── test_jina_embedder.py
│   │   │   └── test_cache.py
│   │   ├── search/
│   │   │   ├── test_meilisearch_client.py
│   │   │   ├── test_hybrid.py
│   │   │   ├── test_indexer.py
│   │   │   └── test_schemas.py
│   │   ├── indexing/
│   │   │   ├── test_incremental.py
│   │   │   ├── test_watcher.py
│   │   │   ├── test_hasher.py
│   │   │   └── test_scheduler.py
│   │   ├── rag/
│   │   │   ├── test_graphrag.py
│   │   │   ├── test_retriever.py
│   │   │   ├── test_generator.py
│   │   │   └── test_prompts.py
│   │   └── api/
│   │       ├── test_health.py
│   │       ├── test_search_endpoint.py
│   │       ├── test_navigate_endpoint.py
│   │       ├── test_analyze_endpoint.py
│   │       ├── test_graph_endpoint.py
│   │       └── test_index_endpoint.py
│   │
│   ├── integration/                   # Integration tests (với external services)
│   │   ├── __init__.py
│   │   ├── test_meilisearch_integration.py
│   │   ├── test_parsing_to_graph.py
│   │   ├── test_embedding_to_search.py
│   │   ├── test_full_indexing_pipeline.py
│   │   ├── test_graphrag_pipeline.py
│   │   └── test_api_integration.py
│   │
│   ├── e2e/                           # End-to-end tests
│   │   ├── __init__.py
│   │   ├── test_search_workflow.py
│   │   ├── test_navigation_workflow.py
│   │   ├── test_impact_analysis_workflow.py
│   │   └── test_incremental_update_workflow.py
│   │
│   ├── performance/                   # Performance & benchmark tests
│   │   ├── __init__.py
│   │   ├── test_search_latency.py
│   │   ├── test_graph_traversal_perf.py
│   │   ├── test_embedding_throughput.py
│   │   ├── test_indexing_speed.py
│   │   └── test_memory_usage.py
│   │
│   └── fixtures/                      # Test data
│       ├── sample_repos/
│       │   ├── python_simple/
│       │   ├── python_complex/
│       │   ├── javascript_project/
│       │   └── mixed_language/
│       ├── sample_code/
│       │   ├── python_samples.py
│       │   ├── javascript_samples.js
│       │   └── java_samples.java
│       ├── expected_outputs/
│       │   ├── parsed_ast.json
│       │   ├── expected_graph.json
│       │   └── expected_embeddings.json
│       └── mocks/
│           ├── mock_meilisearch_responses.json
│           └── mock_embeddings.json
│
├── scripts/
│   ├── setup_meilisearch.py
│   ├── benchmark.py
│   ├── index_repo.py
│   └── run_tests.sh                   # Test runner script
│
└── data/
    ├── indices/
    ├── graphs/
    └── embeddings_cache/
```

---

## 3. Technology Stack

| Component | Technology | Version | Lý do |
|-----------|------------|---------|-------|
| Language | Python | 3.11+ | Ecosystem, async support |
| Framework | FastAPI | 0.109+ | High performance, async |
| Search | Meilisearch | 1.6+ | Hybrid search, <50ms latency |
| Graph | RustworkX | 0.14+ | 3-100x faster than NetworkX |
| Parsing | tree-sitter | 0.25+ | Multi-language AST |
| Embedding | sentence-transformers | 2.2+ | Model wrapper |
| Model | jina-embeddings-v2-base-code | - | 8192 token context |
| Serialization | msgpack | 1.0+ | Fast graph serialization |
| Cache | diskcache | 5.6+ | Embedding cache |
| File Watch | watchdog | 4.0+ | Change detection |
| **Testing** | pytest | 8.0+ | Testing framework |
| **Testing** | pytest-asyncio | 0.23+ | Async test support |
| **Testing** | pytest-cov | 4.1+ | Coverage reporting |
| **Testing** | pytest-xdist | 3.5+ | Parallel test execution |
| **Testing** | pytest-mock | 3.12+ | Mocking support |
| **Testing** | hypothesis | 6.92+ | Property-based testing |
| **Testing** | faker | 22.0+ | Test data generation |
| **Testing** | httpx | 0.27+ | Async HTTP testing |
| **Testing** | testcontainers | 3.7+ | Container-based testing |

---

## 4. Testing Strategy Tổng thể

### 4.1 Testing Pyramid

```
                    ┌─────────────┐
                    │    E2E      │  (~10% - Slow, High confidence)
                    │   Tests     │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │     Integration         │  (~20% - Medium speed)
              │        Tests            │
              └────────────┬────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │            Unit Tests               │  (~70% - Fast, Isolated)
        │  (Mocking external dependencies)    │
        └─────────────────────────────────────┘
```

### 4.2 Coverage Requirements

| Component | Minimum Coverage | Target Coverage |
|-----------|-----------------|-----------------|
| Core Logic (graph, parsing) | 90% | 95% |
| Search/Indexing | 85% | 90% |
| API Endpoints | 80% | 90% |
| Utils | 70% | 80% |
| **Overall Project** | **85%** | **90%** |

### 4.3 Test Categories

| Category | Purpose | Speed | Isolation |
|----------|---------|-------|-----------|
| Unit | Test individual functions/classes | <1s per test | Full (mocked) |
| Integration | Test component interactions | <10s per test | Partial |
| E2E | Test complete workflows | <60s per test | None |
| Performance | Benchmark critical paths | Variable | None |

### 4.4 CI/CD Test Pipeline

```yaml
# .github/workflows/test.yml concept
stages:
  1. Lint & Type Check (ruff, mypy)
  2. Unit Tests (parallel, fast)
  3. Integration Tests (with testcontainers)
  4. E2E Tests (full system)
  5. Performance Tests (on main branch only)
  6. Coverage Report
```

---

## 5. Các Phase Triển khai (với Testing)

---

### Phase 1: Foundation (Core Infrastructure)

**Files cần tạo:**
- `pyproject.toml` - Dependencies và project config
- `docker-compose.yml` - Meilisearch container
- `src/smart_search/config.py` - Configuration management
- `src/smart_search/main.py` - FastAPI app
- `src/smart_search/api/endpoints/health.py` - Health check
- `src/smart_search/core/exceptions.py` - Custom exceptions
- `src/smart_search/utils/logging.py` - Logging setup

**Deliverables:**
- Project structure hoàn chỉnh
- Meilisearch running via Docker
- Basic FastAPI với health endpoint
- Logging framework

#### Phase 1 Testing

**Test Files:**
```
tests/
├── conftest.py                        # Global fixtures
├── unit/
│   ├── test_config.py
│   └── api/
│       └── test_health.py
└── integration/
    └── test_meilisearch_connection.py
```

**Test Cases - `tests/unit/test_config.py`:**
```python
class TestConfig:
    def test_default_config_values(self):
        """Verify default configuration values are set correctly."""

    def test_config_from_env_variables(self, monkeypatch):
        """Config should load from environment variables."""

    def test_config_validation_meilisearch_url(self):
        """Invalid Meilisearch URL should raise ValidationError."""

    def test_config_validation_port_range(self):
        """Port must be between 1-65535."""

    def test_config_sensitive_fields_not_in_repr(self):
        """API keys should not appear in string representation."""
```

**Test Cases - `tests/unit/api/test_health.py`:**
```python
class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, test_client):
        """GET /health should return 200 OK."""

    @pytest.mark.asyncio
    async def test_health_response_structure(self, test_client):
        """Health response should have status and version."""

    @pytest.mark.asyncio
    async def test_health_includes_dependencies_status(self, test_client):
        """Health should report Meilisearch connection status."""

    @pytest.mark.asyncio
    async def test_health_degraded_when_meilisearch_down(self, test_client, mock_meilisearch_down):
        """Health should return degraded status when Meilisearch unavailable."""
```

**Test Cases - `tests/integration/test_meilisearch_connection.py`:**
```python
@pytest.mark.integration
class TestMeilisearchConnection:
    @pytest.mark.asyncio
    async def test_can_connect_to_meilisearch(self, meilisearch_container):
        """Verify connection to Meilisearch container."""

    @pytest.mark.asyncio
    async def test_can_create_index(self, meilisearch_container):
        """Should be able to create a new index."""

    @pytest.mark.asyncio
    async def test_connection_retry_on_failure(self, meilisearch_container):
        """Should retry connection on transient failures."""
```

**Exit Criteria Phase 1:**
- [ ] All unit tests pass
- [ ] Integration test with Meilisearch passes
- [ ] Coverage >= 85%
- [ ] `pytest tests/unit tests/integration -v` passes

---

### Phase 2: AST Parsing & Code Units

**Files cần tạo:**
- `src/smart_search/parsing/tree_sitter_parser.py`
- `src/smart_search/parsing/chunker.py`
- `src/smart_search/parsing/models.py`
- `src/smart_search/parsing/extractors/python.py`
- `src/smart_search/parsing/extractors/javascript.py`

**Deliverables:**
- Parse Python/JavaScript files thành AST
- Extract code units (functions, classes, methods)
- AST-based chunking (không theo dòng)
- Metadata extraction (docstrings, imports)

#### Phase 2 Testing

**Test Files:**
```
tests/unit/parsing/
├── test_tree_sitter_parser.py
├── test_chunker.py
├── test_models.py
└── extractors/
    ├── test_python_extractor.py
    └── test_javascript_extractor.py
```

**Test Cases - `tests/unit/parsing/test_tree_sitter_parser.py`:**
```python
class TestTreeSitterParser:
    def test_parse_valid_python_file(self, sample_python_file):
        """Should parse valid Python file without errors."""

    def test_parse_valid_javascript_file(self, sample_js_file):
        """Should parse valid JavaScript file without errors."""

    def test_parse_syntax_error_returns_partial_ast(self):
        """Should return partial AST for files with syntax errors."""

    def test_detect_language_from_extension(self):
        """Should auto-detect language from file extension."""

    def test_detect_language_from_shebang(self):
        """Should detect language from shebang line."""

    def test_unsupported_language_raises_error(self):
        """Should raise UnsupportedLanguageError for unknown languages."""

    def test_empty_file_returns_empty_ast(self):
        """Empty file should return empty but valid AST."""

    def test_binary_file_raises_error(self):
        """Binary files should raise BinaryFileError."""

    def test_encoding_handling_utf8(self):
        """Should handle UTF-8 encoded files correctly."""

    def test_encoding_handling_latin1(self):
        """Should handle Latin-1 encoded files correctly."""
```

**Test Cases - `tests/unit/parsing/test_chunker.py`:**
```python
class TestCodeChunker:
    def test_chunk_by_function(self, parsed_python_ast):
        """Should create separate chunks for each function."""

    def test_chunk_by_class(self, parsed_python_ast):
        """Should create separate chunks for each class."""

    def test_chunk_includes_docstrings(self):
        """Chunks should include associated docstrings."""

    def test_chunk_preserves_decorators(self):
        """Function chunks should include decorators."""

    def test_chunk_nested_functions(self):
        """Nested functions should be separate chunks with parent reference."""

    def test_chunk_class_methods(self):
        """Class methods should be chunks with class parent reference."""

    def test_chunk_max_size_limit(self):
        """Large functions should be split if exceeding max chunk size."""

    def test_chunk_maintains_line_numbers(self):
        """Each chunk should have accurate start/end line numbers."""

    def test_chunk_module_level_code(self):
        """Module-level code should be in a separate chunk."""

    @pytest.mark.parametrize("code,expected_chunks", [
        ("def foo(): pass", 1),
        ("def foo(): pass\ndef bar(): pass", 2),
        ("class A:\n  def m(self): pass", 2),  # class + method
    ])
    def test_chunk_count(self, code, expected_chunks):
        """Verify correct number of chunks for various inputs."""
```

**Test Cases - `tests/unit/parsing/extractors/test_python_extractor.py`:**
```python
class TestPythonExtractor:
    def test_extract_function_name(self):
        """Should extract function names correctly."""

    def test_extract_function_parameters(self):
        """Should extract function parameters with types."""

    def test_extract_function_return_type(self):
        """Should extract return type annotations."""

    def test_extract_docstring_google_style(self):
        """Should extract Google-style docstrings."""

    def test_extract_docstring_numpy_style(self):
        """Should extract NumPy-style docstrings."""

    def test_extract_docstring_sphinx_style(self):
        """Should extract Sphinx-style docstrings."""

    def test_extract_class_inheritance(self):
        """Should extract parent classes."""

    def test_extract_imports_standard(self):
        """Should extract standard import statements."""

    def test_extract_imports_from(self):
        """Should extract from...import statements."""

    def test_extract_imports_relative(self):
        """Should extract relative imports correctly."""

    def test_extract_decorators(self):
        """Should extract decorator information."""

    def test_extract_async_functions(self):
        """Should correctly identify async functions."""

    def test_extract_class_variables(self):
        """Should extract class-level variables."""

    def test_extract_type_hints(self):
        """Should extract type hints from assignments."""

    # Property-based testing
    @given(st.text(min_size=1, alphabet=string.ascii_letters + "_"))
    def test_valid_identifiers_extracted(self, name):
        """Any valid Python identifier should be extractable."""
```

**Test Cases - `tests/unit/parsing/test_models.py`:**
```python
class TestCodeUnit:
    def test_code_unit_creation(self):
        """Should create CodeUnit with required fields."""

    def test_code_unit_id_generation(self):
        """ID should be deterministic hash of path + name."""

    def test_code_unit_id_uniqueness(self):
        """Different code units should have different IDs."""

    def test_code_unit_serialization(self):
        """Should serialize to dict correctly."""

    def test_code_unit_from_dict(self):
        """Should deserialize from dict correctly."""

class TestImport:
    def test_import_absolute(self):
        """Should parse absolute imports."""

    def test_import_relative(self):
        """Should parse relative imports with level."""

    def test_import_alias(self):
        """Should handle import aliases."""
```

**Integration Test - `tests/integration/test_parsing_to_graph.py`:**
```python
@pytest.mark.integration
class TestParsingToGraph:
    def test_parse_and_build_graph(self, sample_python_project):
        """Parse project and build dependency graph."""

    def test_import_edges_created(self, sample_python_project):
        """Import statements should create edges in graph."""

    def test_function_call_edges_created(self, sample_python_project):
        """Function calls should create edges in graph."""
```

**Exit Criteria Phase 2:**
- [ ] Parser handles Python and JavaScript
- [ ] Chunker creates logical code units
- [ ] All extractors tested with edge cases
- [ ] Coverage >= 90% for parsing module
- [ ] Property-based tests pass

---

### Phase 3: Graph Engine (RustworkX)

**Files cần tạo:**
- `src/smart_search/graph/engine.py` - **CRITICAL**
- `src/smart_search/graph/builder.py`
- `src/smart_search/graph/algorithms.py`
- `src/smart_search/graph/persistence.py`
- `src/smart_search/graph/models.py`

**Deliverables:**
- RustworkX-based CodeGraph class
- Node types: function, class, method, module, file
- Edge types: imports, calls, inherits, contains
- Algorithms: descendants, ancestors, cycles, impact analysis
- Graph serialization với MessagePack

#### Phase 3 Testing

**Test Files:**
```
tests/unit/graph/
├── test_engine.py
├── test_builder.py
├── test_algorithms.py
├── test_persistence.py
└── test_models.py
```

**Test Cases - `tests/unit/graph/test_engine.py`:**
```python
class TestCodeGraph:
    # === Node Operations ===
    def test_add_node(self):
        """Should add node and return index."""

    def test_add_duplicate_node_returns_existing(self):
        """Adding same node twice returns existing index."""

    def test_get_node_by_id(self):
        """Should retrieve node by ID."""

    def test_get_node_nonexistent_returns_none(self):
        """Getting nonexistent node returns None."""

    def test_remove_node(self):
        """Should remove node and associated edges."""

    def test_update_node(self):
        """Should update node data."""

    def test_node_count(self):
        """Should return correct node count."""

    # === Edge Operations ===
    def test_add_edge(self):
        """Should add directed edge between nodes."""

    def test_add_edge_nonexistent_source(self):
        """Adding edge with nonexistent source should fail gracefully."""

    def test_add_edge_nonexistent_target(self):
        """Adding edge with nonexistent target should fail gracefully."""

    def test_add_duplicate_edge(self):
        """Adding duplicate edge should not create multiple edges."""

    def test_remove_edge(self):
        """Should remove edge between nodes."""

    def test_get_edges_by_type(self):
        """Should filter edges by type."""

    def test_edge_count(self):
        """Should return correct edge count."""

    # === Traversal ===
    def test_get_descendants_direct(self):
        """Should get direct descendants (depth=1)."""

    def test_get_descendants_transitive(self):
        """Should get all descendants (no depth limit)."""

    def test_get_descendants_with_depth_limit(self):
        """Should respect depth limit."""

    def test_get_ancestors_direct(self):
        """Should get direct ancestors (depth=1)."""

    def test_get_ancestors_transitive(self):
        """Should get all ancestors (no depth limit)."""

    def test_get_successors(self):
        """Should get immediate successors."""

    def test_get_predecessors(self):
        """Should get immediate predecessors."""

    # === Graph Properties ===
    def test_is_empty(self):
        """Empty graph should return True for is_empty."""

    def test_has_node(self):
        """Should check if node exists."""

    def test_has_edge(self):
        """Should check if edge exists."""

    # === Edge Cases ===
    def test_self_loop_handling(self):
        """Should handle self-referencing edges."""

    def test_very_large_graph(self):
        """Should handle graph with 100k+ nodes efficiently."""

    def test_disconnected_components(self):
        """Should handle graphs with multiple components."""

class TestGraphQueries:
    def test_get_all_nodes_of_type(self):
        """Should filter nodes by type."""

    def test_get_nodes_in_file(self):
        """Should get all nodes in a specific file."""

    def test_get_callers(self):
        """Should get all nodes that call a function."""

    def test_get_callees(self):
        """Should get all nodes called by a function."""

    def test_subgraph_extraction(self):
        """Should extract subgraph around a node."""
```

**Test Cases - `tests/unit/graph/test_algorithms.py`:**
```python
class TestCycleDetection:
    def test_no_cycles_in_dag(self, sample_dag):
        """DAG should have no cycles."""

    def test_detect_simple_cycle(self):
        """Should detect A -> B -> A cycle."""

    def test_detect_complex_cycles(self):
        """Should detect multiple interconnected cycles."""

    def test_cycle_members(self):
        """Should return all nodes in cycle."""

    def test_self_loop_as_cycle(self):
        """Self-loop should be detected as cycle."""

class TestImpactAnalysis:
    def test_impact_direct_dependents(self):
        """Should find direct dependents of changed node."""

    def test_impact_transitive_dependents(self):
        """Should find all transitive dependents."""

    def test_impact_with_depth_limit(self):
        """Should respect depth limit in impact analysis."""

    def test_impact_multiple_changes(self):
        """Should compute union of impacts for multiple changes."""

    def test_impact_includes_distance(self):
        """Impact result should include distance from change."""

    def test_no_impact_for_isolated_node(self):
        """Isolated node should have no impact."""

class TestPathFinding:
    def test_shortest_path_exists(self):
        """Should find shortest path between connected nodes."""

    def test_shortest_path_not_exists(self):
        """Should return None for disconnected nodes."""

    def test_all_paths_between_nodes(self):
        """Should find all paths up to a limit."""

    def test_path_with_edge_types(self):
        """Should find path using specific edge types only."""

class TestCentrality:
    def test_betweenness_centrality(self):
        """Should compute betweenness centrality."""

    def test_in_degree_centrality(self):
        """Should compute in-degree centrality."""

    def test_out_degree_centrality(self):
        """Should compute out-degree centrality."""

    def test_hub_identification(self):
        """Should identify hub nodes (high centrality)."""

class TestBridgesAndCutPoints:
    def test_find_bridges(self):
        """Should find bridge edges."""

    def test_find_cut_points(self):
        """Should find articulation points."""

    def test_no_bridges_in_strongly_connected(self):
        """Strongly connected graph should have no bridges."""
```

**Test Cases - `tests/unit/graph/test_persistence.py`:**
```python
class TestGraphSerialization:
    def test_serialize_empty_graph(self, tmp_path):
        """Should serialize empty graph."""

    def test_serialize_simple_graph(self, sample_graph, tmp_path):
        """Should serialize graph with nodes and edges."""

    def test_deserialize_matches_original(self, sample_graph, tmp_path):
        """Deserialized graph should match original."""

    def test_serialize_preserves_node_data(self):
        """All node attributes should be preserved."""

    def test_serialize_preserves_edge_data(self):
        """All edge attributes should be preserved."""

    def test_serialize_large_graph(self, large_graph, tmp_path):
        """Should handle large graphs efficiently."""

    def test_file_corruption_detection(self, tmp_path):
        """Should detect corrupted graph files."""

    def test_version_migration(self, old_format_graph):
        """Should migrate old graph format to new."""

class TestIncrementalGraphUpdate:
    def test_add_node_to_existing(self, persisted_graph):
        """Should add new node to existing graph."""

    def test_remove_node_from_existing(self, persisted_graph):
        """Should remove node from existing graph."""

    def test_update_node_in_existing(self, persisted_graph):
        """Should update node in existing graph."""

    def test_partial_save(self, large_graph):
        """Should support saving only changed portions."""
```

**Test Cases - `tests/unit/graph/test_builder.py`:**
```python
class TestGraphBuilder:
    def test_build_from_code_units(self, code_units):
        """Should build graph from parsed code units."""

    def test_import_edges_created(self):
        """Import statements should create IMPORTS edges."""

    def test_call_edges_created(self):
        """Function calls should create CALLS edges."""

    def test_inheritance_edges_created(self):
        """Class inheritance should create INHERITS edges."""

    def test_contains_edges_for_methods(self):
        """Classes should have CONTAINS edges to methods."""

    def test_file_contains_module(self):
        """Files should have CONTAINS edge to module."""

    def test_incremental_build(self):
        """Should update graph incrementally from changes."""
```

**Performance Tests - `tests/performance/test_graph_traversal_perf.py`:**
```python
@pytest.mark.performance
class TestGraphPerformance:
    @pytest.mark.benchmark
    def test_descendants_1k_nodes(self, benchmark, graph_1k):
        """Descendants query on 1K nodes should be <1ms."""
        result = benchmark(graph_1k.get_descendants, "node_500")
        assert benchmark.stats['mean'] < 0.001  # 1ms

    @pytest.mark.benchmark
    def test_descendants_100k_nodes(self, benchmark, graph_100k):
        """Descendants query on 100K nodes should be <10ms."""
        result = benchmark(graph_100k.get_descendants, "node_50000")
        assert benchmark.stats['mean'] < 0.01  # 10ms

    @pytest.mark.benchmark
    def test_cycle_detection_1k_nodes(self, benchmark, graph_1k_with_cycles):
        """Cycle detection on 1K nodes should be <100ms."""

    @pytest.mark.benchmark
    def test_impact_analysis_1k_nodes(self, benchmark, graph_1k):
        """Impact analysis on 1K nodes should be <50ms."""

    def test_memory_usage_100k_nodes(self, graph_100k):
        """100K node graph should use <50MB memory."""
        import tracemalloc
        tracemalloc.start()
        # ... operations
        current, peak = tracemalloc.get_traced_memory()
        assert peak < 50 * 1024 * 1024  # 50MB
```

**Exit Criteria Phase 3:**
- [ ] All graph operations tested
- [ ] Cycle detection works correctly
- [ ] Impact analysis verified
- [ ] Serialization/deserialization works
- [ ] Performance targets met (<10ms traversal)
- [ ] Memory usage <50MB for 100K nodes
- [ ] Coverage >= 95% for graph module

---

### Phase 4: Embedding Pipeline

**Files cần tạo:**
- `src/smart_search/embedding/pipeline.py` - **CRITICAL**
- `src/smart_search/embedding/jina_embedder.py`
- `src/smart_search/embedding/cache.py`

**Deliverables:**
- Jina embeddings integration (768 dimensions)
- Batch embedding generation
- Disk-based embedding cache
- Cache invalidation on model updates

#### Phase 4 Testing

**Test Files:**
```
tests/unit/embedding/
├── test_pipeline.py
├── test_jina_embedder.py
└── test_cache.py
```

**Test Cases - `tests/unit/embedding/test_pipeline.py`:**
```python
class TestEmbeddingPipeline:
    def test_embed_single_text(self, pipeline):
        """Should generate embedding for single text."""

    def test_embed_batch(self, pipeline):
        """Should generate embeddings for batch of texts."""

    def test_embedding_dimension(self, pipeline):
        """Embedding should have correct dimensions (768)."""

    def test_embedding_normalized(self, pipeline):
        """Embeddings should be L2 normalized."""

    def test_empty_text_handling(self, pipeline):
        """Should handle empty string gracefully."""

    def test_very_long_text_truncation(self, pipeline):
        """Should truncate text exceeding max length."""

    def test_batch_size_respected(self, pipeline):
        """Should process in specified batch sizes."""

    def test_unicode_handling(self, pipeline):
        """Should handle Unicode characters correctly."""

    def test_code_specific_embedding(self, pipeline):
        """Code snippets should produce meaningful embeddings."""

    def test_similar_code_similar_embedding(self, pipeline):
        """Similar code should have high cosine similarity."""

    def test_different_code_different_embedding(self, pipeline):
        """Different code should have low cosine similarity."""

class TestEmbeddingWithMock:
    """Tests using mocked model for speed."""

    def test_pipeline_initialization(self, mock_model):
        """Pipeline should initialize with mocked model."""

    def test_pipeline_device_selection(self):
        """Should select GPU when available."""

    def test_pipeline_fallback_to_cpu(self):
        """Should fallback to CPU when GPU unavailable."""
```

**Test Cases - `tests/unit/embedding/test_cache.py`:**
```python
class TestEmbeddingCache:
    def test_cache_hit(self, cache, sample_embedding):
        """Should return cached embedding on hit."""

    def test_cache_miss(self, cache):
        """Should return None on cache miss."""

    def test_cache_store(self, cache, sample_embedding):
        """Should store embedding in cache."""

    def test_cache_key_generation(self, cache):
        """Cache key should be deterministic."""

    def test_cache_includes_model_version(self, cache):
        """Cache key should include model version."""

    def test_cache_invalidation(self, cache):
        """Should invalidate cache on model change."""

    def test_cache_persistence(self, cache, tmp_path):
        """Cache should persist across restarts."""

    def test_cache_size_limit(self, cache):
        """Cache should respect size limit."""

    def test_cache_eviction_policy(self, cache):
        """Should evict LRU items when full."""

    def test_cache_stats(self, cache):
        """Should report hit/miss statistics."""

    def test_concurrent_access(self, cache):
        """Should handle concurrent reads/writes."""
```

**Performance Tests - `tests/performance/test_embedding_throughput.py`:**
```python
@pytest.mark.performance
class TestEmbeddingPerformance:
    @pytest.mark.benchmark
    def test_single_embedding_latency(self, benchmark, pipeline):
        """Single embedding should be <100ms."""

    @pytest.mark.benchmark
    def test_batch_32_throughput(self, benchmark, pipeline):
        """Batch of 32 should process <1s."""

    @pytest.mark.benchmark
    def test_cache_speedup(self, benchmark, pipeline_with_cache):
        """Cached embedding should be <1ms."""

    def test_memory_usage_batch_processing(self, pipeline):
        """Batch processing should not leak memory."""
```

**Exit Criteria Phase 4:**
- [ ] Embeddings generate correctly (768D)
- [ ] Batch processing works
- [ ] Cache hit/miss works correctly
- [ ] Model version in cache key
- [ ] Performance targets met
- [ ] Coverage >= 90%

---

### Phase 5: Search Engine (Meilisearch)

**Files cần tạo:**
- `src/smart_search/search/meilisearch_client.py` - **CRITICAL**
- `src/smart_search/search/hybrid.py`
- `src/smart_search/search/indexer.py`
- `src/smart_search/search/schemas.py`

**Deliverables:**
- Async Meilisearch client
- Hybrid search (lexical + semantic)
- Index configuration với embedders
- Batched document ingestion
- Faceted filtering

#### Phase 5 Testing

**Test Files:**
```
tests/unit/search/
├── test_meilisearch_client.py
├── test_hybrid.py
├── test_indexer.py
└── test_schemas.py

tests/integration/
└── test_meilisearch_integration.py
```

**Test Cases - `tests/unit/search/test_meilisearch_client.py`:**
```python
class TestMeilisearchClient:
    # === Connection ===
    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """Should initialize with host and API key."""

    @pytest.mark.asyncio
    async def test_client_close(self, client):
        """Should close connection properly."""

    # === Index Management ===
    @pytest.mark.asyncio
    async def test_create_index(self, mock_meilisearch):
        """Should create index with primary key."""

    @pytest.mark.asyncio
    async def test_delete_index(self, mock_meilisearch):
        """Should delete existing index."""

    @pytest.mark.asyncio
    async def test_configure_index_settings(self, mock_meilisearch):
        """Should configure searchable/filterable attributes."""

    @pytest.mark.asyncio
    async def test_configure_embedders(self, mock_meilisearch):
        """Should configure vector embedders."""

    # === Document Operations ===
    @pytest.mark.asyncio
    async def test_add_documents(self, mock_meilisearch):
        """Should add documents to index."""

    @pytest.mark.asyncio
    async def test_add_documents_batching(self, mock_meilisearch):
        """Should batch documents to respect size limit."""

    @pytest.mark.asyncio
    async def test_update_documents(self, mock_meilisearch):
        """Should update existing documents."""

    @pytest.mark.asyncio
    async def test_delete_documents(self, mock_meilisearch):
        """Should delete documents by ID."""

    # === Search ===
    @pytest.mark.asyncio
    async def test_keyword_search(self, mock_meilisearch):
        """Should perform keyword-only search."""

    @pytest.mark.asyncio
    async def test_hybrid_search(self, mock_meilisearch):
        """Should perform hybrid search with semantic ratio."""

    @pytest.mark.asyncio
    async def test_search_with_filters(self, mock_meilisearch):
        """Should apply filters to search."""

    @pytest.mark.asyncio
    async def test_search_pagination(self, mock_meilisearch):
        """Should handle limit and offset."""

    # === Error Handling ===
    @pytest.mark.asyncio
    async def test_connection_error_handling(self):
        """Should handle connection errors gracefully."""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Should handle request timeouts."""

    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Should handle rate limit responses."""

    # === Task Management ===
    @pytest.mark.asyncio
    async def test_wait_for_task(self, mock_meilisearch):
        """Should wait for async task completion."""

    @pytest.mark.asyncio
    async def test_task_timeout(self, mock_meilisearch):
        """Should timeout if task takes too long."""
```

**Test Cases - `tests/unit/search/test_hybrid.py`:**
```python
class TestHybridSearch:
    def test_semantic_ratio_0_keyword_only(self):
        """semantic_ratio=0 should be keyword-only search."""

    def test_semantic_ratio_1_semantic_only(self):
        """semantic_ratio=1 should be semantic-only search."""

    def test_semantic_ratio_balanced(self):
        """semantic_ratio=0.5 should balance both."""

    def test_result_ranking_combined(self):
        """Results should be ranked by combined score."""

    def test_result_includes_both_scores(self):
        """Each result should have keyword and semantic scores."""
```

**Test Cases - `tests/unit/search/test_indexer.py`:**
```python
class TestSearchIndexer:
    @pytest.mark.asyncio
    async def test_index_code_units(self, indexer, code_units):
        """Should index code units with embeddings."""

    @pytest.mark.asyncio
    async def test_index_with_graph_context(self, indexer, code_units, graph):
        """Should include graph context in documents."""

    @pytest.mark.asyncio
    async def test_batch_indexing(self, indexer, large_code_units):
        """Should batch index large number of documents."""

    @pytest.mark.asyncio
    async def test_reindex_changed_documents(self, indexer):
        """Should reindex only changed documents."""

    @pytest.mark.asyncio
    async def test_remove_deleted_documents(self, indexer):
        """Should remove documents for deleted code."""
```

**Integration Tests - `tests/integration/test_meilisearch_integration.py`:**
```python
@pytest.mark.integration
class TestMeilisearchIntegration:
    @pytest.mark.asyncio
    async def test_full_indexing_workflow(self, meilisearch_container, sample_project):
        """Full workflow: parse -> embed -> index -> search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_relevance(self, meilisearch_container, indexed_project):
        """Hybrid search should return relevant results."""

    @pytest.mark.asyncio
    async def test_filter_by_language(self, meilisearch_container, indexed_project):
        """Should filter results by programming language."""

    @pytest.mark.asyncio
    async def test_filter_by_file_path(self, meilisearch_container, indexed_project):
        """Should filter results by file path pattern."""

    @pytest.mark.asyncio
    async def test_faceted_search(self, meilisearch_container, indexed_project):
        """Should return facet counts."""
```

**Performance Tests - `tests/performance/test_search_latency.py`:**
```python
@pytest.mark.performance
class TestSearchPerformance:
    @pytest.mark.benchmark
    async def test_search_latency_1k_docs(self, benchmark, indexed_1k):
        """Search on 1K docs should be <50ms."""

    @pytest.mark.benchmark
    async def test_search_latency_100k_docs(self, benchmark, indexed_100k):
        """Search on 100K docs should be <50ms."""

    @pytest.mark.benchmark
    async def test_hybrid_search_latency(self, benchmark, indexed_100k):
        """Hybrid search should be <100ms."""

    @pytest.mark.benchmark
    async def test_indexing_throughput(self, benchmark, meilisearch_container):
        """Should index 1000 docs/second."""
```

**Exit Criteria Phase 5:**
- [ ] Meilisearch client works with all operations
- [ ] Hybrid search combines keyword + semantic
- [ ] Filtering works correctly
- [ ] Batching respects size limits
- [ ] Search latency <50ms (P95)
- [ ] Coverage >= 90%

---

### Phase 6: Incremental Indexing

**Files cần tạo:**
- `src/smart_search/indexing/incremental.py` - **CRITICAL**
- `src/smart_search/indexing/watcher.py`
- `src/smart_search/indexing/hasher.py`
- `src/smart_search/indexing/scheduler.py`

**Deliverables:**
- Content hash-based change detection
- Invalidation set computation
- Partial graph updates
- File system watcher integration
- State persistence

#### Phase 6 Testing

**Test Files:**
```
tests/unit/indexing/
├── test_incremental.py
├── test_watcher.py
├── test_hasher.py
└── test_scheduler.py
```

**Test Cases - `tests/unit/indexing/test_hasher.py`:**
```python
class TestContentHasher:
    def test_hash_file(self, sample_file):
        """Should compute SHA256 hash of file content."""

    def test_hash_deterministic(self, sample_file):
        """Same content should produce same hash."""

    def test_hash_different_content(self):
        """Different content should produce different hash."""

    def test_hash_ignores_whitespace_changes(self):
        """Trailing whitespace changes should not affect hash."""
        # Note: Configurable based on requirements

    def test_hash_large_file(self, large_file):
        """Should handle large files efficiently."""

    def test_hash_binary_file(self, binary_file):
        """Should handle binary files."""
```

**Test Cases - `tests/unit/indexing/test_incremental.py`:**
```python
class TestChangeDetection:
    def test_detect_new_files(self, tmp_repo):
        """Should detect newly added files."""

    def test_detect_modified_files(self, tmp_repo):
        """Should detect modified files."""

    def test_detect_deleted_files(self, tmp_repo):
        """Should detect deleted files."""

    def test_detect_renamed_files(self, tmp_repo):
        """Should detect renamed files as delete + add."""

    def test_no_changes_detected(self, tmp_repo):
        """Should return empty changeset when no changes."""

    def test_ignore_patterns(self, tmp_repo):
        """Should ignore files matching patterns (node_modules, etc)."""

    def test_respect_gitignore(self, tmp_repo_with_gitignore):
        """Should respect .gitignore patterns."""

class TestInvalidationSet:
    def test_direct_dependents_included(self, indexer, graph):
        """Direct dependents should be in invalidation set."""

    def test_transitive_dependents_included(self, indexer, graph):
        """Transitive dependents should be in invalidation set."""

    def test_depth_limit_respected(self, indexer, graph):
        """Should respect depth limit for invalidation."""

    def test_deleted_file_dependents(self, indexer, graph):
        """Deleted file should invalidate all dependents."""

    def test_modified_file_dependents(self, indexer, graph):
        """Modified file should invalidate direct dependents."""

class TestIncrementalProcessing:
    @pytest.mark.asyncio
    async def test_process_added_files(self, indexer, tmp_repo):
        """Should parse, embed, and index new files."""

    @pytest.mark.asyncio
    async def test_process_modified_files(self, indexer, tmp_repo):
        """Should update embeddings and search index."""

    @pytest.mark.asyncio
    async def test_process_deleted_files(self, indexer, tmp_repo):
        """Should remove from graph and search index."""

    @pytest.mark.asyncio
    async def test_graph_consistency_after_update(self, indexer, tmp_repo):
        """Graph should remain consistent after updates."""

    @pytest.mark.asyncio
    async def test_search_index_consistency_after_update(self, indexer, tmp_repo):
        """Search index should remain consistent after updates."""

    @pytest.mark.asyncio
    async def test_atomic_update(self, indexer, tmp_repo):
        """Updates should be atomic (all or nothing)."""

    @pytest.mark.asyncio
    async def test_rollback_on_failure(self, indexer, tmp_repo):
        """Should rollback on partial failure."""

class TestStatePersistence:
    def test_save_state(self, indexer, tmp_path):
        """Should save indexing state to file."""

    def test_load_state(self, indexer, tmp_path):
        """Should load indexing state from file."""

    def test_state_includes_file_hashes(self, indexer):
        """State should include all file hashes."""

    def test_state_includes_symbol_ids(self, indexer):
        """State should include symbol IDs per file."""

    def test_state_recovery_after_crash(self, indexer, tmp_path):
        """Should recover state after crash."""
```

**Test Cases - `tests/unit/indexing/test_watcher.py`:**
```python
class TestFileWatcher:
    def test_watch_directory(self, tmp_repo):
        """Should start watching directory."""

    def test_detect_file_created(self, watcher, tmp_repo):
        """Should detect when file is created."""

    def test_detect_file_modified(self, watcher, tmp_repo):
        """Should detect when file is modified."""

    def test_detect_file_deleted(self, watcher, tmp_repo):
        """Should detect when file is deleted."""

    def test_debounce_rapid_changes(self, watcher, tmp_repo):
        """Should debounce rapid successive changes."""

    def test_ignore_non_code_files(self, watcher, tmp_repo):
        """Should ignore non-code files (images, etc)."""

    def test_stop_watching(self, watcher):
        """Should stop watching cleanly."""

    def test_recursive_watching(self, watcher, tmp_repo):
        """Should watch subdirectories."""
```

**Test Cases - `tests/unit/indexing/test_scheduler.py`:**
```python
class TestIndexingScheduler:
    def test_schedule_immediate(self, scheduler):
        """Should process immediately when idle."""

    def test_schedule_debounced(self, scheduler):
        """Should debounce rapid scheduling."""

    def test_queue_during_processing(self, scheduler):
        """Should queue changes during processing."""

    def test_priority_ordering(self, scheduler):
        """Should process higher priority changes first."""

    def test_concurrent_limit(self, scheduler):
        """Should respect concurrent processing limit."""
```

**Integration Tests:**
```python
@pytest.mark.integration
class TestIncrementalIndexingIntegration:
    @pytest.mark.asyncio
    async def test_full_incremental_workflow(self, meilisearch_container, tmp_repo):
        """Full workflow: change file -> detect -> update index."""

    @pytest.mark.asyncio
    async def test_search_finds_new_code(self, meilisearch_container, tmp_repo):
        """New code should be searchable after indexing."""

    @pytest.mark.asyncio
    async def test_search_excludes_deleted_code(self, meilisearch_container, tmp_repo):
        """Deleted code should not appear in search."""
```

**Exit Criteria Phase 6:**
- [ ] Change detection works for all scenarios
- [ ] Invalidation set computed correctly
- [ ] State persists across restarts
- [ ] File watcher detects changes
- [ ] Atomic updates with rollback
- [ ] Coverage >= 90%

---

### Phase 7: GraphRAG Implementation

**Files cần tạo:**
- `src/smart_search/rag/graphrag.py` - **CRITICAL**
- `src/smart_search/rag/retriever.py`
- `src/smart_search/rag/generator.py`
- `src/smart_search/rag/prompts.py`

**Deliverables:**
- Query types: LOCAL, GLOBAL, DRIFT
- Vector retrieval + Graph expansion
- Explanation path generation
- Community detection
- LLM integration (optional)

#### Phase 7 Testing

**Test Files:**
```
tests/unit/rag/
├── test_graphrag.py
├── test_retriever.py
├── test_generator.py
└── test_prompts.py
```

**Test Cases - `tests/unit/rag/test_retriever.py`:**
```python
class TestContextRetriever:
    @pytest.mark.asyncio
    async def test_retrieve_by_vector_similarity(self, retriever, indexed_project):
        """Should retrieve code by vector similarity."""

    @pytest.mark.asyncio
    async def test_expand_with_graph_context(self, retriever, indexed_project):
        """Should expand results using graph relationships."""

    @pytest.mark.asyncio
    async def test_local_query_type(self, retriever, indexed_project):
        """LOCAL query should focus on direct relationships."""

    @pytest.mark.asyncio
    async def test_global_query_type(self, retriever, indexed_project):
        """GLOBAL query should include community context."""

    @pytest.mark.asyncio
    async def test_drift_query_type(self, retriever, indexed_project):
        """DRIFT query should combine local and global."""

    @pytest.mark.asyncio
    async def test_max_context_items_limit(self, retriever):
        """Should respect max context items limit."""

    @pytest.mark.asyncio
    async def test_graph_depth_limit(self, retriever):
        """Should respect graph traversal depth limit."""
```

**Test Cases - `tests/unit/rag/test_graphrag.py`:**
```python
class TestGraphRAGProcessor:
    @pytest.mark.asyncio
    async def test_query_returns_context(self, graphrag, indexed_project):
        """Query should return retrieved context."""

    @pytest.mark.asyncio
    async def test_query_includes_paths(self, graphrag, indexed_project):
        """Query result should include relationship paths."""

    @pytest.mark.asyncio
    async def test_path_explanation_generation(self, graphrag):
        """Should generate A -> B -> C path explanations."""

    @pytest.mark.asyncio
    async def test_community_summaries_included(self, graphrag, indexed_project):
        """Should include community summaries in context."""

    @pytest.mark.asyncio
    async def test_relevance_scores_included(self, graphrag):
        """Results should include relevance scores."""

class TestCommunityDetection:
    def test_detect_communities(self, graph):
        """Should detect code communities/modules."""

    def test_community_membership(self, graph):
        """Each node should belong to a community."""

    def test_community_summary_generation(self, graphrag, graph):
        """Should generate summary for each community."""
```

**Test Cases - `tests/unit/rag/test_generator.py`:**
```python
class TestResponseGenerator:
    @pytest.mark.asyncio
    async def test_generate_without_llm(self, generator):
        """Should format context as structured response without LLM."""

    @pytest.mark.asyncio
    async def test_generate_with_llm(self, generator, mock_llm):
        """Should generate natural language response with LLM."""

    @pytest.mark.asyncio
    async def test_prompt_includes_code_context(self, generator, mock_llm):
        """Prompt should include retrieved code."""

    @pytest.mark.asyncio
    async def test_prompt_includes_relationships(self, generator, mock_llm):
        """Prompt should include relationship paths."""

    @pytest.mark.asyncio
    async def test_response_cites_sources(self, generator, mock_llm):
        """Response should cite source files/functions."""

class TestPromptBuilder:
    def test_build_code_context_prompt(self, prompt_builder):
        """Should build prompt with code snippets."""

    def test_build_relationship_prompt(self, prompt_builder):
        """Should build prompt with relationships."""

    def test_prompt_length_limit(self, prompt_builder):
        """Should respect max prompt length."""

    def test_prompt_prioritizes_relevant_context(self, prompt_builder):
        """Should prioritize most relevant context."""
```

**Integration Tests:**
```python
@pytest.mark.integration
class TestGraphRAGIntegration:
    @pytest.mark.asyncio
    async def test_answer_code_question(self, graphrag_system, sample_project):
        """Should answer natural language question about code."""

    @pytest.mark.asyncio
    async def test_explain_function_purpose(self, graphrag_system, sample_project):
        """Should explain what a function does."""

    @pytest.mark.asyncio
    async def test_find_related_code(self, graphrag_system, sample_project):
        """Should find code related to a concept."""

    @pytest.mark.asyncio
    async def test_trace_dependency_path(self, graphrag_system, sample_project):
        """Should trace and explain dependency paths."""
```

**Exit Criteria Phase 7:**
- [ ] Query types work correctly
- [ ] Graph expansion enhances results
- [ ] Path explanations generated
- [ ] Works with and without LLM
- [ ] Coverage >= 85%

---

### Phase 8: API Layer

**Files cần tạo:**
- `src/smart_search/api/router.py`
- `src/smart_search/api/endpoints/search.py`
- `src/smart_search/api/endpoints/navigate.py`
- `src/smart_search/api/endpoints/analyze.py`
- `src/smart_search/api/endpoints/graph.py`
- `src/smart_search/api/endpoints/index.py`
- `src/smart_search/core/orchestrator.py`

**API Endpoints:**
```
POST /api/v1/search
POST /api/v1/navigate
POST /api/v1/analyze/impact
GET  /api/v1/graph/subgraph/{nodeId}
POST /api/v1/index/incremental
POST /api/v1/index/full
WebSocket /api/v1/ws/index-progress
```

#### Phase 8 Testing

**Test Files:**
```
tests/unit/api/
├── test_search_endpoint.py
├── test_navigate_endpoint.py
├── test_analyze_endpoint.py
├── test_graph_endpoint.py
└── test_index_endpoint.py

tests/e2e/
├── test_search_workflow.py
├── test_navigation_workflow.py
├── test_impact_analysis_workflow.py
└── test_incremental_update_workflow.py
```

**Test Cases - `tests/unit/api/test_search_endpoint.py`:**
```python
class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_200(self, test_client):
        """POST /search should return 200."""

    @pytest.mark.asyncio
    async def test_search_request_validation(self, test_client):
        """Should validate request body."""

    @pytest.mark.asyncio
    async def test_search_query_required(self, test_client):
        """Query field is required."""

    @pytest.mark.asyncio
    async def test_search_semantic_ratio_range(self, test_client):
        """semantic_ratio must be 0-1."""

    @pytest.mark.asyncio
    async def test_search_response_structure(self, test_client):
        """Response should have hits, total, processing_time."""

    @pytest.mark.asyncio
    async def test_search_with_filters(self, test_client):
        """Should apply language and path filters."""

    @pytest.mark.asyncio
    async def test_search_pagination(self, test_client):
        """Should handle limit and offset."""

    @pytest.mark.asyncio
    async def test_search_include_graph_context(self, test_client):
        """Should include graph context when requested."""

    @pytest.mark.asyncio
    async def test_search_error_handling(self, test_client, mock_search_error):
        """Should handle search errors gracefully."""
```

**Test Cases - `tests/unit/api/test_navigate_endpoint.py`:**
```python
class TestNavigateEndpoint:
    @pytest.mark.asyncio
    async def test_navigate_callers(self, test_client):
        """Should return callers of a function."""

    @pytest.mark.asyncio
    async def test_navigate_callees(self, test_client):
        """Should return callees of a function."""

    @pytest.mark.asyncio
    async def test_navigate_inherits(self, test_client):
        """Should return parent classes."""

    @pytest.mark.asyncio
    async def test_navigate_depth_limit(self, test_client):
        """Should respect depth limit."""

    @pytest.mark.asyncio
    async def test_navigate_nonexistent_symbol(self, test_client):
        """Should return 404 for nonexistent symbol."""
```

**Test Cases - `tests/unit/api/test_analyze_endpoint.py`:**
```python
class TestAnalyzeEndpoint:
    @pytest.mark.asyncio
    async def test_impact_analysis(self, test_client):
        """POST /analyze/impact should return affected files."""

    @pytest.mark.asyncio
    async def test_impact_includes_risk_score(self, test_client):
        """Response should include risk score."""

    @pytest.mark.asyncio
    async def test_impact_includes_cycles(self, test_client):
        """Response should include affected cycles."""

    @pytest.mark.asyncio
    async def test_impact_depth_limit(self, test_client):
        """Should respect analysis depth limit."""
```

**Test Cases - `tests/unit/api/test_index_endpoint.py`:**
```python
class TestIndexEndpoint:
    @pytest.mark.asyncio
    async def test_trigger_incremental_index(self, test_client):
        """POST /index/incremental should trigger indexing."""

    @pytest.mark.asyncio
    async def test_trigger_full_index(self, test_client):
        """POST /index/full should trigger full reindex."""

    @pytest.mark.asyncio
    async def test_index_returns_task_id(self, test_client):
        """Should return task ID for tracking."""

    @pytest.mark.asyncio
    async def test_websocket_progress(self, test_client):
        """WebSocket should receive progress updates."""
```

**E2E Tests - `tests/e2e/test_search_workflow.py`:**
```python
@pytest.mark.e2e
class TestSearchWorkflow:
    @pytest.mark.asyncio
    async def test_index_and_search_new_project(self, full_system, sample_project):
        """
        E2E: Index a project and search for code.
        1. Index sample project
        2. Wait for indexing complete
        3. Search for a function
        4. Verify result contains expected code
        """

    @pytest.mark.asyncio
    async def test_semantic_search_finds_concept(self, full_system, indexed_project):
        """
        E2E: Semantic search finds code by concept.
        Search: "authentication handler"
        Expect: Find login/auth functions
        """

    @pytest.mark.asyncio
    async def test_hybrid_search_balances_results(self, full_system, indexed_project):
        """
        E2E: Hybrid search balances keyword and semantic.
        """
```

**E2E Tests - `tests/e2e/test_navigation_workflow.py`:**
```python
@pytest.mark.e2e
class TestNavigationWorkflow:
    @pytest.mark.asyncio
    async def test_navigate_function_dependencies(self, full_system, indexed_project):
        """
        E2E: Navigate through function dependencies.
        1. Find a function
        2. Get its callees
        3. Navigate to each callee
        4. Verify graph consistency
        """

    @pytest.mark.asyncio
    async def test_navigate_class_hierarchy(self, full_system, indexed_project):
        """
        E2E: Navigate class inheritance hierarchy.
        """
```

**E2E Tests - `tests/e2e/test_impact_analysis_workflow.py`:**
```python
@pytest.mark.e2e
class TestImpactAnalysisWorkflow:
    @pytest.mark.asyncio
    async def test_impact_of_file_change(self, full_system, indexed_project):
        """
        E2E: Analyze impact of changing a file.
        1. Index project
        2. Request impact analysis for a file
        3. Verify affected files identified
        4. Verify risk score reasonable
        """
```

**E2E Tests - `tests/e2e/test_incremental_update_workflow.py`:**
```python
@pytest.mark.e2e
class TestIncrementalUpdateWorkflow:
    @pytest.mark.asyncio
    async def test_add_file_and_search(self, full_system, indexed_project):
        """
        E2E: Add new file and verify searchable.
        1. Index project
        2. Add new file
        3. Trigger incremental index
        4. Search finds new code
        """

    @pytest.mark.asyncio
    async def test_modify_file_and_search(self, full_system, indexed_project):
        """
        E2E: Modify file and verify updated in search.
        """

    @pytest.mark.asyncio
    async def test_delete_file_and_search(self, full_system, indexed_project):
        """
        E2E: Delete file and verify removed from search.
        """
```

**Exit Criteria Phase 8:**
- [ ] All API endpoints working
- [ ] Request validation complete
- [ ] Error handling consistent
- [ ] WebSocket for progress works
- [ ] E2E workflows pass
- [ ] Coverage >= 85%

---

## 6. Test Configuration Files

### pytest.ini
```ini
[pytest]
minversion = 8.0
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --tb=short
    -ra
markers =
    unit: Unit tests (fast, isolated)
    integration: Integration tests (with external services)
    e2e: End-to-end tests (full system)
    performance: Performance/benchmark tests
    slow: Slow tests (>10s)
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
```

### .coveragerc
```ini
[run]
source = src/smart_search
branch = True
omit =
    */tests/*
    */__pycache__/*
    */conftest.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if TYPE_CHECKING:
    if __name__ == "__main__":
show_missing = True
fail_under = 85

[html]
directory = coverage_html
```

### conftest.py (Shared Fixtures)
```python
# tests/conftest.py
import pytest
import asyncio
from pathlib import Path
from typing import Generator
import tempfile
import shutil

# === Async Event Loop ===
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# === Temporary Directories ===
@pytest.fixture
def tmp_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary repository with sample files."""
    repo = tmp_path / "test_repo"
    repo.mkdir()

    # Create sample Python files
    (repo / "main.py").write_text('''
def main():
    helper()

def helper():
    pass
''')

    (repo / "utils.py").write_text('''
from main import helper

def utility():
    helper()
''')

    yield repo
    shutil.rmtree(repo, ignore_errors=True)

# === Sample Code Fixtures ===
@pytest.fixture
def sample_python_file(tmp_path: Path) -> Path:
    """Create sample Python file for testing."""
    file = tmp_path / "sample.py"
    file.write_text('''
"""Module docstring."""

import os
from typing import List

class MyClass:
    """Class docstring."""

    def method(self, arg: str) -> List[str]:
        """Method docstring."""
        return [arg]

def standalone_function(x: int) -> int:
    """Function docstring."""
    return x * 2
''')
    return file

# === Mock Fixtures ===
@pytest.fixture
def mock_meilisearch(mocker):
    """Mock Meilisearch client."""
    mock = mocker.MagicMock()
    mock.search.return_value = {"hits": [], "processingTimeMs": 1}
    return mock

@pytest.fixture
def mock_embedding_model(mocker):
    """Mock embedding model for fast tests."""
    import numpy as np
    mock = mocker.MagicMock()
    mock.encode.return_value = np.random.rand(768).astype(np.float32)
    return mock

# === Test Client ===
@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from httpx import AsyncClient
    from smart_search.main import app

    async def _get_client():
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client

    return _get_client

# === Graph Fixtures ===
@pytest.fixture
def empty_graph():
    """Create empty CodeGraph."""
    from smart_search.graph.engine import CodeGraph
    return CodeGraph()

@pytest.fixture
def sample_graph(empty_graph):
    """Create sample graph with nodes and edges."""
    from smart_search.graph.models import GraphNode, GraphEdge, NodeType, EdgeType

    graph = empty_graph

    # Add nodes
    nodes = [
        GraphNode(id="main", name="main", type=NodeType.FUNCTION,
                 file_path="main.py", line_start=1, line_end=5,
                 language="python", content_hash="abc", metadata={}),
        GraphNode(id="helper", name="helper", type=NodeType.FUNCTION,
                 file_path="main.py", line_start=7, line_end=10,
                 language="python", content_hash="def", metadata={}),
        GraphNode(id="utility", name="utility", type=NodeType.FUNCTION,
                 file_path="utils.py", line_start=1, line_end=5,
                 language="python", content_hash="ghi", metadata={}),
    ]

    for node in nodes:
        graph.add_node(node)

    # Add edges
    graph.add_edge("main", "helper", GraphEdge(type=EdgeType.CALLS, metadata={}))
    graph.add_edge("utility", "helper", GraphEdge(type=EdgeType.CALLS, metadata={}))

    return graph

# === Container Fixtures (for integration tests) ===
@pytest.fixture(scope="session")
def meilisearch_container():
    """Start Meilisearch container for integration tests."""
    try:
        from testcontainers.meilisearch import MeilisearchContainer

        with MeilisearchContainer() as container:
            yield container.get_client()
    except ImportError:
        pytest.skip("testcontainers not installed")
```

---

## 7. Test Runner Script

```bash
#!/bin/bash
# scripts/run_tests.sh

set -e

echo "=== Smart Search Test Suite ==="

# Parse arguments
RUN_UNIT=true
RUN_INTEGRATION=false
RUN_E2E=false
RUN_PERF=false
COVERAGE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit) RUN_UNIT=true; shift ;;
        --integration) RUN_INTEGRATION=true; shift ;;
        --e2e) RUN_E2E=true; shift ;;
        --perf) RUN_PERF=true; shift ;;
        --all) RUN_UNIT=true; RUN_INTEGRATION=true; RUN_E2E=true; shift ;;
        --no-cov) COVERAGE=false; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Build pytest command
CMD="pytest"

if $COVERAGE; then
    CMD="$CMD --cov=src/smart_search --cov-report=html --cov-report=term-missing"
fi

# Run unit tests
if $RUN_UNIT; then
    echo ""
    echo ">>> Running Unit Tests..."
    $CMD tests/unit -v --tb=short
fi

# Run integration tests
if $RUN_INTEGRATION; then
    echo ""
    echo ">>> Running Integration Tests..."
    $CMD tests/integration -v --tb=short -m integration
fi

# Run E2E tests
if $RUN_E2E; then
    echo ""
    echo ">>> Running E2E Tests..."
    $CMD tests/e2e -v --tb=short -m e2e
fi

# Run performance tests
if $RUN_PERF; then
    echo ""
    echo ">>> Running Performance Tests..."
    $CMD tests/performance -v --tb=short -m performance --benchmark-only
fi

echo ""
echo "=== Test Suite Complete ==="
```

---

## 8. CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install ruff mypy
      - name: Lint with ruff
        run: ruff check src/ tests/
      - name: Type check with mypy
        run: mypy src/

  unit-tests:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run unit tests
        run: pytest tests/unit -v --cov=src/smart_search --cov-fail-under=85
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      meilisearch:
        image: getmeili/meilisearch:v1.6
        ports:
          - 7700:7700
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run integration tests
        run: pytest tests/integration -v -m integration
        env:
          MEILISEARCH_URL: http://localhost:7700

  e2e-tests:
    runs-on: ubuntu-latest
    needs: integration-tests
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run E2E tests
        run: pytest tests/e2e -v -m e2e
```

---

## 9. Exit Criteria Summary

| Phase | Coverage Target | Key Tests | Pass Criteria |
|-------|-----------------|-----------|---------------|
| 1 | 85% | Config, Health | All pass, Meilisearch connects |
| 2 | 90% | Parser, Chunker, Extractors | Python/JS parsing works |
| 3 | 95% | Graph operations, Algorithms | Cycles detected, <10ms traversal |
| 4 | 90% | Embedding, Cache | 768D vectors, cache works |
| 5 | 90% | Search, Indexing | <50ms search, hybrid works |
| 6 | 90% | Change detection, Incremental | State persists, atomic updates |
| 7 | 85% | GraphRAG, Retriever | Context expansion works |
| 8 | 85% | All API endpoints | E2E workflows pass |
| **Overall** | **85%** | **All test categories** | **CI pipeline green** |

---

## 10. Thứ tự Triển khai (với Testing)

```
Phase 1 (Foundation) ─────────────────────┐
    │                                     │
    └── Unit Tests Phase 1 ───────────────┤
                                          │
Phase 2 (AST Parsing) ────────────────────┤
    │                                     │
    └── Unit Tests Phase 2 ───────────────┤
        │                                 │
        ├── Phase 3 (Graph Engine) ───────┤
        │       │                         │
        │       └── Unit Tests Phase 3 ───┤
        │                                 │
        └── Phase 4 (Embedding) ──────────┤  [parallel]
                │                         │
                └── Unit Tests Phase 4 ───┤
                                          │
            Phase 5 (Search) ─────────────┤
                │                         │
                └── Unit Tests Phase 5 ───┤
                    │                     │
                    └── Integration Tests ┤
                                          │
            Phase 6 (Incremental) ────────┤
                │                         │
                └── Unit Tests Phase 6 ───┤
                                          │
            Phase 7 (GraphRAG) ───────────┤
                │                         │
                └── Unit Tests Phase 7 ───┤
                                          │
            Phase 8 (API Layer) ──────────┤
                │                         │
                ├── Unit Tests Phase 8 ───┤
                │                         │
                └── E2E Tests ────────────┤
                                          │
            Performance Tests ────────────┘
```

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Meilisearch 100MB payload limit | Batch documents, max 1000/batch |
| 65,535 attribute position limit | AST-based chunking, không full files |
| Graph-Search data inconsistency | Transactional updates, atomic operations |
| Memory với large graphs | RustworkX (không dùng NetworkX) |
| GraphRAG over-engineering | Start với core relations: calls, imports |
| **Flaky tests** | **Use deterministic fixtures, mock external services** |
| **Slow test suite** | **Parallel execution, mock heavy operations** |
| **Test data maintenance** | **Generate fixtures programmatically** |

---

## 12. Scalability Estimates (1M LOC)

| Component | Estimate |
|-----------|----------|
| Graph Nodes | ~200,000 |
| Graph Edges | ~1,000,000 |
| Graph Memory (RustworkX) | <50MB |
| Search Index RAM | 1-2GB |
| Vector Storage | ~20MB (binary quantized) |
