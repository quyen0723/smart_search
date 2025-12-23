"""API Orchestrator.

Central orchestrator for Smart Search API services.
Manages service initialization, lifecycle, and dependency injection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from contextlib import asynccontextmanager


@dataclass
class ServiceConfig:
    """Configuration for API services."""

    # Paths
    project_root: Path = field(default_factory=Path.cwd)
    data_dir: Path = field(default_factory=lambda: Path.cwd() / ".smart_search")

    # Meilisearch
    meilisearch_url: str = "http://localhost:7700"
    meilisearch_api_key: str | None = None

    # Embeddings
    embedding_model: str = "jinaai/jina-embeddings-v3"
    embedding_dimensions: int = 1024

    # LLM
    llm_provider: str = "openai"  # openai, anthropic, mock
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str | None = None

    # Graph
    enable_communities: bool = True
    community_algorithm: str = "louvain"

    # Cache
    enable_cache: bool = True
    cache_ttl: int = 3600

    @classmethod
    def from_env(cls) -> "ServiceConfig":
        """Create config from environment variables."""
        import os

        return cls(
            project_root=Path(os.getenv("PROJECT_ROOT", Path.cwd())),
            data_dir=Path(os.getenv("DATA_DIR", Path.cwd() / ".smart_search")),
            meilisearch_url=os.getenv("MEILISEARCH_URL", "http://localhost:7700"),
            meilisearch_api_key=os.getenv("MEILISEARCH_API_KEY"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v3"),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            enable_cache=os.getenv("ENABLE_CACHE", "true").lower() == "true",
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
        )


@dataclass
class ServiceRegistry:
    """Registry of initialized services."""

    graph: Any = None
    searcher: Any = None
    indexer: Any = None
    graphrag: Any = None
    embedder: Any = None

    def is_initialized(self) -> bool:
        """Check if core services are initialized."""
        return all([
            self.graph is not None,
            self.searcher is not None,
        ])


class APIOrchestrator:
    """Central orchestrator for Smart Search API.

    Manages service initialization, lifecycle, and coordinates
    between different API endpoints.
    """

    def __init__(self, config: ServiceConfig | None = None):
        """Initialize orchestrator.

        Args:
            config: Service configuration. Uses env vars if not provided.
        """
        self.config = config or ServiceConfig.from_env()
        self.services = ServiceRegistry()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all services.

        Called during application startup.
        """
        if self._initialized:
            return

        # Ensure data directory exists
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize services in order
        await self._init_embedder()
        await self._init_graph()
        await self._init_searcher()
        await self._init_indexer()
        await self._init_graphrag()

        # Inject dependencies into endpoint modules
        self._inject_dependencies()

        self._initialized = True

    async def _init_embedder(self) -> None:
        """Initialize embedding service."""
        try:
            from smart_search.embedding import MockEmbedder

            # Use mock embedder for now (real embedder requires model download)
            self.services.embedder = MockEmbedder()
        except Exception:
            self.services.embedder = None

    async def _init_graph(self) -> None:
        """Initialize code graph."""
        try:
            from smart_search.graph import CodeGraph

            self.services.graph = CodeGraph()

            # Load persisted graph if exists
            graph_path = self.config.data_dir / "graph.json"
            if graph_path.exists():
                from smart_search.graph import GraphPersistence
                persistence = GraphPersistence(graph_path)
                persistence.load(self.services.graph)
        except Exception:
            self.services.graph = None

    async def _init_searcher(self) -> None:
        """Initialize search service."""
        try:
            from smart_search.search import HybridSearcher, MeilisearchClient

            # Create search client
            client = MeilisearchClient(
                url=self.config.meilisearch_url,
                api_key=self.config.meilisearch_api_key,
            )
            self.services.searcher = HybridSearcher(
                client=client,
                embedder=self.services.embedder,
            )
        except Exception:
            self.services.searcher = None

    async def _init_indexer(self) -> None:
        """Initialize indexer service."""
        try:
            from smart_search.graph import GraphBuilder
            from smart_search.parsing.tree_sitter_parser import TreeSitterParser
            from smart_search.parsing.extractors.python import PythonExtractor
            from smart_search.parsing.extractors.php import PHPExtractor
            from smart_search.parsing.models import Language

            # Create a simple indexer that wraps parsing and graph building
            class SimpleIndexer:
                def __init__(self, graph, data_dir):
                    self.graph = graph
                    self.data_dir = data_dir
                    self.parser = TreeSitterParser()
                    self.parser.register_extractor(Language.PYTHON, PythonExtractor())
                    self.parser.register_extractor(Language.PHP, PHPExtractor())
                    self.builder = GraphBuilder()
                    self.indexed_files = {}

                async def index_file(self, file_path, force=False):
                    """Index a single file."""
                    from pathlib import Path
                    path = Path(file_path)
                    if not path.exists() or not path.is_file():
                        return

                    # Detect language from extension
                    ext = path.suffix.lower()
                    lang_map = {".py": "python", ".php": "php"}
                    language = lang_map.get(ext)
                    if not language:
                        return  # Skip unsupported files

                    # Parse file
                    result = self.parser.parse_file(path)
                    if result and result.units:
                        # Add units to graph
                        for unit in result.units:
                            self.builder.add_unit(unit)

                        self.indexed_files[str(path)] = {
                            "units": len(result.units),
                            "language": language,
                        }

                async def remove_path(self, path: str) -> int:
                    removed = 0
                    to_remove = [k for k in self.indexed_files if k.startswith(path)]
                    for k in to_remove:
                        del self.indexed_files[k]
                        removed += 1
                    return removed

                async def get_stats(self):
                    # Count by language
                    lang_counts = {}
                    ext_counts = {}
                    for path, info in self.indexed_files.items():
                        lang = info.get("language", "unknown")
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
                        ext = Path(path).suffix.lower()
                        ext_counts[ext] = ext_counts.get(ext, 0) + 1

                    return {
                        "total_files": len(self.indexed_files),
                        "total_code_units": sum(f.get("units", 0) for f in self.indexed_files.values()),
                        "languages": lang_counts,
                        "files_by_type": ext_counts,
                        "index_size_bytes": 0,
                    }

                async def list_files(self, pattern=None, language=None, limit=100, offset=0):
                    files = list(self.indexed_files.items())[offset:offset + limit]
                    return [
                        {
                            "file_path": k,
                            "language": v.get("language", "unknown"),
                            "size_bytes": 0,
                            "last_modified": "",
                            "code_units": v.get("units", 0),
                        }
                        for k, v in files
                    ]

                async def get_file_info(self, file_path: str):
                    info = self.indexed_files.get(file_path)
                    if info:
                        return {
                            "file_path": file_path,
                            "language": info.get("language", "unknown"),
                            "size_bytes": 0,
                            "last_modified": "",
                            "code_units": info.get("units", 0),
                        }
                    return None

                async def refresh(self):
                    return {"updated": 0}

                async def clear(self):
                    self.indexed_files.clear()

                async def health_check(self):
                    return {"healthy": True, "services": {}}

                async def find_references(self, filename: str, limit: int = 50) -> list[dict]:
                    """Find files that reference/use the given filename.

                    Searches for:
                    - PHP: require, require_once, include, include_once statements
                    - Python: import, from...import statements
                    - Function/class usage from the file
                    """
                    import re
                    from pathlib import Path

                    results = []
                    filename_lower = filename.lower()

                    # Extract just the filename without path
                    base_name = Path(filename).name
                    base_name_no_ext = Path(filename).stem

                    # Patterns to search for in PHP files
                    php_patterns = [
                        rf"require\s*\(?['\"].*{re.escape(base_name)}['\"]",
                        rf"require_once\s*\(?['\"].*{re.escape(base_name)}['\"]",
                        rf"include\s*\(?['\"].*{re.escape(base_name)}['\"]",
                        rf"include_once\s*\(?['\"].*{re.escape(base_name)}['\"]",
                        rf"['\"].*{re.escape(base_name)}['\"]",  # String containing filename
                    ]

                    # Python patterns
                    python_patterns = [
                        rf"from\s+{re.escape(base_name_no_ext)}\s+import",
                        rf"import\s+{re.escape(base_name_no_ext)}",
                    ]

                    # Get functions/classes from the target file
                    target_units = []
                    for node in self.builder.graph.get_all_nodes():
                        data = node.data
                        file_path = str(data.file_path) if data.file_path else ""
                        if base_name in file_path or filename_lower in file_path.lower():
                            target_units.append(data.name)

                    # Search through indexed files
                    for file_path, info in self.indexed_files.items():
                        # Skip the target file itself
                        if base_name in file_path:
                            continue

                        try:
                            path = Path(file_path)
                            if not path.exists():
                                continue

                            content = path.read_text(encoding='utf-8', errors='ignore')
                            content_lower = content.lower()

                            matches = []
                            match_score = 0

                            # Check for require/include patterns
                            language = info.get("language", "unknown")
                            patterns = php_patterns if language == "php" else python_patterns

                            for pattern in patterns:
                                for match in re.finditer(pattern, content, re.IGNORECASE):
                                    line_num = content[:match.start()].count('\n') + 1
                                    line_text = content.split('\n')[line_num - 1].strip()
                                    if line_text and line_text not in matches:
                                        matches.append({
                                            "line": line_num,
                                            "text": line_text[:200],
                                            "type": "include/require"
                                        })
                                        match_score += 10

                            # Check for function/class usage from target file
                            for unit_name in target_units:
                                if unit_name and len(unit_name) > 2:  # Skip very short names
                                    # Look for function calls or class usage
                                    func_pattern = rf'\b{re.escape(unit_name)}\s*\('
                                    for match in re.finditer(func_pattern, content):
                                        line_num = content[:match.start()].count('\n') + 1
                                        line_text = content.split('\n')[line_num - 1].strip()
                                        if line_text and not any(m['text'] == line_text for m in matches):
                                            matches.append({
                                                "line": line_num,
                                                "text": line_text[:200],
                                                "type": f"uses {unit_name}"
                                            })
                                            match_score += 5

                            if matches:
                                results.append({
                                    "file_path": file_path,
                                    "language": language,
                                    "matches": matches[:10],  # Limit matches per file
                                    "match_count": len(matches),
                                    "score": match_score
                                })

                        except Exception:
                            continue

                    # Sort by score
                    results.sort(key=lambda x: x["score"], reverse=True)
                    return results[:limit]

                async def search(self, query: str, limit: int = 20, language: str = None, code_type: str = None):
                    """Fuzzy text search in indexed code units."""
                    import re
                    from difflib import SequenceMatcher

                    def tokenize(text: str) -> list[str]:
                        """Split camelCase, PascalCase, snake_case into words."""
                        # Split by non-alphanumeric
                        text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
                        # Split camelCase: 'getUserData' -> 'get User Data'
                        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
                        # Split consecutive uppercase: 'XMLParser' -> 'XML Parser'
                        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
                        return [w.lower() for w in text.split() if w]

                    def fuzzy_match(s1: str, s2: str) -> float:
                        """Calculate fuzzy similarity (0-1)."""
                        return SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

                    def word_in_tokens(word: str, tokens: list[str], threshold: float = 0.8) -> float:
                        """Check if word matches any token (exact or fuzzy)."""
                        word_lower = word.lower()
                        best_score = 0
                        for token in tokens:
                            # Exact substring
                            if word_lower in token or token in word_lower:
                                best_score = max(best_score, 1.0)
                            # Fuzzy match
                            else:
                                ratio = fuzzy_match(word_lower, token)
                                if ratio >= threshold:
                                    best_score = max(best_score, ratio)
                        return best_score

                    results = []
                    query_lower = query.lower()
                    query_words = tokenize(query)

                    for node in self.builder.graph.get_all_nodes():
                        data = node.data
                        score = 0

                        name = data.name or ""
                        qualified = getattr(data, 'qualified_name', '') or ""
                        file_path = str(data.file_path) if data.file_path else ""

                        name_lower = name.lower()
                        qualified_lower = qualified.lower()
                        file_lower = file_path.lower()

                        # Tokenize for fuzzy matching
                        name_tokens = tokenize(name)
                        qualified_tokens = tokenize(qualified)
                        file_tokens = tokenize(file_path.split('/')[-1] if file_path else "")
                        all_tokens = name_tokens + qualified_tokens + file_tokens

                        # Exact substring match in name (highest priority)
                        if query_lower in name_lower:
                            score += 20

                        # Exact substring in qualified name
                        if query_lower in qualified_lower:
                            score += 10

                        # Exact substring in file path
                        if query_lower in file_lower:
                            score += 5

                        # Word-based fuzzy matching
                        for word in query_words:
                            # Check in name tokens
                            name_match = word_in_tokens(word, name_tokens)
                            if name_match > 0:
                                score += 8 * name_match

                            # Check in qualified tokens
                            qual_match = word_in_tokens(word, qualified_tokens)
                            if qual_match > 0:
                                score += 4 * qual_match

                            # Check in file tokens
                            file_match = word_in_tokens(word, file_tokens)
                            if file_match > 0:
                                score += 2 * file_match

                        if score > 0:
                            # Apply filters
                            if language and hasattr(data, 'language'):
                                if str(data.language).lower() != language.lower():
                                    continue
                            if code_type and hasattr(data, 'node_type'):
                                if str(data.node_type).lower() != code_type.lower():
                                    continue

                            results.append({
                                "id": data.id,
                                "name": data.name,
                                "qualified_name": getattr(data, 'qualified_name', data.name),
                                "code_type": str(getattr(data, 'node_type', 'unknown')),
                                "file_path": file_path,
                                "line_start": data.line_start or 0,
                                "line_end": data.line_end or 0,
                                "score": round(score, 2),
                            })

                    # Sort by score and limit
                    results.sort(key=lambda x: x["score"], reverse=True)
                    return results[:limit]

            self.services.indexer = SimpleIndexer(self.services.graph, self.config.data_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.services.indexer = None

    async def _init_graphrag(self) -> None:
        """Initialize GraphRAG service."""
        try:
            from smart_search.rag import GraphRAG, GraphRAGConfig

            config = GraphRAGConfig(
                llm_provider=self.config.llm_provider,
                llm_model=self.config.llm_model,
            )

            self.services.graphrag = GraphRAG(
                config=config,
                graph=self.services.graph,
                retriever=None,  # Will use default
            )
        except Exception:
            self.services.graphrag = None

    def _inject_dependencies(self) -> None:
        """Inject service dependencies into endpoint modules."""
        # Import endpoint modules
        from smart_search.api.endpoints import search, navigate, analyze, graph, index

        # Inject into search endpoints
        if self.services.searcher:
            search.set_searcher(self.services.searcher)
        if self.services.graphrag:
            search.set_graphrag(self.services.graphrag)

        # Inject into navigate endpoints
        if self.services.graph:
            navigate.set_graph(self.services.graph)
        if self.services.searcher:
            navigate.set_searcher(self.services.searcher)

        # Inject into analyze endpoints
        if self.services.graph:
            analyze.set_graph(self.services.graph)
        if self.services.graphrag:
            analyze.set_graphrag(self.services.graphrag)

        # Inject into graph endpoints
        if self.services.graph:
            graph.set_graph(self.services.graph)

        # Inject into index endpoints
        if self.services.indexer:
            index.set_indexer(self.services.indexer)

    async def shutdown(self) -> None:
        """Shutdown all services.

        Called during application shutdown.
        """
        # Persist graph
        if self.services.graph:
            graph_path = self.config.data_dir / "graph.json"
            self.services.graph.save(graph_path)

        # Close connections
        if self.services.searcher:
            await self.services.searcher.close()

        self._initialized = False

    def create_app(self) -> FastAPI:
        """Create FastAPI application with all routes.

        Returns:
            Configured FastAPI application.
        """
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.initialize()
            yield
            # Shutdown
            await self.shutdown()

        app = FastAPI(
            title="Smart Search API",
            description="Intelligent Source Code Navigation System",
            version="2.1.0",
            lifespan=lifespan,
        )

        # Include routers
        from smart_search.api.endpoints import search, navigate, analyze, graph, index

        app.include_router(search.router, prefix="/api/v1")
        app.include_router(navigate.router, prefix="/api/v1")
        app.include_router(analyze.router, prefix="/api/v1")
        app.include_router(graph.router, prefix="/api/v1")
        app.include_router(index.router, prefix="/api/v1")

        # Root endpoint
        @app.get("/")
        async def root():
            return {
                "name": "Smart Search API",
                "version": "2.1.0",
                "status": "running",
            }

        # Health check
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "services": {
                    "graph": self.services.graph is not None,
                    "searcher": self.services.searcher is not None,
                    "indexer": self.services.indexer is not None,
                    "graphrag": self.services.graphrag is not None,
                },
            }

        return app


# Convenience function for creating app
def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """Create Smart Search API application.

    Args:
        config: Optional service configuration.

    Returns:
        Configured FastAPI application.
    """
    orchestrator = APIOrchestrator(config)
    return orchestrator.create_app()


# Mock services for testing
class MockGraph:
    """Mock graph for testing."""

    def __init__(self):
        self._nodes = {}
        self._edges = []

    def get_node(self, node_id: str):
        return self._nodes.get(node_id)

    def get_all_nodes(self):
        return list(self._nodes.values())

    def get_nodes_in_file(self, file_path: str):
        return [n for n in self._nodes.values() if str(getattr(n, "file_path", "")) == file_path]

    def get_callers(self, node_id: str, depth: int = 1):
        return []

    def get_callees(self, node_id: str, depth: int = 1):
        return []

    def get_stats(self):
        return {
            "total_nodes": len(self._nodes),
            "total_edges": len(self._edges),
            "node_types": {},
            "edge_types": {},
            "files_count": 0,
            "communities_count": 0,
            "avg_degree": 0.0,
        }

    def get_all_communities(self):
        return []

    def get_community_members(self, community_id: str):
        return []

    def get_all_files(self):
        return list(set(str(getattr(n, "file_path", "")) for n in self._nodes.values()))

    def find_node_at_position(self, file_path: str, line: int, column: int):
        return None

    def get_definition(self, node_id: str):
        return self._nodes.get(node_id)

    def get_references(self, node_id: str, limit: int = 50):
        return []

    def get_children(self, node_id: str):
        return []

    def load(self, path):
        pass

    def save(self, path):
        pass


class MockSearcher:
    """Mock searcher for testing."""

    async def search(self, query):
        from smart_search.search.schemas import SearchResult, SearchType
        return SearchResult(
            query=query.query,
            hits=[],
            total=0,
            search_type=query.search_type if hasattr(query, 'search_type') else SearchType.HYBRID,
        )

    async def initialize(self):
        pass

    async def close(self):
        pass


class MockIndexer:
    """Mock indexer for testing."""

    async def index_file(self, file_path, force=False):
        pass

    async def remove_path(self, path: str) -> int:
        return 0

    async def get_stats(self):
        return {
            "total_files": 0,
            "total_code_units": 0,
            "languages": {},
            "files_by_type": {},
            "index_size_bytes": 0,
        }

    async def list_files(self, pattern=None, language=None, limit=100, offset=0):
        return []

    async def get_file_info(self, file_path: str):
        return None

    async def refresh(self):
        return {"updated": 0}

    async def clear(self):
        pass

    async def health_check(self):
        return {"healthy": True, "services": {}}


class MockGraphRAG:
    """Mock GraphRAG for testing."""

    async def query(self, query: str, mode=None, filters=None):
        from smart_search.rag import SearchMode
        from dataclasses import dataclass, field as dc_field

        @dataclass
        class MockResult:
            query: str
            mode: SearchMode = SearchMode.HYBRID
            response: str = "Mock response"
            contexts: list = dc_field(default_factory=list)
            total_time_ms: float = 0.0

            @dataclass
            class Generation:
                citations: list = dc_field(default_factory=list)

            generation: Generation = dc_field(default_factory=Generation)

        return MockResult(query=query, mode=mode or SearchMode.HYBRID)

    async def find_similar(self, code_content: str, limit: int = 5):
        return []

    async def explain_code(self, code_id: str, question: str | None = None):
        from dataclasses import dataclass

        @dataclass
        class MockExplainResult:
            response: str = "Mock explanation"

        return MockExplainResult()


def create_test_app() -> FastAPI:
    """Create test application with mock services.

    Returns:
        FastAPI application with mock services injected.
    """
    app = FastAPI(
        title="Smart Search API (Test)",
        version="2.1.0-test",
    )

    # Create mock services
    mock_graph = MockGraph()
    mock_searcher = MockSearcher()
    mock_indexer = MockIndexer()
    mock_graphrag = MockGraphRAG()

    # Import and configure endpoints
    from smart_search.api.endpoints import search, navigate, analyze, graph, index

    search.set_searcher(mock_searcher)
    search.set_graphrag(mock_graphrag)
    navigate.set_graph(mock_graph)
    navigate.set_searcher(mock_searcher)
    analyze.set_graph(mock_graph)
    analyze.set_graphrag(mock_graphrag)
    graph.set_graph(mock_graph)
    index.set_indexer(mock_indexer)

    # Include routers
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(navigate.router, prefix="/api/v1")
    app.include_router(analyze.router, prefix="/api/v1")
    app.include_router(graph.router, prefix="/api/v1")
    app.include_router(index.router, prefix="/api/v1")

    @app.get("/")
    async def root():
        return {"name": "Smart Search API (Test)", "version": "2.1.0-test"}

    @app.get("/health")
    async def health():
        return {"status": "healthy", "mode": "test"}

    return app
