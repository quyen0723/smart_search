"""API Orchestrator.

Central orchestrator for Smart Search API services.
Manages service initialization, lifecycle, and dependency injection.
"""

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from contextlib import asynccontextmanager

from smart_search.config import get_settings, FeatureFlags
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


def get_feature_flags() -> FeatureFlags:
    """Get current feature flags."""
    return get_settings().feature_flags


class SearchServiceAdapter:
    """Adapter that routes search to HybridSearcher or SimpleIndexer based on FF.

    This adapter implements the same interface as HybridSearcher but can fallback
    to SimpleIndexer.search() when FF_USE_HYBRID_SEARCHER is disabled.
    """

    def __init__(
        self,
        hybrid_searcher: Any = None,
        simple_indexer: Any = None,
    ):
        """Initialize search adapter.

        Args:
            hybrid_searcher: HybridSearcher instance (for Meilisearch).
            simple_indexer: SimpleIndexer instance (for legacy brute-force).
        """
        self.hybrid_searcher = hybrid_searcher
        self.simple_indexer = simple_indexer

    async def search(self, query: Any) -> Any:
        """Route search to appropriate backend based on feature flags.

        Args:
            query: SearchQuery object.

        Returns:
            SearchResult from either HybridSearcher or SimpleIndexer.
        """
        from smart_search.search.schemas import SearchHit, SearchResult, SearchType

        ff = get_feature_flags()

        # Use HybridSearcher if enabled and available
        if ff.use_hybrid_searcher and self.hybrid_searcher is not None:
            logger.debug(
                "Routing search to HybridSearcher",
                query=query.query if hasattr(query, 'query') else str(query),
            )
            return await self.hybrid_searcher.search(query)

        # Fallback to SimpleIndexer
        if self.simple_indexer is not None:
            logger.debug(
                "Routing search to SimpleIndexer (legacy)",
                query=query.query if hasattr(query, 'query') else str(query),
            )
            # SimpleIndexer.search() has different signature
            query_str = query.query if hasattr(query, 'query') else str(query)
            limit = query.limit if hasattr(query, 'limit') else 20
            language = query.filters.languages[0] if (
                hasattr(query, 'filters') and query.filters and query.filters.languages
            ) else None
            code_type = query.filters.code_types[0] if (
                hasattr(query, 'filters') and query.filters and query.filters.code_types
            ) else None

            results = await self.simple_indexer.search(
                query=query_str,
                limit=limit,
                language=language,
                code_type=code_type,
            )

            # Convert SimpleIndexer results to SearchResult format
            hits = [
                SearchHit(
                    id=r.get("id", ""),
                    name=r.get("name", ""),
                    qualified_name=r.get("qualified_name", r.get("name", "")),
                    code_type=r.get("code_type", "unknown"),
                    file_path=r.get("file_path", ""),
                    line_start=r.get("line_start", 0),
                    line_end=r.get("line_end", 0),
                    content="",  # SimpleIndexer doesn't return content
                    language=r.get("language", "unknown"),
                    score=r.get("score", 0.0),
                    highlights={},
                    metadata={},
                )
                for r in results
            ]

            return SearchResult(
                hits=hits,
                total=len(hits),
                query=query_str,
                search_type=SearchType.KEYWORD,
                processing_time_ms=0,
                offset=query.offset if hasattr(query, 'offset') else 0,
                limit=limit,
            )

        # No search backend available
        logger.error("No search backend available")
        return SearchResult(
            hits=[],
            total=0,
            query=query.query if hasattr(query, 'query') else str(query),
            search_type=SearchType.KEYWORD,
            processing_time_ms=0,
            offset=0,
            limit=20,
        )

    async def close(self) -> None:
        """Close resources."""
        if self.hybrid_searcher is not None:
            try:
                await self.hybrid_searcher.close()
            except Exception:
                pass


# File content cache with mtime validation
# Cache max 500 files, invalidate if file modified
_FILE_CACHE_MAX_SIZE = 500


def _get_file_mtime(file_path: str) -> float:
    """Get file modification time."""
    try:
        return os.path.getmtime(file_path)
    except OSError:
        return 0.0


@lru_cache(maxsize=_FILE_CACHE_MAX_SIZE)
def _read_file_cached(file_path: str, mtime: float) -> str:
    """Read file content with LRU cache.

    The mtime parameter ensures cache invalidation when file changes.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except (OSError, IOError):
        return ""


def read_file_with_cache(file_path: str) -> str:
    """Read file content using LRU cache with mtime validation.

    Returns empty string if file doesn't exist or can't be read.
    """
    mtime = _get_file_mtime(file_path)
    if mtime == 0.0:
        return ""
    return _read_file_cached(file_path, mtime)


def get_file_cache_info() -> dict:
    """Get file cache statistics."""
    info = _read_file_cached.cache_info()
    return {
        "hits": info.hits,
        "misses": info.misses,
        "maxsize": info.maxsize,
        "currsize": info.currsize,
        "hit_ratio": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
    }


def clear_file_cache() -> None:
    """Clear the file cache."""
    _read_file_cached.cache_clear()


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
            from smart_search.graph import CodeGraph, GraphPersistence

            # Load persisted graph if exists
            graph_path = self.config.data_dir / "graph.json"
            if graph_path.exists():
                persistence = GraphPersistence(self.config.data_dir)
                self.services.graph = persistence.load_json(graph_path)
                logger.info("Loaded existing graph", path=str(graph_path))
            else:
                self.services.graph = CodeGraph()
                logger.info("Created new empty graph")
        except Exception as e:
            logger.warning(f"Failed to initialize graph: {e}")
            from smart_search.graph import CodeGraph
            self.services.graph = CodeGraph()  # Fallback to empty graph

    async def _init_searcher(self) -> None:
        """Initialize search service.

        When FF_USE_HYBRID_SEARCHER=true, uses HybridSearcher with Meilisearch.
        Otherwise, search will fallback to SimpleIndexer.search().
        """
        ff = get_feature_flags()

        if not ff.use_hybrid_searcher:
            logger.info(
                "HybridSearcher disabled by feature flag",
                ff_use_hybrid_searcher=ff.use_hybrid_searcher,
            )
            self.services.searcher = None
            return

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
            logger.info(
                "HybridSearcher initialized (Full Mode)",
                meilisearch_url=self.config.meilisearch_url,
                ff_use_hybrid_searcher=True,
            )
        except Exception as e:
            logger.warning(
                f"Failed to initialize HybridSearcher: {e}",
                fallback="SimpleIndexer",
            )
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
                # Pre-compiled regex patterns for tokenization (class-level)
                _RE_NON_ALPHANUM = re.compile(r'[^a-zA-Z0-9]')
                _RE_CAMEL_SPLIT = re.compile(r'([a-z])([A-Z])')
                _RE_UPPER_SPLIT = re.compile(r'([A-Z]+)([A-Z][a-z])')

                def __init__(self, graph, data_dir, meilisearch_client=None, embedder=None, reference_index=None):
                    self.graph = graph
                    self.data_dir = data_dir
                    self.meilisearch_client = meilisearch_client  # For dual-write
                    self.embedder = embedder  # For generating embeddings
                    self.reference_index = reference_index  # For O(1) find_references
                    self.parser = TreeSitterParser()
                    self.parser.register_extractor(Language.PYTHON, PythonExtractor())
                    self.parser.register_extractor(Language.PHP, PHPExtractor())
                    self.builder = GraphBuilder()
                    self.indexed_files = {}
                    # Pre-compiled patterns cache (instance-level for dynamic patterns)
                    self._pattern_cache = {}

                def _get_compiled_pattern(self, pattern: str, flags: int = 0) -> re.Pattern:
                    """Get or create a compiled regex pattern from cache."""
                    key = (pattern, flags)
                    if key not in self._pattern_cache:
                        self._pattern_cache[key] = re.compile(pattern, flags)
                    return self._pattern_cache[key]

                def _tokenize(self, text: str) -> list[str]:
                    """Split camelCase, PascalCase, snake_case into words (using pre-compiled patterns)."""
                    # Use class-level pre-compiled patterns
                    text = self._RE_NON_ALPHANUM.sub(' ', text)
                    text = self._RE_CAMEL_SPLIT.sub(r'\1 \2', text)
                    text = self._RE_UPPER_SPLIT.sub(r'\1 \2', text)
                    return [w.lower() for w in text.split() if w]

                async def index_file(self, file_path, force=False):
                    """Index a single file.

                    Performs:
                    1. Graph write (always)
                    2. Reference Index write (when FF enabled)
                    3. Meilisearch write (when FF enabled)
                    """
                    from pathlib import Path
                    from smart_search.search.schemas import IndexDocument

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
                        # 1. Add units to graph (always)
                        for unit in result.units:
                            self.builder.add_unit(unit)

                        self.indexed_files[str(path)] = {
                            "units": len(result.units),
                            "language": language,
                        }

                        # 2. Index references (when FF enabled)
                        ff = get_feature_flags()
                        if ff.use_reference_index and self.reference_index is not None:
                            try:
                                from smart_search.search.reference_index import (
                                    SymbolDefinition, ReferenceLocation, ReferenceType
                                )
                                # Remove old references from this file first
                                self.reference_index.remove_file(str(path))

                                # Add definitions
                                for unit in result.units:
                                    defn = SymbolDefinition(
                                        name=unit.name,
                                        qualified_name=getattr(unit, 'qualified_name', unit.name),
                                        symbol_type=str(unit.type.value) if hasattr(unit, 'type') else 'unknown',
                                        file_path=str(path),
                                        line_start=unit.start_line or 0,
                                        line_end=unit.end_line or 0,
                                    )
                                    self.reference_index.add_definition(defn)

                                    # Add references from callees
                                    callees = getattr(unit, 'callees', []) or []
                                    for callee in callees:
                                        if callee and len(callee) > 2:
                                            ref = ReferenceLocation(
                                                file_path=str(path),
                                                line=unit.start_line or 0,
                                                ref_type=ReferenceType.CALL,
                                                context=f"Called from {unit.name}",
                                            )
                                            self.reference_index.add_reference(callee, ref)

                                logger.debug(
                                    "Indexed references",
                                    file=str(path),
                                    units=len(result.units),
                                )
                            except Exception as e:
                                logger.warning(f"Reference indexing failed: {e}", file=str(path))

                        # 3. Dual-write to Meilisearch (when FF enabled)
                        if ff.use_meilisearch_search and self.meilisearch_client is not None:
                            try:
                                # Convert parsed units to IndexDocuments
                                documents = []
                                for unit in result.units:
                                    # Read file content for the unit's lines
                                    content = ""
                                    try:
                                        file_content = path.read_text(encoding='utf-8', errors='ignore')
                                        lines = file_content.split('\n')
                                        start = max(0, (unit.start_line or 1) - 1)
                                        end = min(len(lines), unit.end_line or len(lines))
                                        content = '\n'.join(lines[start:end])
                                    except Exception:
                                        pass

                                    doc = IndexDocument(
                                        id=unit.id,
                                        name=unit.name,
                                        qualified_name=getattr(unit, 'qualified_name', unit.name),
                                        code_type=str(unit.type.value) if hasattr(unit, 'type') else 'unknown',
                                        file_path=str(path),
                                        line_start=unit.start_line or 0,
                                        line_end=unit.end_line or 0,
                                        content=content,
                                        language=language,
                                        signature=getattr(unit, 'signature', ''),
                                        docstring=getattr(unit, 'docstring', ''),
                                        embedding=None,  # Embeddings added separately if needed
                                    )
                                    documents.append(doc)

                                if documents:
                                    await self.meilisearch_client.add_documents(documents)
                                    logger.debug(
                                        "Dual-write to Meilisearch",
                                        file=str(path),
                                        documents=len(documents),
                                    )
                            except Exception as e:
                                # Log but don't fail - graph write succeeded
                                logger.warning(
                                    f"Meilisearch dual-write failed: {e}",
                                    file=str(path),
                                )

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

                    When FF_USE_REFERENCE_INDEX is enabled, uses O(1) index lookup.
                    Otherwise, falls back to file scanning.

                    Searches for:
                    - PHP: require, require_once, include, include_once statements
                    - Python: import, from...import statements
                    - Function/class usage from the file
                    """
                    from pathlib import Path

                    # Fast path: use reference index if available
                    ff = get_feature_flags()
                    if ff.use_reference_index and self.reference_index is not None:
                        refs = self.reference_index.get_references(
                            filename, include_definition=False, limit=limit
                        )
                        results = []
                        for ref in refs:
                            results.append({
                                "file_path": ref.file_path,
                                "language": "unknown",  # Not stored in ref index
                                "matches": [{
                                    "line": ref.line,
                                    "text": ref.context[:200],
                                    "type": ref.ref_type.value,
                                }],
                                "match_count": 1,
                                "score": 10,
                            })
                        logger.debug(
                            "find_references via ReferenceIndex",
                            filename=filename,
                            results=len(results),
                        )
                        return results

                    # Slow path: file scanning
                    from smart_search.utils.async_io import async_read_files_parallel

                    results = []
                    filename_lower = filename.lower()

                    # Extract just the filename without path
                    base_name = Path(filename).name
                    base_name_no_ext = Path(filename).stem
                    escaped_base = re.escape(base_name)
                    escaped_stem = re.escape(base_name_no_ext)

                    # Pre-compile patterns for PHP files (using cache)
                    php_patterns = [
                        self._get_compiled_pattern(rf"require\s*\(?['\"].*{escaped_base}['\"]", re.IGNORECASE),
                        self._get_compiled_pattern(rf"require_once\s*\(?['\"].*{escaped_base}['\"]", re.IGNORECASE),
                        self._get_compiled_pattern(rf"include\s*\(?['\"].*{escaped_base}['\"]", re.IGNORECASE),
                        self._get_compiled_pattern(rf"include_once\s*\(?['\"].*{escaped_base}['\"]", re.IGNORECASE),
                        self._get_compiled_pattern(rf"['\"].*{escaped_base}['\"]", re.IGNORECASE),
                    ]

                    # Pre-compile patterns for Python files
                    python_patterns = [
                        self._get_compiled_pattern(rf"from\s+{escaped_stem}\s+import", re.IGNORECASE),
                        self._get_compiled_pattern(rf"import\s+{escaped_stem}", re.IGNORECASE),
                    ]

                    # Get functions/classes from the target file
                    target_units = []
                    for node in self.builder.graph.get_all_nodes():
                        data = node.data
                        file_path = str(data.file_path) if data.file_path else ""
                        if base_name in file_path or filename_lower in file_path.lower():
                            target_units.append(data.name)

                    # Filter files to search (exclude target file)
                    files_to_search = [
                        (fp, info) for fp, info in self.indexed_files.items()
                        if base_name not in fp
                    ]

                    if not files_to_search:
                        return []

                    # Read all files in parallel using async I/O (non-blocking)
                    file_paths = [fp for fp, _ in files_to_search]
                    file_contents = await async_read_files_parallel(file_paths, max_concurrent=50)

                    # Process files (CPU-bound, but now with pre-loaded content)
                    for file_path, info in files_to_search:
                        content = file_contents.get(file_path, "")
                        if not content:
                            continue

                        try:
                            matches = []
                            match_score = 0

                            # Check for require/include patterns
                            language = info.get("language", "unknown")
                            patterns = php_patterns if language == "php" else python_patterns
                            content_lines = content.split('\n')

                            for compiled_pattern in patterns:
                                for match in compiled_pattern.finditer(content):
                                    line_num = content[:match.start()].count('\n') + 1
                                    line_text = content_lines[line_num - 1].strip()
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
                                    # Use cached compiled pattern for function calls
                                    func_pattern = self._get_compiled_pattern(rf'\b{re.escape(unit_name)}\s*\(')
                                    for match in func_pattern.finditer(content):
                                        line_num = content[:match.start()].count('\n') + 1
                                        line_text = content_lines[line_num - 1].strip()
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
                    from difflib import SequenceMatcher

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
                    query_words = self._tokenize(query)

                    for node in self.builder.graph.get_all_nodes():
                        data = node.data
                        score = 0

                        name = data.name or ""
                        qualified = getattr(data, 'qualified_name', '') or ""
                        file_path = str(data.file_path) if data.file_path else ""

                        name_lower = name.lower()
                        qualified_lower = qualified.lower()
                        file_lower = file_path.lower()

                        # Tokenize for fuzzy matching (using pre-compiled patterns)
                        name_tokens = self._tokenize(name)
                        qualified_tokens = self._tokenize(qualified)
                        file_tokens = self._tokenize(file_path.split('/')[-1] if file_path else "")
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

            # Get Meilisearch client from searcher for dual-write
            meilisearch_client = None
            if self.services.searcher is not None and hasattr(self.services.searcher, 'client'):
                meilisearch_client = self.services.searcher.client

            # Initialize reference index for O(1) find_references
            ff = get_feature_flags()
            reference_index = None
            if ff.use_reference_index:
                from smart_search.search.reference_index import ReferenceIndex
                ref_index_path = self.config.data_dir / "reference_index.json"
                reference_index = ReferenceIndex.load(ref_index_path)
                logger.info(
                    "ReferenceIndex loaded",
                    stats=reference_index.get_stats(),
                )

            self.services.indexer = SimpleIndexer(
                graph=self.services.graph,
                data_dir=self.config.data_dir,
                meilisearch_client=meilisearch_client,
                embedder=self.services.embedder,
                reference_index=reference_index,
            )
            logger.info(
                "SimpleIndexer initialized",
                dual_write_enabled=meilisearch_client is not None,
                reference_index_enabled=reference_index is not None,
            )
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
        """Inject service dependencies into endpoint modules.

        Uses SearchServiceAdapter to route between HybridSearcher and SimpleIndexer
        based on FF_USE_HYBRID_SEARCHER feature flag.
        """
        # Import endpoint modules
        from smart_search.api.endpoints import search, navigate, analyze, graph, index

        # Create search adapter that wraps both HybridSearcher and SimpleIndexer
        # This allows runtime switching via feature flag
        search_adapter = SearchServiceAdapter(
            hybrid_searcher=self.services.searcher,
            simple_indexer=self.services.indexer,
        )

        # Inject adapter into search endpoints (always inject - adapter handles None cases)
        search.set_searcher(search_adapter)
        logger.info(
            "SearchServiceAdapter injected",
            hybrid_available=self.services.searcher is not None,
            simple_available=self.services.indexer is not None,
        )

        if self.services.graphrag:
            search.set_graphrag(self.services.graphrag)

        # Inject into navigate endpoints
        if self.services.graph:
            navigate.set_graph(self.services.graph)
        # Navigate also uses search adapter for symbol search
        navigate.set_searcher(search_adapter)

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
        # Persist graph using GraphPersistence
        if self.services.graph:
            try:
                from smart_search.graph import GraphPersistence
                persistence = GraphPersistence(self.config.data_dir)
                persistence.save_json(self.services.graph, "graph")
            except Exception:
                pass  # Ignore persistence errors on shutdown

        # Close connections
        if self.services.searcher:
            try:
                await self.services.searcher.close()
            except Exception:
                pass  # Ignore close errors on shutdown

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
            services = {
                "graph": self.services.graph is not None,
                "searcher": self.services.searcher is not None,
                "indexer": self.services.indexer is not None,
                "graphrag": self.services.graphrag is not None,
            }

            # Check Meilisearch health if available
            meilisearch_status = {"available": False, "healthy": False}
            if self.services.searcher is not None:
                try:
                    if hasattr(self.services.searcher, 'client'):
                        client = self.services.searcher.client
                        # health_check() returns bool, not dict
                        is_healthy = await client.health_check()
                        meilisearch_status = {
                            "available": True,
                            "healthy": is_healthy,
                            "url": self.config.meilisearch_url,
                        }
                except Exception as e:
                    meilisearch_status = {
                        "available": True,
                        "healthy": False,
                        "error": str(e),
                    }
            services["meilisearch"] = meilisearch_status

            # Check reference index stats if available
            if self.services.indexer is not None and hasattr(self.services.indexer, 'reference_index'):
                ref_index = self.services.indexer.reference_index
                if ref_index is not None:
                    services["reference_index"] = ref_index.get_stats()

            # Overall status
            all_healthy = all([
                services["graph"],
                services["indexer"],
            ])

            return {
                "status": "healthy" if all_healthy else "degraded",
                "services": services,
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
