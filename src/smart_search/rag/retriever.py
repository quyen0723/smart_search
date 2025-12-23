"""Context retriever for GraphRAG.

Retrieves relevant code context using vector search and graph expansion.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from smart_search.graph.engine import CodeGraph
from smart_search.graph.models import EdgeType, NodeType
from smart_search.rag.prompts import (
    CodeContext,
    CommunitySummary,
    PromptContext,
    QueryType,
    RelationshipPath,
)
from smart_search.search.hybrid import HybridSearcher
from smart_search.search.schemas import SearchQuery, SearchResult, SearchType
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetrieverConfig:
    """Configuration for context retriever.

    Attributes:
        max_vector_results: Maximum vector search results.
        max_graph_depth: Maximum graph traversal depth.
        max_context_items: Maximum context items to return.
        include_callers: Include caller relationships.
        include_callees: Include callee relationships.
        include_siblings: Include sibling (same parent) nodes.
        semantic_threshold: Minimum semantic similarity (0-1).
    """

    max_vector_results: int = 20
    max_graph_depth: int = 2
    max_context_items: int = 10
    include_callers: bool = True
    include_callees: bool = True
    include_siblings: bool = False
    semantic_threshold: float = 0.5


@dataclass
class RetrievalResult:
    """Result of context retrieval.

    Attributes:
        query: Original query.
        query_type: Type of query performed.
        contexts: Retrieved code contexts.
        relationships: Discovered relationships.
        communities: Relevant communities.
        processing_time_ms: Time taken in milliseconds.
    """

    query: str
    query_type: QueryType
    contexts: list[CodeContext] = field(default_factory=list)
    relationships: list[RelationshipPath] = field(default_factory=list)
    communities: list[CommunitySummary] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_prompt_context(self, max_tokens: int = 4000) -> PromptContext:
        """Convert to PromptContext.

        Args:
            max_tokens: Maximum prompt tokens.

        Returns:
            PromptContext instance.
        """
        return PromptContext(
            query=self.query,
            query_type=self.query_type,
            code_contexts=self.contexts,
            relationships=self.relationships,
            communities=self.communities,
            max_tokens=max_tokens,
        )

    @property
    def has_results(self) -> bool:
        """Whether any results were found."""
        return bool(self.contexts or self.relationships or self.communities)


class ContextRetriever:
    """Retrieves code context using vector search and graph expansion."""

    def __init__(
        self,
        searcher: HybridSearcher,
        graph: CodeGraph,
        config: RetrieverConfig | None = None,
    ) -> None:
        """Initialize context retriever.

        Args:
            searcher: Hybrid searcher for vector search.
            graph: Code graph for relationship expansion.
            config: Retriever configuration.
        """
        self.searcher = searcher
        self.graph = graph
        self.config = config or RetrieverConfig()
        self._community_cache: dict[str, CommunitySummary] = {}

    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.LOCAL,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve context for a query.

        Args:
            query: User query string.
            query_type: Type of retrieval to perform.
            filters: Optional search filters.

        Returns:
            RetrievalResult with contexts and relationships.
        """
        import time

        start = time.time()

        result = RetrievalResult(query=query, query_type=query_type)

        # Step 1: Vector search
        search_results = await self._vector_search(query, filters)

        # Step 2: Convert to code contexts
        result.contexts = self._search_to_contexts(search_results)

        # Step 3: Graph expansion based on query type
        if query_type == QueryType.LOCAL:
            result.relationships = await self._expand_local(result.contexts)
        elif query_type == QueryType.GLOBAL:
            result.communities = await self._get_communities(result.contexts)
        elif query_type == QueryType.DRIFT:
            result.relationships = await self._expand_local(result.contexts)
            result.communities = await self._get_communities(result.contexts)

        # Step 4: Limit results
        result.contexts = result.contexts[: self.config.max_context_items]

        result.processing_time_ms = (time.time() - start) * 1000

        logger.info(
            "Context retrieved",
            query=query[:50],
            query_type=query_type.value,
            contexts=len(result.contexts),
            relationships=len(result.relationships),
            communities=len(result.communities),
            time_ms=result.processing_time_ms,
        )

        return result

    async def retrieve_similar(
        self,
        code_content: str,
        limit: int = 10,
        exclude_ids: list[str] | None = None,
    ) -> list[CodeContext]:
        """Retrieve similar code contexts.

        Args:
            code_content: Code to find similar items for.
            limit: Maximum results.
            exclude_ids: IDs to exclude from results.

        Returns:
            List of similar code contexts.
        """
        search_result = await self.searcher.search_similar(
            code_content,
            limit=limit,
            exclude_ids=exclude_ids,
        )
        return self._search_to_contexts(search_result)

    async def retrieve_by_relationship(
        self,
        code_id: str,
        relationship: str,
        depth: int = 1,
    ) -> list[CodeContext]:
        """Retrieve code by relationship.

        Args:
            code_id: Source code unit ID.
            relationship: Relationship type (callers, callees, etc).
            depth: Traversal depth.

        Returns:
            List of related code contexts.
        """
        related_ids: list[str] = []

        if relationship == "callers":
            related_ids = self.graph.get_callers(code_id, depth)
        elif relationship == "callees":
            related_ids = self.graph.get_callees(code_id, depth)
        elif relationship == "ancestors":
            related_ids = self.graph.get_ancestors(code_id, depth)
        elif relationship == "descendants":
            related_ids = self.graph.get_descendants(code_id, depth)
        elif relationship == "siblings":
            related_ids = self._get_siblings(code_id)

        contexts = []
        for rid in related_ids:
            node = self.graph.get_node(rid)
            if node:
                contexts.append(self._node_to_context(node, relevance_score=0.8))

        return contexts

    async def _vector_search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Perform vector search.

        Args:
            query: Search query.
            filters: Optional filters.

        Returns:
            Search result.
        """
        search_query = SearchQuery(
            query=query,
            search_type=SearchType.HYBRID,
            limit=self.config.max_vector_results,
        )

        return await self.searcher.search(search_query)

    def _search_to_contexts(self, result: SearchResult) -> list[CodeContext]:
        """Convert search results to code contexts.

        Args:
            result: Search result.

        Returns:
            List of code contexts.
        """
        contexts = []

        for hit in result.hits:
            # Filter by threshold
            if hit.score < self.config.semantic_threshold:
                continue

            ctx = CodeContext(
                code_id=hit.id,
                name=hit.name,
                qualified_name=hit.qualified_name,
                code_type=hit.code_type,
                file_path=hit.file_path,
                line_start=hit.line_start,
                line_end=hit.line_end,
                content=hit.content,
                language=hit.language,
                docstring=hit.metadata.get("docstring") if hit.metadata else None,
                relevance_score=hit.score,
            )
            contexts.append(ctx)

        return contexts

    def _node_to_context(
        self,
        node: Any,
        relevance_score: float = 0.0,
    ) -> CodeContext:
        """Convert graph node to code context.

        Args:
            node: Graph node.
            relevance_score: Relevance score.

        Returns:
            CodeContext instance.
        """
        return CodeContext(
            code_id=node.id,
            name=node.name,
            qualified_name=node.qualified_name or node.name,
            code_type=node.type.value if hasattr(node.type, "value") else str(node.type),
            file_path=Path(node.file_path) if isinstance(node.file_path, str) else node.file_path,
            line_start=node.line_start,
            line_end=node.line_end,
            content=getattr(node, "content", ""),
            language=node.language if hasattr(node, "language") else "unknown",
            docstring=getattr(node, "docstring", None),
            relevance_score=relevance_score,
        )

    async def _expand_local(
        self,
        contexts: list[CodeContext],
    ) -> list[RelationshipPath]:
        """Expand contexts with local relationships.

        Args:
            contexts: Initial contexts.

        Returns:
            Discovered relationship paths.
        """
        relationships = []
        seen = set()

        for ctx in contexts:
            code_id = ctx.code_id

            # Get callers
            if self.config.include_callers:
                callers = self.graph.get_callers(code_id, depth=1)
                for caller_id in callers[:5]:  # Limit per node
                    key = (caller_id, code_id)
                    if key not in seen:
                        seen.add(key)
                        relationships.append(
                            RelationshipPath(
                                source=caller_id,
                                target=code_id,
                                path=[(code_id, "CALLS")],
                                description=f"{caller_id} calls {code_id}",
                            )
                        )

            # Get callees
            if self.config.include_callees:
                callees = self.graph.get_callees(code_id, depth=1)
                for callee_id in callees[:5]:
                    key = (code_id, callee_id)
                    if key not in seen:
                        seen.add(key)
                        relationships.append(
                            RelationshipPath(
                                source=code_id,
                                target=callee_id,
                                path=[(callee_id, "CALLS")],
                                description=f"{code_id} calls {callee_id}",
                            )
                        )

        return relationships

    def _get_siblings(self, code_id: str) -> list[str]:
        """Get sibling nodes (same parent).

        Args:
            code_id: Node ID.

        Returns:
            List of sibling IDs.
        """
        # Find parent
        node = self.graph.get_node(code_id)
        if not node or not hasattr(node, "parent_id") or not node.parent_id:
            return []

        # Get all children of parent
        parent_id = node.parent_id
        children = self.graph.get_children(parent_id)

        # Exclude self
        return [c for c in children if c != code_id]

    async def _get_communities(
        self,
        contexts: list[CodeContext],
    ) -> list[CommunitySummary]:
        """Get community summaries for contexts.

        Args:
            contexts: Code contexts.

        Returns:
            Relevant community summaries.
        """
        communities = []
        seen_communities = set()

        for ctx in contexts:
            # Get community for this node
            community_id = self.graph.get_node_community(ctx.code_id)
            if community_id and community_id not in seen_communities:
                seen_communities.add(community_id)

                # Check cache
                if community_id in self._community_cache:
                    communities.append(self._community_cache[community_id])
                else:
                    # Generate summary
                    summary = self._generate_community_summary(community_id)
                    if summary:
                        self._community_cache[community_id] = summary
                        communities.append(summary)

        return communities

    def _generate_community_summary(self, community_id: str) -> CommunitySummary | None:
        """Generate summary for a community.

        Args:
            community_id: Community identifier.

        Returns:
            CommunitySummary or None.
        """
        members = self.graph.get_community_members(community_id)
        if not members:
            return None

        # Get key components (high-degree nodes)
        key_components = []
        for member_id in members[:10]:
            node = self.graph.get_node(member_id)
            if node:
                key_components.append(node.name)

        # Generate description based on common patterns
        description = self._describe_community(members)

        return CommunitySummary(
            community_id=community_id,
            name=f"Community {community_id}",
            description=description,
            key_components=key_components[:5],
            size=len(members),
        )

    def _describe_community(self, member_ids: list[str]) -> str:
        """Generate description for community.

        Args:
            member_ids: Community member IDs.

        Returns:
            Description string.
        """
        if not member_ids:
            return "Empty community"

        # Analyze member types
        type_counts: dict[str, int] = {}
        file_paths: set[str] = set()

        for mid in member_ids[:50]:  # Limit analysis
            node = self.graph.get_node(mid)
            if node:
                node_type = node.type.value if hasattr(node.type, "value") else str(node.type)
                type_counts[node_type] = type_counts.get(node_type, 0) + 1
                if hasattr(node, "file_path"):
                    file_paths.add(str(node.file_path))

        # Build description
        parts = []

        if type_counts:
            main_type = max(type_counts, key=type_counts.get)
            parts.append(f"Primarily contains {main_type}s")

        if file_paths:
            parts.append(f"across {len(file_paths)} file(s)")

        return " ".join(parts) if parts else "Code module"

    def clear_community_cache(self) -> None:
        """Clear the community summary cache."""
        self._community_cache.clear()


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self) -> None:
        """Initialize mock retriever."""
        self._contexts: list[CodeContext] = []
        self._relationships: list[RelationshipPath] = []
        self._communities: list[CommunitySummary] = []

    def set_contexts(self, contexts: list[CodeContext]) -> None:
        """Set contexts to return."""
        self._contexts = contexts

    def set_relationships(self, relationships: list[RelationshipPath]) -> None:
        """Set relationships to return."""
        self._relationships = relationships

    def set_communities(self, communities: list[CommunitySummary]) -> None:
        """Set communities to return."""
        self._communities = communities

    async def retrieve(
        self,
        query: str,
        query_type: QueryType = QueryType.LOCAL,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Mock retrieve."""
        return RetrievalResult(
            query=query,
            query_type=query_type,
            contexts=self._contexts,
            relationships=self._relationships,
            communities=self._communities,
            processing_time_ms=1.0,
        )

    async def retrieve_similar(
        self,
        code_content: str,
        limit: int = 10,
        exclude_ids: list[str] | None = None,
    ) -> list[CodeContext]:
        """Mock retrieve similar."""
        return self._contexts[:limit]

    async def retrieve_by_relationship(
        self,
        code_id: str,
        relationship: str,
        depth: int = 1,
    ) -> list[CodeContext]:
        """Mock retrieve by relationship."""
        return self._contexts
