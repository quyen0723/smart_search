"""Prompt templates for GraphRAG.

Provides prompt builders for various RAG operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class QueryType(str, Enum):
    """Type of GraphRAG query."""

    LOCAL = "local"  # Focus on direct relationships
    GLOBAL = "global"  # Include community context
    DRIFT = "drift"  # Combine local and global


@dataclass
class CodeContext:
    """Code context for prompt building.

    Attributes:
        code_id: Unique identifier.
        name: Code unit name.
        qualified_name: Full qualified name.
        code_type: Type (function, class, etc).
        file_path: Path to file.
        line_start: Starting line.
        line_end: Ending line.
        content: Source code content.
        language: Programming language.
        docstring: Documentation string.
        relevance_score: Relevance to query (0-1).
    """

    code_id: str
    name: str
    qualified_name: str
    code_type: str
    file_path: Path
    line_start: int
    line_end: int
    content: str
    language: str
    docstring: str | None = None
    relevance_score: float = 0.0

    @property
    def line_count(self) -> int:
        """Get line count."""
        return self.line_end - self.line_start + 1

    @property
    def formatted(self) -> str:
        """Get formatted representation."""
        return self.to_prompt_str(include_content=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "code_id": self.code_id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "code_type": self.code_type,
            "file_path": str(self.file_path),
            "line_start": self.line_start,
            "line_end": self.line_end,
            "content": self.content,
            "language": self.language,
            "docstring": self.docstring,
            "relevance_score": self.relevance_score,
        }

    def to_prompt_str(self, include_content: bool = True) -> str:
        """Format as prompt string.

        Args:
            include_content: Whether to include source code.

        Returns:
            Formatted string for prompt.
        """
        parts = [
            f"[{self.code_type.upper()}] {self.qualified_name}",
            f"  File: {self.file_path}:{self.line_start}-{self.line_end}",
            f"  Language: {self.language}",
        ]

        if self.docstring:
            # Truncate long docstrings
            doc = self.docstring[:200] + "..." if len(self.docstring) > 200 else self.docstring
            parts.append(f"  Doc: {doc}")

        if include_content:
            # Indent content
            indented = "\n".join(f"    {line}" for line in self.content.split("\n")[:30])
            parts.append(f"  Code:\n{indented}")

        return "\n".join(parts)


@dataclass
class RelationshipPath:
    """A path of relationships between code units.

    Attributes:
        source: Source code unit ID.
        target: Target code unit ID.
        path: List of (node_id, edge_type) tuples.
        description: Human-readable description.
    """

    source: str
    target: str
    path: list[tuple[str, str]]
    description: str | None = None

    @property
    def depth(self) -> int:
        """Get path depth."""
        return len(self.path)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "path": self.path,
            "description": self.description,
        }

    def to_prompt_str(self) -> str:
        """Format as prompt string."""
        if not self.path:
            return f"{self.source} -> {self.target}"

        parts = [self.source]
        for node_id, edge_type in self.path:
            parts.append(f"--[{edge_type}]--> {node_id}")

        path_str = " ".join(parts)
        if self.description:
            return f"{path_str}\n  ({self.description})"
        return path_str


@dataclass
class CommunitySummary:
    """Summary of a code community/module.

    Attributes:
        community_id: Unique identifier.
        name: Community name.
        description: Summary description.
        key_components: Main components in community.
        size: Number of nodes.
    """

    community_id: str
    name: str
    description: str
    key_components: list[str] = field(default_factory=list)
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "community_id": self.community_id,
            "name": self.name,
            "description": self.description,
            "key_components": self.key_components,
            "size": self.size,
        }

    def to_prompt_str(self) -> str:
        """Format as prompt string."""
        parts = [
            f"[COMMUNITY] {self.name} ({self.size} components)",
            f"  {self.description}",
        ]

        if self.key_components:
            components = ", ".join(self.key_components[:5])
            parts.append(f"  Key: {components}")

        return "\n".join(parts)


@dataclass
class PromptContext:
    """Full context for prompt building.

    Attributes:
        query: User query.
        query_type: Type of query.
        code_contexts: Retrieved code contexts.
        relationships: Relationship paths.
        communities: Community summaries.
        max_tokens: Maximum prompt tokens.
    """

    query: str
    query_type: QueryType = QueryType.LOCAL
    code_contexts: list[CodeContext] = field(default_factory=list)
    relationships: list[RelationshipPath] = field(default_factory=list)
    communities: list[CommunitySummary] = field(default_factory=list)
    max_tokens: int = 4000

    @property
    def has_context(self) -> bool:
        """Whether any context is available."""
        return bool(self.code_contexts or self.relationships or self.communities)

    @property
    def context_count(self) -> int:
        """Get total context count."""
        return len(self.code_contexts) + len(self.relationships) + len(self.communities)

    def estimate_tokens(self) -> int:
        """Estimate token count for context.

        Returns:
            Estimated token count.
        """
        # Rough estimate: 1 token ≈ 4 chars
        total_chars = len(self.query)

        for ctx in self.code_contexts:
            total_chars += len(ctx.to_prompt_str())

        for rel in self.relationships:
            total_chars += len(rel.to_prompt_str())

        for comm in self.communities:
            total_chars += len(comm.to_prompt_str())

        return total_chars // 4


class PromptBuilder:
    """Builds prompts for GraphRAG queries."""

    # System prompts for different query types
    SYSTEM_PROMPTS = {
        QueryType.LOCAL: """You are a code analysis assistant. Analyze the provided code context
and answer questions about specific functions, classes, and their direct relationships.
Focus on precise, technical answers based on the code shown.""",
        QueryType.GLOBAL: """You are a codebase architecture assistant. Analyze the provided
code communities and high-level structure to answer questions about the overall design,
patterns, and organization of the codebase.""",
        QueryType.DRIFT: """You are a comprehensive code analysis assistant. Use both
the specific code details and the high-level architecture context to provide
complete answers about the codebase.""",
    }

    # Templates for different sections
    SECTION_TEMPLATES = {
        "code_context": """
## Relevant Code

{contexts}
""",
        "relationships": """
## Code Relationships

{relationships}
""",
        "communities": """
## Code Communities/Modules

{communities}
""",
        "query": """
## Question

{query}

Please provide a clear, accurate answer based on the code context above.
""",
    }

    def __init__(
        self,
        max_code_contexts: int = 10,
        max_relationships: int = 10,
        max_communities: int = 5,
        include_code_content: bool = True,
    ) -> None:
        """Initialize prompt builder.

        Args:
            max_code_contexts: Maximum code contexts to include.
            max_relationships: Maximum relationships to include.
            max_communities: Maximum communities to include.
            include_code_content: Whether to include source code.
        """
        self.max_code_contexts = max_code_contexts
        self.max_relationships = max_relationships
        self.max_communities = max_communities
        self.include_code_content = include_code_content

    def build_system_prompt(self, context: PromptContext | QueryType) -> str:
        """Build system prompt for query type.

        Args:
            context: PromptContext or QueryType.

        Returns:
            System prompt string.
        """
        if isinstance(context, PromptContext):
            query_type = context.query_type
        else:
            query_type = context
        return self.SYSTEM_PROMPTS.get(query_type, self.SYSTEM_PROMPTS[QueryType.LOCAL])

    def build_user_prompt(self, context: PromptContext) -> str:
        """Build user prompt from context.

        Args:
            context: Prompt context.

        Returns:
            User prompt string.
        """
        sections = []

        # Add code contexts (sorted by relevance)
        if context.code_contexts:
            sorted_contexts = sorted(
                context.code_contexts,
                key=lambda c: c.relevance_score,
                reverse=True,
            )[: self.max_code_contexts]

            context_strs = [
                c.to_prompt_str(include_content=self.include_code_content)
                for c in sorted_contexts
            ]
            sections.append(
                self.SECTION_TEMPLATES["code_context"].format(
                    contexts="\n\n".join(context_strs)
                )
            )

        # Add relationships
        if context.relationships and context.query_type != QueryType.GLOBAL:
            rel_strs = [
                r.to_prompt_str() for r in context.relationships[: self.max_relationships]
            ]
            sections.append(
                self.SECTION_TEMPLATES["relationships"].format(
                    relationships="\n".join(rel_strs)
                )
            )

        # Add communities for global/drift queries
        if context.communities and context.query_type in (QueryType.GLOBAL, QueryType.DRIFT):
            comm_strs = [
                c.to_prompt_str() for c in context.communities[: self.max_communities]
            ]
            sections.append(
                self.SECTION_TEMPLATES["communities"].format(
                    communities="\n".join(comm_strs)
                )
            )

        # Add query
        sections.append(self.SECTION_TEMPLATES["query"].format(query=context.query))

        return "\n".join(sections)

    def build_full_prompt(self, context: PromptContext) -> dict[str, str]:
        """Build complete prompt with system and user parts.

        Args:
            context: Prompt context.

        Returns:
            Dict with 'system' and 'user' keys.
        """
        return {
            "system": self.build_system_prompt(context.query_type),
            "user": self.build_user_prompt(context),
        }

    def _format_code_context(self, context: CodeContext) -> str:
        """Format a single code context.

        Args:
            context: Code context.

        Returns:
            Formatted string.
        """
        return context.to_prompt_str(include_content=self.include_code_content)

    def _format_relationships(self, relationships: list[RelationshipPath]) -> str:
        """Format relationships.

        Args:
            relationships: List of relationships.

        Returns:
            Formatted string.
        """
        return "\n".join(r.to_prompt_str() for r in relationships)

    def _format_communities(self, communities: list[CommunitySummary]) -> str:
        """Format communities.

        Args:
            communities: List of communities.

        Returns:
            Formatted string.
        """
        return "\n".join(c.to_prompt_str() for c in communities)

    def truncate_to_max_tokens(
        self,
        context: PromptContext,
        max_tokens: int | None = None,
    ) -> PromptContext:
        """Truncate context to fit within token limit.

        Args:
            context: Prompt context.
            max_tokens: Maximum tokens (uses context.max_tokens if None).

        Returns:
            Truncated context.
        """
        max_tokens = max_tokens or context.max_tokens

        # Start with full context
        current = PromptContext(
            query=context.query,
            query_type=context.query_type,
            code_contexts=list(context.code_contexts),
            relationships=list(context.relationships),
            communities=list(context.communities),
            max_tokens=max_tokens,
        )

        # Iteratively remove least relevant items
        while current.estimate_tokens() > max_tokens:
            # Remove least relevant code context
            if len(current.code_contexts) > 1:
                current.code_contexts = sorted(
                    current.code_contexts,
                    key=lambda c: c.relevance_score,
                    reverse=True,
                )[:-1]
                continue

            # Remove relationships
            if current.relationships:
                current.relationships = current.relationships[:-1]
                continue

            # Remove communities
            if current.communities:
                current.communities = current.communities[:-1]
                continue

            # Truncate remaining code content
            if current.code_contexts and current.code_contexts[0].content:
                ctx = current.code_contexts[0]
                lines = ctx.content.split("\n")
                if len(lines) > 5:
                    current.code_contexts[0] = CodeContext(
                        code_id=ctx.code_id,
                        name=ctx.name,
                        qualified_name=ctx.qualified_name,
                        code_type=ctx.code_type,
                        file_path=ctx.file_path,
                        line_start=ctx.line_start,
                        line_end=ctx.line_end,
                        content="\n".join(lines[: len(lines) // 2]) + "\n... (truncated)",
                        language=ctx.language,
                        docstring=ctx.docstring,
                        relevance_score=ctx.relevance_score,
                    )
                    continue

            # Can't reduce further
            break

        return current


class ExplanationBuilder:
    """Builds explanations for code relationships."""

    EDGE_DESCRIPTIONS = {
        "calls": "calls",
        "imports": "imports",
        "inherits": "inherits from",
        "contains": "contains",
        "uses": "uses",
        "implements": "implements",
        "overrides": "overrides",
    }

    def explain_path(self, path: RelationshipPath) -> str:
        """Generate natural language explanation of path.

        Args:
            path: Relationship path.

        Returns:
            Human-readable explanation.
        """
        if not path.path:
            return f"{path.source} is directly related to {path.target}"

        explanations = []
        current = path.source

        for node_id, edge_type in path.path:
            edge_desc = self.EDGE_DESCRIPTIONS.get(edge_type.lower(), "relates to")
            explanations.append(f"{current} {edge_desc} {node_id}")
            current = node_id

        return " → ".join(explanations)

    def explain_code(self, context: CodeContext | list[CodeContext]) -> str:
        """Generate explanation for code context(s).

        Args:
            context: Single code context or list of contexts.

        Returns:
            Human-readable explanation.
        """
        if isinstance(context, list):
            explanations = []
            for ctx in context:
                explanations.append(self._explain_single_code(ctx))
            return "\n\n".join(explanations)
        return self._explain_single_code(context)

    def _explain_single_code(self, context: CodeContext) -> str:
        """Explain a single code context.

        Args:
            context: Code context.

        Returns:
            Explanation string.
        """
        parts = [
            f"**{context.name}** ({context.code_type})",
            f"Location: {self._format_location(context)}",
        ]

        if context.docstring:
            parts.append(f"Description: {context.docstring}")

        return "\n".join(parts)

    def _format_location(self, context: CodeContext) -> str:
        """Format file location.

        Args:
            context: Code context.

        Returns:
            Formatted location string.
        """
        filename = context.file_path.name if hasattr(context.file_path, "name") else str(context.file_path).split("/")[-1]
        return f"{filename}:{context.line_start}-{context.line_end}"

    def explain_impact(
        self,
        source: str,
        affected: list[tuple[str, int]],
    ) -> str:
        """Explain impact of changing source.

        Args:
            source: Source code unit being changed.
            affected: List of (affected_unit, distance) tuples.

        Returns:
            Impact explanation.
        """
        if not affected:
            return f"Changing {source} has no detected downstream impact."

        direct = [u for u, d in affected if d == 1]
        indirect = [u for u, d in affected if d > 1]

        parts = [f"Changing {source} may affect:"]

        if direct:
            parts.append(f"  Direct dependents ({len(direct)}): {', '.join(direct[:5])}")
            if len(direct) > 5:
                parts.append(f"    ... and {len(direct) - 5} more")

        if indirect:
            parts.append(f"  Indirect dependents ({len(indirect)}): {', '.join(indirect[:5])}")
            if len(indirect) > 5:
                parts.append(f"    ... and {len(indirect) - 5} more")

        return "\n".join(parts)


# Convenience functions
def build_code_context_prompt(
    contexts: list[CodeContext],
    query: str,
    query_type: QueryType = QueryType.LOCAL,
) -> dict[str, str]:
    """Build a prompt from code contexts and query.

    Args:
        contexts: Code contexts.
        query: User query.
        query_type: Type of query.

    Returns:
        Dict with system and user prompts.
    """
    builder = PromptBuilder()
    context = PromptContext(
        query=query,
        query_type=query_type,
        code_contexts=contexts,
    )
    return builder.build_full_prompt(context)


def build_relationship_prompt(
    relationships: list[RelationshipPath],
    query: str,
) -> str:
    """Build a prompt focused on relationships.

    Args:
        relationships: Relationship paths.
        query: User query.

    Returns:
        Prompt string.
    """
    builder = ExplanationBuilder()
    explanations = [builder.explain_path(r) for r in relationships]

    return f"""Based on these code relationships:

{chr(10).join(explanations)}

{query}"""
