"""GraphRAG module for intelligent code understanding.

This module provides Graph-enhanced Retrieval Augmented Generation
for source code navigation and understanding.
"""

from smart_search.rag.generator import (
    AnthropicClient,
    CachedGenerator,
    GenerationResult,
    GeneratorConfig,
    GeneratorModel,
    LLMClient,
    MockLLMClient,
    OpenAIClient,
    ResponseGenerator,
)
from smart_search.rag.graphrag import (
    GraphRAG,
    GraphRAGConfig,
    GraphRAGResult,
    MockGraphRAG,
    SearchMode,
    create_mock_graphrag,
)
from smart_search.rag.prompts import (
    CodeContext,
    CommunitySummary,
    ExplanationBuilder,
    PromptBuilder,
    PromptContext,
    QueryType,
    RelationshipPath,
)
from smart_search.rag.retriever import (
    ContextRetriever,
    MockRetriever,
    RetrievalResult,
    RetrieverConfig,
)

__all__ = [
    # Prompts
    "QueryType",
    "CodeContext",
    "RelationshipPath",
    "CommunitySummary",
    "PromptContext",
    "PromptBuilder",
    "ExplanationBuilder",
    # Retriever
    "RetrieverConfig",
    "RetrievalResult",
    "ContextRetriever",
    "MockRetriever",
    # Generator
    "GeneratorModel",
    "GeneratorConfig",
    "GenerationResult",
    "LLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "MockLLMClient",
    "ResponseGenerator",
    "CachedGenerator",
    # GraphRAG
    "SearchMode",
    "GraphRAGConfig",
    "GraphRAGResult",
    "GraphRAG",
    "MockGraphRAG",
    "create_mock_graphrag",
]
