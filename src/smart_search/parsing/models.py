"""Data models for code parsing and AST analysis.

This module defines the core data structures used to represent
parsed code units, their metadata, and relationships.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CodeUnitType(str, Enum):
    """Types of code units that can be extracted from source code."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    VARIABLE = "variable"
    IMPORT = "import"
    DECORATOR = "decorator"


class Language(str, Enum):
    """Supported programming languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    CPP = "cpp"
    C = "c"
    PHP = "php"

    @classmethod
    def from_extension(cls, ext: str) -> "Language | None":
        """Get language from file extension.

        Args:
            ext: File extension (with or without dot).

        Returns:
            Language enum or None if not supported.
        """
        ext = ext.lstrip(".")
        mapping = {
            "py": cls.PYTHON,
            "js": cls.JAVASCRIPT,
            "jsx": cls.JAVASCRIPT,
            "ts": cls.TYPESCRIPT,
            "tsx": cls.TYPESCRIPT,
            "java": cls.JAVA,
            "go": cls.GO,
            "rs": cls.RUST,
            "cpp": cls.CPP,
            "cc": cls.CPP,
            "cxx": cls.CPP,
            "c": cls.C,
            "h": cls.C,
            "php": cls.PHP,
        }
        return mapping.get(ext.lower())


@dataclass
class Position:
    """Position in source code."""

    line: int  # 1-indexed
    column: int  # 0-indexed

    def __post_init__(self) -> None:
        if self.line < 1:
            raise ValueError("Line must be >= 1")
        if self.column < 0:
            raise ValueError("Column must be >= 0")


@dataclass
class Span:
    """A span of source code with start and end positions."""

    start: Position
    end: Position

    @property
    def start_line(self) -> int:
        """Get starting line number."""
        return self.start.line

    @property
    def end_line(self) -> int:
        """Get ending line number."""
        return self.end.line

    @property
    def line_count(self) -> int:
        """Get number of lines spanned."""
        return self.end.line - self.start.line + 1


@dataclass
class Parameter:
    """Function or method parameter."""

    name: str
    type_annotation: str | None = None
    default_value: str | None = None
    is_variadic: bool = False  # *args
    is_keyword: bool = False  # **kwargs


@dataclass
class CodeUnit:
    """A unit of code extracted from source (function, class, method, etc.).

    This is the primary data structure representing parsed code elements.
    Each CodeUnit can be indexed, searched, and analyzed.
    """

    # Identity
    id: str  # Unique identifier: file_path::qualified_name
    name: str  # Simple name (e.g., "my_function")
    qualified_name: str  # Full qualified name (e.g., "module.MyClass.my_method")
    type: CodeUnitType

    # Location
    file_path: Path
    span: Span
    language: Language

    # Content
    content: str  # Full source code of this unit
    docstring: str | None = None
    signature: str | None = None  # For functions/methods

    # Metadata
    decorators: list[str] = field(default_factory=list)
    parameters: list[Parameter] = field(default_factory=list)
    return_type: str | None = None
    base_classes: list[str] = field(default_factory=list)  # For classes

    # References (populated during graph building)
    imports: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # Functions/methods this unit calls
    references: list[str] = field(default_factory=list)  # Variables/attributes referenced

    # Parent relationship
    parent_id: str | None = None  # ID of containing class/module

    # Hash for change detection
    content_hash: str | None = None

    @property
    def start_line(self) -> int:
        """Get starting line number."""
        return self.span.start_line

    @property
    def end_line(self) -> int:
        """Get ending line number."""
        return self.span.end_line

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/indexing."""
        return {
            "id": self.id,
            "name": self.name,
            "qualified_name": self.qualified_name,
            "type": self.type.value,
            "file_path": str(self.file_path),
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language.value,
            "content": self.content,
            "docstring": self.docstring,
            "signature": self.signature,
            "decorators": self.decorators,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type_annotation,
                    "default": p.default_value,
                }
                for p in self.parameters
            ],
            "return_type": self.return_type,
            "base_classes": self.base_classes,
            "imports": self.imports,
            "calls": self.calls,
            "references": self.references,
            "parent_id": self.parent_id,
            "content_hash": self.content_hash,
        }


@dataclass
class ParsedFile:
    """Result of parsing a single source file."""

    file_path: Path
    language: Language
    content: str
    content_hash: str

    # Extracted units
    units: list[CodeUnit] = field(default_factory=list)

    # File-level metadata
    imports: list[str] = field(default_factory=list)
    module_docstring: str | None = None

    # Parsing metadata
    parse_errors: list[str] = field(default_factory=list)
    parse_time_ms: float = 0.0

    @property
    def has_errors(self) -> bool:
        """Check if parsing had errors."""
        return len(self.parse_errors) > 0

    @property
    def unit_count(self) -> int:
        """Get number of extracted units."""
        return len(self.units)


@dataclass
class CodeChunk:
    """A chunk of code for embedding generation.

    Chunks are created from CodeUnits but may be split if the unit
    is too large for the embedding model's context window.
    """

    id: str  # Unique chunk ID
    unit_id: str  # Parent CodeUnit ID
    content: str
    chunk_index: int  # Index within the parent unit (0 if not split)
    total_chunks: int  # Total number of chunks for this unit

    # Context for better embeddings
    context_before: str = ""  # Code/docstring before this chunk
    context_after: str = ""  # Code after this chunk

    # Metadata inherited from parent
    file_path: str = ""
    language: str = ""
    unit_type: str = ""
    unit_name: str = ""

    @property
    def is_split(self) -> bool:
        """Check if this chunk is part of a split unit."""
        return self.total_chunks > 1

    def to_embedding_text(self) -> str:
        """Generate text suitable for embedding.

        Combines context and content for better semantic representation.
        """
        parts = []
        if self.context_before:
            parts.append(self.context_before)
        parts.append(self.content)
        if self.context_after:
            parts.append(self.context_after)
        return "\n".join(parts)
