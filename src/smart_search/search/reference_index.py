"""Reference Index for O(1) find_references lookups.

This module provides an inverted index for fast reference lookups.
Instead of scanning all files to find references, we pre-index references
at parse time and query the index in O(1).

Schema Design:
    The index maps symbol names/identifiers to their usage locations.

    Primary Index (symbol -> usages):
        {
            "function_name": [
                {"file": "path/to/file.py", "line": 42, "type": "call"},
                {"file": "path/to/other.py", "line": 100, "type": "import"},
            ],
            "ClassName": [
                {"file": "path/to/file.py", "line": 15, "type": "instantiation"},
                {"file": "path/to/other.py", "line": 5, "type": "inheritance"},
            ]
        }

    File Index (file -> symbols defined):
        {
            "path/to/file.py": ["function_name", "ClassName", "variable"],
        }

    This allows:
        1. O(1) lookup of all references to a symbol
        2. O(n) removal of all references from a file when re-indexing
        3. Efficient partial updates

Feature Flag: FF_USE_REFERENCE_INDEX
    When enabled, find_references() uses this index instead of file scanning.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterator
import json

from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class ReferenceType(str, Enum):
    """Type of reference to a symbol."""
    CALL = "call"                    # Function/method call
    IMPORT = "import"                # Import statement
    INHERITANCE = "inheritance"      # Class inheritance
    INSTANTIATION = "instantiation"  # Class instantiation
    ASSIGNMENT = "assignment"        # Variable assignment
    ARGUMENT = "argument"            # Function argument type
    RETURN = "return"                # Return type
    ATTRIBUTE = "attribute"          # Attribute access
    UNKNOWN = "unknown"              # Unknown reference type


@dataclass
class ReferenceLocation:
    """Location where a symbol is referenced."""
    file_path: str
    line: int
    column: int = 0
    ref_type: ReferenceType = ReferenceType.UNKNOWN
    context: str = ""  # Line content for display

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "ref_type": self.ref_type.value,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceLocation":
        """Create from dictionary."""
        return cls(
            file_path=data["file_path"],
            line=data["line"],
            column=data.get("column", 0),
            ref_type=ReferenceType(data.get("ref_type", "unknown")),
            context=data.get("context", ""),
        )


@dataclass
class SymbolDefinition:
    """Definition of a symbol (function, class, variable)."""
    name: str
    qualified_name: str
    symbol_type: str  # function, class, variable, etc.
    file_path: str
    line_start: int
    line_end: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "symbol_type": self.symbol_type,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbolDefinition":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            qualified_name=data["qualified_name"],
            symbol_type=data["symbol_type"],
            file_path=data["file_path"],
            line_start=data["line_start"],
            line_end=data["line_end"],
        )


class ReferenceIndex:
    """Inverted index for fast reference lookups.

    Provides O(1) lookup of all references to a symbol.
    """

    def __init__(self):
        """Initialize empty index."""
        # symbol_name -> list of ReferenceLocation
        self._symbol_refs: dict[str, list[ReferenceLocation]] = {}
        # qualified_name -> SymbolDefinition
        self._definitions: dict[str, SymbolDefinition] = {}
        # file_path -> set of symbol names referenced in that file
        self._file_to_symbols: dict[str, set[str]] = {}
        # file_path -> set of symbol names defined in that file
        self._file_to_definitions: dict[str, set[str]] = {}

    def add_definition(self, definition: SymbolDefinition) -> None:
        """Add a symbol definition to the index.

        Args:
            definition: Symbol definition to add.
        """
        self._definitions[definition.qualified_name] = definition

        # Track which file defines this symbol
        if definition.file_path not in self._file_to_definitions:
            self._file_to_definitions[definition.file_path] = set()
        self._file_to_definitions[definition.file_path].add(definition.qualified_name)

        logger.debug(
            "Added definition",
            symbol=definition.qualified_name,
            file=definition.file_path,
        )

    def add_reference(self, symbol_name: str, location: ReferenceLocation) -> None:
        """Add a reference to the index.

        Args:
            symbol_name: Name of the referenced symbol.
            location: Where the reference occurs.
        """
        if symbol_name not in self._symbol_refs:
            self._symbol_refs[symbol_name] = []
        self._symbol_refs[symbol_name].append(location)

        # Track which file contains references to which symbols
        if location.file_path not in self._file_to_symbols:
            self._file_to_symbols[location.file_path] = set()
        self._file_to_symbols[location.file_path].add(symbol_name)

        logger.debug(
            "Added reference",
            symbol=symbol_name,
            file=location.file_path,
            line=location.line,
        )

    def get_references(
        self,
        symbol_name: str,
        include_definition: bool = True,
        limit: int = 100,
    ) -> list[ReferenceLocation]:
        """Get all references to a symbol.

        Args:
            symbol_name: Name of the symbol (simple or qualified).
            include_definition: Whether to include the definition location.
            limit: Maximum number of references to return.

        Returns:
            List of reference locations.
        """
        refs = list(self._symbol_refs.get(symbol_name, []))

        # Include definition if requested
        if include_definition:
            defn = self.get_definition(symbol_name)
            if defn:
                def_loc = ReferenceLocation(
                    file_path=defn.file_path,
                    line=defn.line_start,
                    ref_type=ReferenceType.UNKNOWN,
                    context=f"Definition of {defn.name}",
                )
                refs = [def_loc] + refs

        return refs[:limit]

    def get_definition(self, symbol_name: str) -> SymbolDefinition | None:
        """Get definition of a symbol.

        Args:
            symbol_name: Name or qualified name of the symbol.

        Returns:
            Symbol definition or None if not found.
        """
        # Try qualified name first
        if symbol_name in self._definitions:
            return self._definitions[symbol_name]

        # Try to find by simple name
        for qname, defn in self._definitions.items():
            if defn.name == symbol_name:
                return defn

        return None

    def remove_file(self, file_path: str) -> int:
        """Remove all references from/to a file.

        Call this before re-indexing a file to avoid stale references.

        Args:
            file_path: Path of the file to remove.

        Returns:
            Number of references removed.
        """
        removed = 0

        # Remove definitions from this file
        if file_path in self._file_to_definitions:
            for qname in self._file_to_definitions[file_path]:
                if qname in self._definitions:
                    del self._definitions[qname]
                    removed += 1
            del self._file_to_definitions[file_path]

        # Remove references from this file
        if file_path in self._file_to_symbols:
            symbols_to_check = self._file_to_symbols[file_path]
            for symbol in symbols_to_check:
                if symbol in self._symbol_refs:
                    original_len = len(self._symbol_refs[symbol])
                    self._symbol_refs[symbol] = [
                        ref for ref in self._symbol_refs[symbol]
                        if ref.file_path != file_path
                    ]
                    removed += original_len - len(self._symbol_refs[symbol])
                    # Clean up empty entries
                    if not self._symbol_refs[symbol]:
                        del self._symbol_refs[symbol]
            del self._file_to_symbols[file_path]

        logger.debug("Removed file from index", file=file_path, removed=removed)
        return removed

    def clear(self) -> None:
        """Clear the entire index."""
        self._symbol_refs.clear()
        self._definitions.clear()
        self._file_to_symbols.clear()
        self._file_to_definitions.clear()
        logger.info("Reference index cleared")

    def get_stats(self) -> dict[str, int]:
        """Get index statistics.

        Returns:
            Dictionary with index stats.
        """
        return {
            "symbols": len(self._symbol_refs),
            "definitions": len(self._definitions),
            "files_with_refs": len(self._file_to_symbols),
            "files_with_defs": len(self._file_to_definitions),
            "total_refs": sum(len(refs) for refs in self._symbol_refs.values()),
        }

    def iter_all_symbols(self) -> Iterator[str]:
        """Iterate over all indexed symbols.

        Yields:
            Symbol names.
        """
        yield from self._symbol_refs.keys()

    def to_dict(self) -> dict[str, Any]:
        """Serialize index to dictionary for persistence.

        Returns:
            Dictionary representation.
        """
        return {
            "version": 1,
            "symbol_refs": {
                symbol: [ref.to_dict() for ref in refs]
                for symbol, refs in self._symbol_refs.items()
            },
            "definitions": {
                qname: defn.to_dict()
                for qname, defn in self._definitions.items()
            },
            "file_to_symbols": {
                fp: list(symbols)
                for fp, symbols in self._file_to_symbols.items()
            },
            "file_to_definitions": {
                fp: list(symbols)
                for fp, symbols in self._file_to_definitions.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReferenceIndex":
        """Deserialize index from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            ReferenceIndex instance.
        """
        index = cls()

        # Load symbol refs
        for symbol, refs in data.get("symbol_refs", {}).items():
            index._symbol_refs[symbol] = [
                ReferenceLocation.from_dict(ref) for ref in refs
            ]

        # Load definitions
        for qname, defn in data.get("definitions", {}).items():
            index._definitions[qname] = SymbolDefinition.from_dict(defn)

        # Load file mappings
        for fp, symbols in data.get("file_to_symbols", {}).items():
            index._file_to_symbols[fp] = set(symbols)

        for fp, symbols in data.get("file_to_definitions", {}).items():
            index._file_to_definitions[fp] = set(symbols)

        return index

    def save(self, path: Path) -> None:
        """Save index to file.

        Args:
            path: Path to save to.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Reference index saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> "ReferenceIndex":
        """Load index from file.

        Args:
            path: Path to load from.

        Returns:
            ReferenceIndex instance.
        """
        if not path.exists():
            logger.info("No existing reference index", path=str(path))
            return cls()

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            index = cls.from_dict(data)
            stats = index.get_stats()
            logger.info(
                "Reference index loaded",
                path=str(path),
                symbols=stats["symbols"],
                definitions=stats["definitions"],
            )
            return index
        except Exception as e:
            logger.warning(f"Failed to load reference index: {e}")
            return cls()


# Singleton instance for the application
_reference_index: ReferenceIndex | None = None


def get_reference_index() -> ReferenceIndex:
    """Get the global reference index instance.

    Returns:
        ReferenceIndex singleton.
    """
    global _reference_index
    if _reference_index is None:
        _reference_index = ReferenceIndex()
    return _reference_index


def set_reference_index(index: ReferenceIndex) -> None:
    """Set the global reference index instance.

    Args:
        index: ReferenceIndex to use.
    """
    global _reference_index
    _reference_index = index
