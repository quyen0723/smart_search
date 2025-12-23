"""Tree-sitter based parser for multi-language AST parsing.

This module provides the core parsing functionality using tree-sitter
to generate ASTs from source code files.
"""

import hashlib
import time
from pathlib import Path
from typing import Protocol

import tree_sitter_python as ts_python
import tree_sitter_php as ts_php
from tree_sitter import Language, Parser, Tree, Node

from smart_search.core.exceptions import SyntaxParseError, UnsupportedLanguageError
from smart_search.parsing.models import (
    CodeUnit,
    CodeUnitType,
    Language as CodeLanguage,
    ParsedFile,
    Position,
    Span,
)
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class LanguageExtractor(Protocol):
    """Protocol for language-specific code extractors."""

    def extract_units(
        self,
        tree: Tree,
        source: bytes,
        file_path: Path,
    ) -> list[CodeUnit]:
        """Extract code units from a parsed AST tree."""
        ...


class TreeSitterParser:
    """Multi-language parser using tree-sitter.

    Provides unified interface for parsing source code into ASTs
    and extracting code units.
    """

    # Mapping of languages to their tree-sitter implementations
    _LANGUAGE_MODULES = {
        CodeLanguage.PYTHON: ts_python,
        CodeLanguage.PHP: ts_php,
    }

    def __init__(self) -> None:
        """Initialize the parser with language support."""
        self._parsers: dict[CodeLanguage, Parser] = {}
        self._extractors: dict[CodeLanguage, LanguageExtractor] = {}
        self._initialize_parsers()

    def _initialize_parsers(self) -> None:
        """Initialize tree-sitter parsers for supported languages."""
        for lang, module in self._LANGUAGE_MODULES.items():
            try:
                # Different tree-sitter modules use different function names
                if hasattr(module, 'language'):
                    lang_func = module.language
                elif hasattr(module, 'language_php'):
                    lang_func = module.language_php
                else:
                    logger.warning(
                        "No language function found",
                        language=lang.value,
                    )
                    continue

                parser = Parser(Language(lang_func()))
                self._parsers[lang] = parser
                logger.debug("Initialized parser", language=lang.value)
            except Exception as e:
                logger.warning(
                    "Failed to initialize parser",
                    language=lang.value,
                    error=str(e),
                )

    def register_extractor(
        self,
        language: CodeLanguage,
        extractor: LanguageExtractor,
    ) -> None:
        """Register a language-specific extractor.

        Args:
            language: The programming language.
            extractor: The extractor implementation.
        """
        self._extractors[language] = extractor
        logger.debug("Registered extractor", language=language.value)

    @property
    def supported_languages(self) -> list[CodeLanguage]:
        """Get list of supported languages."""
        return list(self._parsers.keys())

    def is_supported(self, language: CodeLanguage) -> bool:
        """Check if a language is supported.

        Args:
            language: The language to check.

        Returns:
            True if the language is supported.
        """
        return language in self._parsers

    def parse_source(
        self,
        source: str | bytes,
        language: CodeLanguage,
    ) -> Tree:
        """Parse source code into an AST tree.

        Args:
            source: The source code to parse.
            language: The programming language.

        Returns:
            The parsed AST tree.

        Raises:
            UnsupportedLanguageError: If language is not supported.
            SyntaxParseError: If parsing fails.
        """
        if language not in self._parsers:
            raise UnsupportedLanguageError(
                language.value,
                [l.value for l in self.supported_languages],
            )

        parser = self._parsers[language]

        if isinstance(source, str):
            source = source.encode("utf-8")

        try:
            tree = parser.parse(source)
            return tree
        except Exception as e:
            raise SyntaxParseError(
                file_path="<string>",
                cause=e,
            )

    def parse_file(self, file_path: Path) -> ParsedFile:
        """Parse a source file and extract code units.

        Args:
            file_path: Path to the source file.

        Returns:
            ParsedFile with extracted code units.

        Raises:
            UnsupportedLanguageError: If file type is not supported.
            SyntaxParseError: If parsing fails.
            FileNotFoundError: If file does not exist.
        """
        start_time = time.perf_counter()

        # Detect language from extension
        language = CodeLanguage.from_extension(file_path.suffix)
        if language is None:
            raise UnsupportedLanguageError(
                file_path.suffix,
                [l.value for l in self.supported_languages],
            )

        if not self.is_supported(language):
            raise UnsupportedLanguageError(
                language.value,
                [l.value for l in self.supported_languages],
            )

        # Read file content
        content = file_path.read_text(encoding="utf-8")
        content_bytes = content.encode("utf-8")
        content_hash = hashlib.sha256(content_bytes).hexdigest()[:16]

        # Parse into AST
        parse_errors: list[str] = []
        try:
            tree = self.parse_source(content_bytes, language)
        except SyntaxParseError as e:
            parse_errors.append(str(e))
            # Return partial result with errors
            return ParsedFile(
                file_path=file_path,
                language=language,
                content=content,
                content_hash=content_hash,
                parse_errors=parse_errors,
                parse_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Check for syntax errors in tree
        if tree.root_node.has_error:
            errors = self._collect_errors(tree.root_node)
            parse_errors.extend(errors)

        # Extract code units using language-specific extractor
        units: list[CodeUnit] = []
        imports: list[str] = []
        module_docstring: str | None = None

        if language in self._extractors:
            extractor = self._extractors[language]
            units = extractor.extract_units(tree, content_bytes, file_path)

            # Extract module-level imports and docstring
            imports = self._extract_imports(tree.root_node, content_bytes, language)
            module_docstring = self._extract_module_docstring(
                tree.root_node, content_bytes, language
            )

        parse_time = (time.perf_counter() - start_time) * 1000

        logger.debug(
            "Parsed file",
            file=str(file_path),
            language=language.value,
            units=len(units),
            errors=len(parse_errors),
            time_ms=f"{parse_time:.2f}",
        )

        return ParsedFile(
            file_path=file_path,
            language=language,
            content=content,
            content_hash=content_hash,
            units=units,
            imports=imports,
            module_docstring=module_docstring,
            parse_errors=parse_errors,
            parse_time_ms=parse_time,
        )

    def _collect_errors(self, node: Node, max_errors: int = 10) -> list[str]:
        """Collect syntax errors from AST.

        Args:
            node: The root node to search.
            max_errors: Maximum number of errors to collect.

        Returns:
            List of error descriptions.
        """
        errors: list[str] = []

        def visit(n: Node) -> None:
            if len(errors) >= max_errors:
                return
            if n.is_error or n.is_missing:
                error_type = "Missing" if n.is_missing else "Error"
                errors.append(
                    f"{error_type} at line {n.start_point[0] + 1}, "
                    f"column {n.start_point[1]}: {n.type}"
                )
            for child in n.children:
                visit(child)

        visit(node)
        return errors

    def _extract_imports(
        self,
        root: Node,
        source: bytes,
        language: CodeLanguage,
    ) -> list[str]:
        """Extract import statements from module.

        Args:
            root: The root AST node.
            source: The source bytes.
            language: The programming language.

        Returns:
            List of imported module names.
        """
        imports: list[str] = []

        if language == CodeLanguage.PYTHON:
            for node in root.children:
                if node.type == "import_statement":
                    # import foo, bar
                    for child in node.children:
                        if child.type == "dotted_name":
                            imports.append(self._get_node_text(child, source))
                elif node.type == "import_from_statement":
                    # from foo import bar
                    for child in node.children:
                        if child.type == "dotted_name":
                            imports.append(self._get_node_text(child, source))
                            break

        return imports

    def _extract_module_docstring(
        self,
        root: Node,
        source: bytes,
        language: CodeLanguage,
    ) -> str | None:
        """Extract module-level docstring.

        Args:
            root: The root AST node.
            source: The source bytes.
            language: The programming language.

        Returns:
            The module docstring if present.
        """
        if language == CodeLanguage.PYTHON:
            for node in root.children:
                if node.type == "expression_statement":
                    for child in node.children:
                        if child.type == "string":
                            docstring = self._get_node_text(child, source)
                            # Remove quotes
                            return self._clean_docstring(docstring)
                elif node.type not in ("comment", "import_statement", "import_from_statement"):
                    # First non-import, non-comment statement is not a docstring
                    break
        return None

    def _get_node_text(self, node: Node, source: bytes) -> str:
        """Get the text content of a node.

        Args:
            node: The AST node.
            source: The source bytes.

        Returns:
            The text content.
        """
        return source[node.start_byte:node.end_byte].decode("utf-8")

    def _clean_docstring(self, docstring: str) -> str:
        """Clean a docstring by removing quotes and extra whitespace.

        Args:
            docstring: The raw docstring.

        Returns:
            Cleaned docstring.
        """
        # Remove triple quotes
        if docstring.startswith('"""') or docstring.startswith("'''"):
            docstring = docstring[3:-3]
        elif docstring.startswith('"') or docstring.startswith("'"):
            docstring = docstring[1:-1]
        return docstring.strip()


def create_span_from_node(node: Node) -> Span:
    """Create a Span from a tree-sitter node.

    Args:
        node: The tree-sitter node.

    Returns:
        A Span representing the node's location.
    """
    return Span(
        start=Position(
            line=node.start_point[0] + 1,  # Convert to 1-indexed
            column=node.start_point[1],
        ),
        end=Position(
            line=node.end_point[0] + 1,
            column=node.end_point[1],
        ),
    )


def get_node_text(node: Node, source: bytes) -> str:
    """Get the text content of a node.

    Args:
        node: The AST node.
        source: The source bytes.

    Returns:
        The text content.
    """
    return source[node.start_byte:node.end_byte].decode("utf-8")
