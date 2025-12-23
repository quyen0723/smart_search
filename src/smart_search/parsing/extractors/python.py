"""Python-specific code extractor using tree-sitter.

Extracts functions, classes, methods, and their metadata from Python source code.
"""

import hashlib
from pathlib import Path

from tree_sitter import Node, Tree

from smart_search.parsing.models import (
    CodeUnit,
    CodeUnitType,
    Language,
    Parameter,
    Position,
    Span,
)
from smart_search.parsing.tree_sitter_parser import create_span_from_node, get_node_text
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class PythonExtractor:
    """Extracts code units from Python source code."""

    def extract_units(
        self,
        tree: Tree,
        source: bytes,
        file_path: Path,
    ) -> list[CodeUnit]:
        """Extract all code units from a Python AST.

        Args:
            tree: The parsed AST tree.
            source: The source code bytes.
            file_path: Path to the source file.

        Returns:
            List of extracted CodeUnits.
        """
        units: list[CodeUnit] = []
        module_name = file_path.stem

        # Extract top-level definitions
        self._extract_from_node(
            node=tree.root_node,
            source=source,
            file_path=file_path,
            module_name=module_name,
            parent_id=None,
            parent_qualified_name=module_name,
            units=units,
        )

        return units

    def _extract_from_node(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        module_name: str,
        parent_id: str | None,
        parent_qualified_name: str,
        units: list[CodeUnit],
    ) -> None:
        """Recursively extract code units from a node.

        Args:
            node: Current AST node.
            source: Source code bytes.
            file_path: Path to source file.
            module_name: Name of the module.
            parent_id: ID of parent unit (if any).
            parent_qualified_name: Qualified name of parent.
            units: List to append extracted units to.
        """
        for child in node.children:
            if child.type == "function_definition":
                unit = self._extract_function(
                    child, source, file_path, parent_id, parent_qualified_name
                )
                if unit:
                    units.append(unit)

            elif child.type == "class_definition":
                class_unit = self._extract_class(
                    child, source, file_path, parent_id, parent_qualified_name
                )
                if class_unit:
                    units.append(class_unit)
                    # Extract methods from class body
                    class_body = self._find_child_by_type(child, "block")
                    if class_body:
                        self._extract_from_node(
                            node=class_body,
                            source=source,
                            file_path=file_path,
                            module_name=module_name,
                            parent_id=class_unit.id,
                            parent_qualified_name=class_unit.qualified_name,
                            units=units,
                        )

            elif child.type == "decorated_definition":
                # Handle decorated functions and classes
                decorators = self._extract_decorators(child, source)
                definition = self._find_child_by_type(child, "function_definition")
                if definition:
                    unit = self._extract_function(
                        definition, source, file_path, parent_id, parent_qualified_name
                    )
                    if unit:
                        unit.decorators = decorators
                        units.append(unit)
                else:
                    definition = self._find_child_by_type(child, "class_definition")
                    if definition:
                        class_unit = self._extract_class(
                            definition, source, file_path, parent_id, parent_qualified_name
                        )
                        if class_unit:
                            class_unit.decorators = decorators
                            units.append(class_unit)
                            # Extract methods
                            class_body = self._find_child_by_type(definition, "block")
                            if class_body:
                                self._extract_from_node(
                                    node=class_body,
                                    source=source,
                                    file_path=file_path,
                                    module_name=module_name,
                                    parent_id=class_unit.id,
                                    parent_qualified_name=class_unit.qualified_name,
                                    units=units,
                                )

    def _extract_function(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        parent_id: str | None,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract a function or method definition.

        Args:
            node: The function_definition node.
            source: Source code bytes.
            file_path: Path to source file.
            parent_id: ID of parent class (if method).
            parent_qualified_name: Qualified name of parent.

        Returns:
            CodeUnit for the function/method.
        """
        # Get function name
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        # Determine if method or function
        unit_type = CodeUnitType.METHOD if parent_id else CodeUnitType.FUNCTION

        # Check for property decorator (would need to look at parent decorated_definition)
        # For now, we'll handle this at a higher level

        # Build qualified name and ID
        qualified_name = f"{parent_qualified_name}.{name}"
        unit_id = f"{file_path}::{qualified_name}"

        # Get full content
        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Extract parameters
        params = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        # Build signature
        signature = self._build_signature(name, params, return_type)

        # Extract calls (function calls within this function)
        calls = self._extract_calls(node, source)

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=unit_type,
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PYTHON,
            content=content,
            docstring=docstring,
            signature=signature,
            parameters=params,
            return_type=return_type,
            parent_id=parent_id,
            content_hash=content_hash,
            calls=calls,
        )

    def _extract_class(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        parent_id: str | None,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract a class definition.

        Args:
            node: The class_definition node.
            source: Source code bytes.
            file_path: Path to source file.
            parent_id: ID of parent (if nested class).
            parent_qualified_name: Qualified name of parent.

        Returns:
            CodeUnit for the class.
        """
        # Get class name
        name_node = self._find_child_by_type(node, "identifier")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        # Build qualified name and ID
        qualified_name = f"{parent_qualified_name}.{name}"
        unit_id = f"{file_path}::{qualified_name}"

        # Get full content
        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Extract base classes
        base_classes = self._extract_base_classes(node, source)

        # Extract docstring
        docstring = self._extract_docstring(node, source)

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=CodeUnitType.CLASS,
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PYTHON,
            content=content,
            docstring=docstring,
            base_classes=base_classes,
            parent_id=parent_id,
            content_hash=content_hash,
        )

    def _extract_parameters(self, func_node: Node, source: bytes) -> list[Parameter]:
        """Extract function parameters.

        Args:
            func_node: The function_definition node.
            source: Source code bytes.

        Returns:
            List of Parameter objects.
        """
        params: list[Parameter] = []
        params_node = self._find_child_by_type(func_node, "parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type == "identifier":
                # Simple parameter
                params.append(Parameter(name=get_node_text(child, source)))

            elif child.type == "typed_parameter":
                # Parameter with type annotation
                name = ""
                type_ann = None
                for sub in child.children:
                    if sub.type == "identifier":
                        name = get_node_text(sub, source)
                    elif sub.type == "type":
                        type_ann = get_node_text(sub, source)
                if name:
                    params.append(Parameter(name=name, type_annotation=type_ann))

            elif child.type == "default_parameter":
                # Parameter with default value (no type annotation)
                name = ""
                default = None
                for sub in child.children:
                    if sub.type == "identifier":
                        name = get_node_text(sub, source)
                    elif sub.type not in ("=", ":"):
                        default = get_node_text(sub, source)
                if name:
                    params.append(
                        Parameter(name=name, default_value=default)
                    )

            elif child.type == "typed_default_parameter":
                # Parameter with both type annotation and default value (b: int = 0)
                name = ""
                default = None
                type_ann = None
                for sub in child.children:
                    if sub.type == "identifier":
                        name = get_node_text(sub, source)
                    elif sub.type == "type":
                        type_ann = get_node_text(sub, source)
                    elif sub.type not in ("=", ":", "identifier", "type"):
                        default = get_node_text(sub, source)
                if name:
                    params.append(
                        Parameter(name=name, type_annotation=type_ann, default_value=default)
                    )

            elif child.type == "list_splat_pattern":
                # *args
                for sub in child.children:
                    if sub.type == "identifier":
                        params.append(
                            Parameter(name=get_node_text(sub, source), is_variadic=True)
                        )

            elif child.type == "dictionary_splat_pattern":
                # **kwargs
                for sub in child.children:
                    if sub.type == "identifier":
                        params.append(
                            Parameter(name=get_node_text(sub, source), is_keyword=True)
                        )

        return params

    def _extract_return_type(self, func_node: Node, source: bytes) -> str | None:
        """Extract function return type annotation.

        Args:
            func_node: The function_definition node.
            source: Source code bytes.

        Returns:
            Return type as string, or None.
        """
        for child in func_node.children:
            if child.type == "type":
                return get_node_text(child, source)
        return None

    def _extract_docstring(self, node: Node, source: bytes) -> str | None:
        """Extract docstring from a function or class.

        Args:
            node: The function_definition or class_definition node.
            source: Source code bytes.

        Returns:
            Docstring if present.
        """
        body = self._find_child_by_type(node, "block")
        if not body or not body.children:
            return None

        for child in body.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type == "string":
                        docstring = get_node_text(sub, source)
                        return self._clean_docstring(docstring)
            elif child.type not in ("comment", "pass_statement"):
                # First non-comment, non-pass statement ends docstring search
                break

        return None

    def _extract_base_classes(self, class_node: Node, source: bytes) -> list[str]:
        """Extract base classes from a class definition.

        Args:
            class_node: The class_definition node.
            source: Source code bytes.

        Returns:
            List of base class names.
        """
        bases: list[str] = []
        arg_list = self._find_child_by_type(class_node, "argument_list")
        if not arg_list:
            return bases

        for child in arg_list.children:
            if child.type == "identifier":
                bases.append(get_node_text(child, source))
            elif child.type == "attribute":
                bases.append(get_node_text(child, source))

        return bases

    def _extract_decorators(self, decorated_node: Node, source: bytes) -> list[str]:
        """Extract decorator names from a decorated definition.

        Args:
            decorated_node: The decorated_definition node.
            source: Source code bytes.

        Returns:
            List of decorator names.
        """
        decorators: list[str] = []
        for child in decorated_node.children:
            if child.type == "decorator":
                # Get the decorator name (excluding @)
                for sub in child.children:
                    if sub.type == "identifier":
                        decorators.append(get_node_text(sub, source))
                    elif sub.type == "attribute":
                        decorators.append(get_node_text(sub, source))
                    elif sub.type == "call":
                        # Decorator with arguments like @decorator(arg)
                        for sub2 in sub.children:
                            if sub2.type == "identifier":
                                decorators.append(get_node_text(sub2, source))
                                break
                            elif sub2.type == "attribute":
                                decorators.append(get_node_text(sub2, source))
                                break
        return decorators

    def _extract_calls(self, node: Node, source: bytes) -> list[str]:
        """Extract function/method calls from a node.

        Args:
            node: The AST node to search.
            source: Source code bytes.

        Returns:
            List of called function/method names.
        """
        calls: list[str] = []

        def visit(n: Node) -> None:
            if n.type == "call":
                for child in n.children:
                    if child.type == "identifier":
                        calls.append(get_node_text(child, source))
                    elif child.type == "attribute":
                        calls.append(get_node_text(child, source))
                    break  # Only get the function name, not arguments
            for child in n.children:
                visit(child)

        visit(node)
        return list(set(calls))  # Deduplicate

    def _build_signature(
        self,
        name: str,
        params: list[Parameter],
        return_type: str | None,
    ) -> str:
        """Build a function signature string.

        Args:
            name: Function name.
            params: List of parameters.
            return_type: Return type annotation.

        Returns:
            Signature string like "func(a, b: int) -> str"
        """
        param_strs = []
        for p in params:
            s = p.name
            if p.is_variadic:
                s = f"*{s}"
            elif p.is_keyword:
                s = f"**{s}"
            if p.type_annotation:
                s = f"{s}: {p.type_annotation}"
            if p.default_value:
                s = f"{s} = {p.default_value}"
            param_strs.append(s)

        sig = f"{name}({', '.join(param_strs)})"
        if return_type:
            sig = f"{sig} -> {return_type}"
        return sig

    def _find_child_by_type(self, node: Node, type_name: str) -> Node | None:
        """Find first child node of given type.

        Args:
            node: Parent node.
            type_name: Type name to find.

        Returns:
            First matching child or None.
        """
        for child in node.children:
            if child.type == type_name:
                return child
        return None

    def _clean_docstring(self, docstring: str) -> str:
        """Clean a docstring by removing quotes.

        Args:
            docstring: Raw docstring with quotes.

        Returns:
            Cleaned docstring.
        """
        if docstring.startswith('"""') or docstring.startswith("'''"):
            docstring = docstring[3:-3]
        elif docstring.startswith('"') or docstring.startswith("'"):
            docstring = docstring[1:-1]
        return docstring.strip()
