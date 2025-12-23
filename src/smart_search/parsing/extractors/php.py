"""PHP-specific code extractor using tree-sitter.

Extracts functions, classes, methods, and their metadata from PHP source code.
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


class PHPExtractor:
    """Extracts code units from PHP source code."""

    def extract_units(
        self,
        tree: Tree,
        source: bytes,
        file_path: Path,
    ) -> list[CodeUnit]:
        """Extract all code units from a PHP AST.

        Args:
            tree: The parsed AST tree.
            source: The source code bytes.
            file_path: Path to the source file.

        Returns:
            List of extracted CodeUnits.
        """
        units: list[CodeUnit] = []
        module_name = file_path.stem

        # Extract definitions from root
        self._extract_from_node(
            node=tree.root_node,
            source=source,
            file_path=file_path,
            namespace="",
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
        namespace: str,
        parent_id: str | None,
        parent_qualified_name: str,
        units: list[CodeUnit],
    ) -> None:
        """Recursively extract code units from a node.

        Args:
            node: Current AST node.
            source: Source code bytes.
            file_path: Path to source file.
            namespace: Current PHP namespace.
            parent_id: ID of parent unit (if any).
            parent_qualified_name: Qualified name of parent.
            units: List to append extracted units to.
        """
        for child in node.children:
            # Handle namespace declarations
            if child.type == "namespace_definition":
                ns_name = self._extract_namespace_name(child, source)
                if ns_name:
                    namespace = ns_name
                # Continue extracting from namespace body
                for sub_child in child.children:
                    if sub_child.type == "compound_statement":
                        self._extract_from_node(
                            node=sub_child,
                            source=source,
                            file_path=file_path,
                            namespace=namespace,
                            parent_id=parent_id,
                            parent_qualified_name=namespace or parent_qualified_name,
                            units=units,
                        )

            # Handle function declarations
            elif child.type == "function_definition":
                unit = self._extract_function(
                    child, source, file_path, namespace, parent_id, parent_qualified_name
                )
                if unit:
                    units.append(unit)

            # Handle class declarations
            elif child.type == "class_declaration":
                class_unit = self._extract_class(
                    child, source, file_path, namespace, parent_id, parent_qualified_name
                )
                if class_unit:
                    units.append(class_unit)
                    # Extract methods from class body
                    body = self._find_child_by_type(child, "declaration_list")
                    if body:
                        self._extract_class_members(
                            node=body,
                            source=source,
                            file_path=file_path,
                            namespace=namespace,
                            parent_id=class_unit.id,
                            parent_qualified_name=class_unit.qualified_name,
                            units=units,
                        )

            # Handle interface declarations
            elif child.type == "interface_declaration":
                interface_unit = self._extract_interface(
                    child, source, file_path, namespace, parent_id, parent_qualified_name
                )
                if interface_unit:
                    units.append(interface_unit)

            # Handle trait declarations
            elif child.type == "trait_declaration":
                trait_unit = self._extract_trait(
                    child, source, file_path, namespace, parent_id, parent_qualified_name
                )
                if trait_unit:
                    units.append(trait_unit)
                    # Extract methods from trait body
                    body = self._find_child_by_type(child, "declaration_list")
                    if body:
                        self._extract_class_members(
                            node=body,
                            source=source,
                            file_path=file_path,
                            namespace=namespace,
                            parent_id=trait_unit.id,
                            parent_qualified_name=trait_unit.qualified_name,
                            units=units,
                        )

            # Handle program/compound statement (recurse)
            elif child.type in ("program", "compound_statement"):
                self._extract_from_node(
                    node=child,
                    source=source,
                    file_path=file_path,
                    namespace=namespace,
                    parent_id=parent_id,
                    parent_qualified_name=parent_qualified_name,
                    units=units,
                )

    def _extract_class_members(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str,
        parent_qualified_name: str,
        units: list[CodeUnit],
    ) -> None:
        """Extract methods and properties from a class body."""
        for child in node.children:
            if child.type == "method_declaration":
                unit = self._extract_method(
                    child, source, file_path, namespace, parent_id, parent_qualified_name
                )
                if unit:
                    units.append(unit)

            elif child.type == "property_declaration":
                props = self._extract_properties(
                    child, source, file_path, namespace, parent_id, parent_qualified_name
                )
                units.extend(props)

    def _extract_function(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str | None,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract a function definition."""
        # Get function name
        name_node = self._find_child_by_type(node, "name")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        # Build qualified name
        if namespace:
            qualified_name = f"{namespace}\\{name}"
        else:
            qualified_name = f"{parent_qualified_name}.{name}"
        unit_id = f"{file_path}::{qualified_name}"

        # Get content
        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Extract parameters
        params = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Extract docstring (PHP DocBlock)
        docstring = self._extract_docblock(node, source)

        # Build signature
        signature = self._build_signature(name, params, return_type)

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=CodeUnitType.FUNCTION,
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PHP,
            content=content,
            docstring=docstring,
            signature=signature,
            parameters=params,
            return_type=return_type,
            parent_id=parent_id,
            content_hash=content_hash,
        )

    def _extract_method(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract a method definition."""
        # Get method name
        name_node = self._find_child_by_type(node, "name")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        # Build qualified name
        qualified_name = f"{parent_qualified_name}::{name}"
        unit_id = f"{file_path}::{qualified_name}"

        # Get content
        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Extract visibility
        visibility = self._extract_visibility(node, source)

        # Check if static
        is_static = self._is_static(node, source)

        # Extract parameters
        params = self._extract_parameters(node, source)

        # Extract return type
        return_type = self._extract_return_type(node, source)

        # Extract docstring
        docstring = self._extract_docblock(node, source)

        # Build signature
        prefix = f"{visibility} " if visibility else ""
        if is_static:
            prefix += "static "
        signature = f"{prefix}function {name}({self._format_params(params)})"
        if return_type:
            signature += f": {return_type}"

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=CodeUnitType.METHOD,
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PHP,
            content=content,
            docstring=docstring,
            signature=signature,
            parameters=params,
            return_type=return_type,
            parent_id=parent_id,
            content_hash=content_hash,
        )

    def _extract_class(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str | None,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract a class definition."""
        # Get class name
        name_node = self._find_child_by_type(node, "name")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        # Build qualified name
        if namespace:
            qualified_name = f"{namespace}\\{name}"
        else:
            qualified_name = f"{parent_qualified_name}.{name}"
        unit_id = f"{file_path}::{qualified_name}"

        # Get content
        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Extract base class
        base_classes = self._extract_base_class(node, source)

        # Extract interfaces
        interfaces = self._extract_interfaces(node, source)

        # Extract docstring
        docstring = self._extract_docblock(node, source)

        # Build signature
        signature = f"class {name}"
        if base_classes:
            signature += f" extends {', '.join(base_classes)}"
        if interfaces:
            signature += f" implements {', '.join(interfaces)}"

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=CodeUnitType.CLASS,
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PHP,
            content=content,
            docstring=docstring,
            signature=signature,
            parent_id=parent_id,
            content_hash=content_hash,
            base_classes=base_classes,
        )

    def _extract_interface(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str | None,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract an interface definition."""
        name_node = self._find_child_by_type(node, "name")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        if namespace:
            qualified_name = f"{namespace}\\{name}"
        else:
            qualified_name = f"{parent_qualified_name}.{name}"
        unit_id = f"{file_path}::{qualified_name}"

        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        docstring = self._extract_docblock(node, source)

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=CodeUnitType.CLASS,  # Treat interface as class type
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PHP,
            content=content,
            docstring=docstring,
            signature=f"interface {name}",
            parent_id=parent_id,
            content_hash=content_hash,
        )

    def _extract_trait(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str | None,
        parent_qualified_name: str,
    ) -> CodeUnit | None:
        """Extract a trait definition."""
        name_node = self._find_child_by_type(node, "name")
        if not name_node:
            return None
        name = get_node_text(name_node, source)

        if namespace:
            qualified_name = f"{namespace}\\{name}"
        else:
            qualified_name = f"{parent_qualified_name}.{name}"
        unit_id = f"{file_path}::{qualified_name}"

        content = get_node_text(node, source)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        docstring = self._extract_docblock(node, source)

        return CodeUnit(
            id=unit_id,
            name=name,
            qualified_name=qualified_name,
            type=CodeUnitType.CLASS,  # Treat trait as class type
            file_path=file_path,
            span=create_span_from_node(node),
            language=Language.PHP,
            content=content,
            docstring=docstring,
            signature=f"trait {name}",
            parent_id=parent_id,
            content_hash=content_hash,
        )

    def _extract_properties(
        self,
        node: Node,
        source: bytes,
        file_path: Path,
        namespace: str,
        parent_id: str,
        parent_qualified_name: str,
    ) -> list[CodeUnit]:
        """Extract property declarations."""
        units = []
        visibility = self._extract_visibility(node, source)

        for child in node.children:
            if child.type == "property_element":
                var_node = self._find_child_by_type(child, "variable_name")
                if var_node:
                    name = get_node_text(var_node, source).lstrip("$")
                    qualified_name = f"{parent_qualified_name}::${name}"
                    unit_id = f"{file_path}::{qualified_name}"

                    content = get_node_text(node, source)
                    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

                    units.append(CodeUnit(
                        id=unit_id,
                        name=f"${name}",
                        qualified_name=qualified_name,
                        type=CodeUnitType.PROPERTY,
                        file_path=file_path,
                        span=create_span_from_node(node),
                        language=Language.PHP,
                        content=content,
                        signature=f"{visibility} ${name}" if visibility else f"${name}",
                        parent_id=parent_id,
                        content_hash=content_hash,
                    ))
        return units

    def _extract_namespace_name(self, node: Node, source: bytes) -> str | None:
        """Extract namespace name from namespace definition."""
        name_node = self._find_child_by_type(node, "namespace_name")
        if name_node:
            return get_node_text(name_node, source)
        return None

    def _extract_parameters(self, node: Node, source: bytes) -> list[Parameter]:
        """Extract function/method parameters."""
        params = []
        params_node = self._find_child_by_type(node, "formal_parameters")
        if not params_node:
            return params

        for child in params_node.children:
            if child.type in ("simple_parameter", "property_promotion_parameter"):
                param = self._extract_single_parameter(child, source)
                if param:
                    params.append(param)
        return params

    def _extract_single_parameter(self, node: Node, source: bytes) -> Parameter | None:
        """Extract a single parameter."""
        var_node = self._find_child_by_type(node, "variable_name")
        if not var_node:
            return None

        name = get_node_text(var_node, source).lstrip("$")

        # Get type hint
        type_hint = None
        type_node = self._find_child_by_type(node, "type_list") or \
                    self._find_child_by_type(node, "named_type") or \
                    self._find_child_by_type(node, "primitive_type")
        if type_node:
            type_hint = get_node_text(type_node, source)

        # Get default value
        default = None
        for child in node.children:
            if child.type == "=":
                # Next sibling is the default value
                idx = node.children.index(child)
                if idx + 1 < len(node.children):
                    default = get_node_text(node.children[idx + 1], source)
                break

        return Parameter(
            name=name,
            type_annotation=type_hint,
            default_value=default,
        )

    def _extract_return_type(self, node: Node, source: bytes) -> str | None:
        """Extract return type annotation."""
        for child in node.children:
            if child.type == "return_type":
                # Get the actual type from return_type
                for sub in child.children:
                    if sub.type in ("named_type", "primitive_type", "type_list"):
                        return get_node_text(sub, source)
        return None

    def _extract_visibility(self, node: Node, source: bytes) -> str | None:
        """Extract visibility modifier (public, private, protected)."""
        for child in node.children:
            if child.type == "visibility_modifier":
                return get_node_text(child, source)
        return None

    def _is_static(self, node: Node, source: bytes) -> bool:
        """Check if method is static."""
        for child in node.children:
            if child.type == "static_modifier":
                return True
        return False

    def _extract_docblock(self, node: Node, source: bytes) -> str | None:
        """Extract PHPDoc comment before the node."""
        # In tree-sitter, comments may be siblings before the node
        if node.prev_sibling and node.prev_sibling.type == "comment":
            comment = get_node_text(node.prev_sibling, source)
            if comment.startswith("/**"):
                # Clean up docblock
                lines = comment.split("\n")
                cleaned = []
                for line in lines:
                    line = line.strip()
                    if line.startswith("/**"):
                        line = line[3:]
                    if line.endswith("*/"):
                        line = line[:-2]
                    if line.startswith("*"):
                        line = line[1:]
                    line = line.strip()
                    if line:
                        cleaned.append(line)
                return "\n".join(cleaned)
        return None

    def _extract_base_class(self, node: Node, source: bytes) -> list[str]:
        """Extract base class from extends clause."""
        for child in node.children:
            if child.type == "base_clause":
                name_node = self._find_child_by_type(child, "name") or \
                           self._find_child_by_type(child, "qualified_name")
                if name_node:
                    return [get_node_text(name_node, source)]
        return []

    def _extract_interfaces(self, node: Node, source: bytes) -> list[str]:
        """Extract implemented interfaces."""
        interfaces = []
        for child in node.children:
            if child.type == "class_interface_clause":
                for name_child in child.children:
                    if name_child.type in ("name", "qualified_name"):
                        interfaces.append(get_node_text(name_child, source))
        return interfaces

    def _build_signature(
        self,
        name: str,
        params: list[Parameter],
        return_type: str | None,
    ) -> str:
        """Build function signature string."""
        signature = f"function {name}({self._format_params(params)})"
        if return_type:
            signature += f": {return_type}"
        return signature

    def _format_params(self, params: list[Parameter]) -> str:
        """Format parameters for signature."""
        parts = []
        for p in params:
            part = ""
            if p.type_annotation:
                part += f"{p.type_annotation} "
            part += f"${p.name}"
            if p.default_value:
                part += f" = {p.default_value}"
            parts.append(part)
        return ", ".join(parts)

    def _find_child_by_type(self, node: Node, type_name: str) -> Node | None:
        """Find first child with given type."""
        for child in node.children:
            if child.type == type_name:
                return child
        return None
