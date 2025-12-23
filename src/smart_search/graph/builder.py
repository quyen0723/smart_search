"""Graph builder for constructing code graphs from parsed units.

Builds the code dependency graph from CodeUnits extracted by parsers.
"""

from pathlib import Path

from smart_search.graph.engine import CodeGraph
from smart_search.graph.models import EdgeData, EdgeType, NodeData, NodeType
from smart_search.parsing.models import CodeUnit, CodeUnitType, ParsedFile
from smart_search.utils.logging import get_logger

logger = get_logger(__name__)


class GraphBuilder:
    """Builds a CodeGraph from parsed code units.

    Handles the conversion of CodeUnits to graph nodes and
    establishes relationships (edges) between them.
    """

    def __init__(self, graph: CodeGraph | None = None) -> None:
        """Initialize the builder.

        Args:
            graph: Existing graph to build on, or None for new graph.
        """
        self.graph = graph or CodeGraph()

    def build_from_units(self, units: list[CodeUnit]) -> CodeGraph:
        """Build graph from a list of code units.

        Args:
            units: List of CodeUnits to add to the graph.

        Returns:
            The built CodeGraph.
        """
        # First pass: add all nodes
        for unit in units:
            self._add_unit_node(unit)

        # Second pass: add edges
        for unit in units:
            self._add_unit_edges(unit, units)

        logger.info(
            "Built graph from units",
            unit_count=len(units),
            node_count=self.graph.node_count,
            edge_count=self.graph.edge_count,
        )
        return self.graph

    def build_from_parsed_files(self, parsed_files: list[ParsedFile]) -> CodeGraph:
        """Build graph from parsed files.

        Args:
            parsed_files: List of ParsedFile objects.

        Returns:
            The built CodeGraph.
        """
        # First add module nodes
        for pf in parsed_files:
            self._add_module_node(pf)

        # Collect all units
        all_units: list[CodeUnit] = []
        for pf in parsed_files:
            all_units.extend(pf.units)

        # Build from units
        self.build_from_units(all_units)

        # Add import edges between modules
        for pf in parsed_files:
            self._add_import_edges(pf)

        return self.graph

    def add_unit(self, unit: CodeUnit) -> None:
        """Add a single unit to the graph.

        Args:
            unit: The CodeUnit to add.
        """
        self._add_unit_node(unit)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
    ) -> bool:
        """Add an edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Type of relationship.
            weight: Edge weight.

        Returns:
            True if edge was added.
        """
        return self.graph.add_edge(
            source_id,
            target_id,
            EdgeData(edge_type=edge_type, weight=weight),
        )

    def _add_unit_node(self, unit: CodeUnit) -> None:
        """Add a CodeUnit as a graph node.

        Args:
            unit: The CodeUnit to add.
        """
        node_type = self._map_unit_type(unit.type)

        node_data = NodeData(
            id=unit.id,
            name=unit.name,
            qualified_name=unit.qualified_name,
            node_type=node_type,
            file_path=unit.file_path,
            line_start=unit.span.start.line,
            line_end=unit.span.end.line,
            metadata={
                "signature": unit.signature,
                "docstring": unit.docstring,
                "decorators": unit.decorators,
                "base_classes": unit.base_classes,
            },
        )

        self.graph.add_node_if_not_exists(node_data)

    def _add_module_node(self, parsed_file: ParsedFile) -> None:
        """Add a module node for a parsed file.

        Args:
            parsed_file: The ParsedFile to add as module.
        """
        module_id = f"{parsed_file.file_path}::module"

        node_data = NodeData(
            id=module_id,
            name=parsed_file.file_path.stem,
            qualified_name=str(parsed_file.file_path),
            node_type=NodeType.MODULE,
            file_path=parsed_file.file_path,
            line_start=1,
            line_end=None,
            metadata={
                "docstring": parsed_file.module_docstring,
                "imports": parsed_file.imports,
            },
        )

        self.graph.add_node_if_not_exists(node_data)

    def _add_unit_edges(self, unit: CodeUnit, all_units: list[CodeUnit]) -> None:
        """Add edges for a unit's relationships.

        Args:
            unit: The unit to add edges for.
            all_units: All units for relationship resolution.
        """
        # Parent-child containment
        if unit.parent_id:
            self.graph.add_edge(
                unit.parent_id,
                unit.id,
                EdgeData(edge_type=EdgeType.CONTAINS),
            )
            self.graph.add_edge(
                unit.id,
                unit.parent_id,
                EdgeData(edge_type=EdgeType.CONTAINED_BY),
            )

        # Inheritance relationships
        if unit.base_classes:
            self._add_inheritance_edges(unit, all_units)

        # Call relationships
        if unit.calls:
            self._add_call_edges(unit, all_units)

    def _add_inheritance_edges(
        self,
        unit: CodeUnit,
        all_units: list[CodeUnit],
    ) -> None:
        """Add inheritance edges for a class.

        Args:
            unit: The class unit.
            all_units: All units for base class resolution.
        """
        if not unit.base_classes:
            return

        # Build a map of class names to units for resolution
        class_map: dict[str, CodeUnit] = {}
        for u in all_units:
            if u.type == CodeUnitType.CLASS:
                class_map[u.name] = u
                class_map[u.qualified_name] = u

        for base_class in unit.base_classes:
            # Try to find the base class
            base_unit = class_map.get(base_class)
            if base_unit:
                self.graph.add_edge(
                    unit.id,
                    base_unit.id,
                    EdgeData(edge_type=EdgeType.INHERITS),
                )
                self.graph.add_edge(
                    base_unit.id,
                    unit.id,
                    EdgeData(edge_type=EdgeType.INHERITED_BY),
                )
            else:
                # Create a placeholder node for external base classes
                external_id = f"external::{base_class}"
                if not self.graph.has_node(external_id):
                    self.graph.add_node(
                        NodeData(
                            id=external_id,
                            name=base_class,
                            qualified_name=base_class,
                            node_type=NodeType.CLASS,
                            metadata={"external": True},
                        )
                    )
                self.graph.add_edge(
                    unit.id,
                    external_id,
                    EdgeData(edge_type=EdgeType.INHERITS),
                )

    def _add_call_edges(self, unit: CodeUnit, all_units: list[CodeUnit]) -> None:
        """Add call edges for function/method calls.

        Args:
            unit: The calling unit.
            all_units: All units for callee resolution.
        """
        if not unit.calls:
            return

        # Build a map of function/method names to units
        callable_map: dict[str, CodeUnit] = {}
        for u in all_units:
            if u.type in (CodeUnitType.FUNCTION, CodeUnitType.METHOD):
                callable_map[u.name] = u
                callable_map[u.qualified_name] = u

        for call in unit.calls:
            # Extract the function name from method calls like "obj.method"
            call_name = call.split(".")[-1] if "." in call else call

            # Try to find the called function
            callee = callable_map.get(call_name) or callable_map.get(call)
            if callee and callee.id != unit.id:  # Avoid self-references
                self.graph.add_edge(
                    unit.id,
                    callee.id,
                    EdgeData(edge_type=EdgeType.CALLS),
                )
                self.graph.add_edge(
                    callee.id,
                    unit.id,
                    EdgeData(edge_type=EdgeType.CALLED_BY),
                )

    def _add_import_edges(self, parsed_file: ParsedFile) -> None:
        """Add import edges between modules.

        Args:
            parsed_file: The parsed file with import information.
        """
        if not parsed_file.imports:
            return

        source_id = f"{parsed_file.file_path}::module"

        for import_name in parsed_file.imports:
            # Create import node if doesn't exist
            import_id = f"import::{import_name}"
            if not self.graph.has_node(import_id):
                self.graph.add_node(
                    NodeData(
                        id=import_id,
                        name=import_name,
                        qualified_name=import_name,
                        node_type=NodeType.IMPORT,
                        metadata={"external": True},
                    )
                )

            self.graph.add_edge(
                source_id,
                import_id,
                EdgeData(edge_type=EdgeType.IMPORTS),
            )

    def _map_unit_type(self, unit_type: CodeUnitType) -> NodeType:
        """Map CodeUnitType to NodeType.

        Args:
            unit_type: The code unit type.

        Returns:
            Corresponding NodeType.
        """
        mapping = {
            CodeUnitType.MODULE: NodeType.MODULE,
            CodeUnitType.CLASS: NodeType.CLASS,
            CodeUnitType.FUNCTION: NodeType.FUNCTION,
            CodeUnitType.METHOD: NodeType.METHOD,
            CodeUnitType.PROPERTY: NodeType.METHOD,  # Treat as method
            CodeUnitType.VARIABLE: NodeType.VARIABLE,
            CodeUnitType.IMPORT: NodeType.IMPORT,
            CodeUnitType.DECORATOR: NodeType.FUNCTION,  # Treat as function
        }
        return mapping.get(unit_type, NodeType.FUNCTION)


def build_graph_from_directory(
    directory: Path,
    parser: "TreeSitterParser",  # noqa: F821 - forward reference
) -> CodeGraph:
    """Build a graph from all supported files in a directory.

    Args:
        directory: The directory to scan.
        parser: The parser to use.

    Returns:
        The built CodeGraph.
    """
    from smart_search.parsing.tree_sitter_parser import TreeSitterParser

    if not isinstance(parser, TreeSitterParser):
        raise TypeError("parser must be a TreeSitterParser instance")

    builder = GraphBuilder()
    parsed_files: list[ParsedFile] = []

    # Find all supported files
    for ext in [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"]:
        for file_path in directory.rglob(f"*{ext}"):
            # Skip common non-source directories
            if any(
                part in file_path.parts
                for part in ["node_modules", ".git", "__pycache__", ".venv", "venv"]
            ):
                continue

            try:
                parsed = parser.parse_file(file_path)
                parsed_files.append(parsed)
            except Exception as e:
                logger.warning(
                    "Failed to parse file",
                    file_path=str(file_path),
                    error=str(e),
                )

    return builder.build_from_parsed_files(parsed_files)
