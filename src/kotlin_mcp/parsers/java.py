import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Node

from .base import BaseParser, ClassInfo, FileSummary, FunctionInfo

JAVA_LANGUAGE = Language(tsjava.language())


class JavaParser(BaseParser):
    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    def parse(self, source: str, file_name: str = "") -> FileSummary:
        tree = self._parser.parse(source.encode())
        root = tree.root_node

        errors: list[str] = []
        self._collect_errors(root, source, errors)

        package = self._extract_package(root)
        imports = self._extract_imports(root)
        classes = self._extract_classes(root)

        return FileSummary(
            file_name=file_name,
            package=package,
            classes=classes,
            top_level_functions=[],
            imports=imports,
            has_syntax_errors=len(errors) > 0,
            error_snippets=errors,
        )

    def _collect_errors(self, node: Node, source: str, errors: list[str]) -> None:
        if node.type == "ERROR" or node.is_missing:
            start = max(0, node.start_byte - 30)
            end = min(len(source), node.end_byte + 30)
            errors.append(source[start:end].strip())
        for child in node.children:
            self._collect_errors(child, source, errors)

    def _extract_package(self, root: Node) -> str:
        for child in root.children:
            if child.type == "package_declaration":
                for sub in child.children:
                    if sub.type == "scoped_identifier":
                        return sub.text.decode()
        return ""

    def _extract_imports(self, root: Node) -> list[str]:
        imports: list[str] = []
        for child in root.children:
            if child.type == "import_declaration":
                for sub in child.children:
                    if sub.type in ("scoped_identifier", "identifier"):
                        imports.append(sub.text.decode())
        return imports

    def _extract_classes(self, root: Node) -> list[ClassInfo]:
        classes: list[ClassInfo] = []
        for child in root.children:
            if child.type == "class_declaration":
                classes.append(self._parse_class(child))
            elif child.type == "interface_declaration":
                classes.append(self._parse_interface(child))
            elif child.type == "enum_declaration":
                classes.append(self._parse_enum(child))
        return classes

    def _parse_class(self, node: Node) -> ClassInfo:
        name = self._get_name(node)
        annotations = self._get_annotations(node)
        modifiers = self._get_modifiers(node)
        kind = "abstract class" if "abstract" in modifiers else "class"
        constructor_params = self._extract_constructor_params(node)
        functions = self._extract_methods(node)

        return ClassInfo(
            name=name,
            kind=kind,
            constructor_params=constructor_params,
            functions=functions,
            annotations=annotations,
        )

    def _parse_interface(self, node: Node) -> ClassInfo:
        name = self._get_name(node)
        annotations = self._get_annotations(node)
        functions = self._extract_methods(node)

        return ClassInfo(
            name=name,
            kind="interface",
            constructor_params=[],
            functions=functions,
            annotations=annotations,
        )

    def _parse_enum(self, node: Node) -> ClassInfo:
        name = self._get_name(node)
        annotations = self._get_annotations(node)
        functions = self._extract_methods(node)

        return ClassInfo(
            name=name,
            kind="enum",
            constructor_params=[],
            functions=functions,
            annotations=annotations,
        )

    def _get_name(self, node: Node) -> str:
        name_node = node.child_by_field_name("name")
        return name_node.text.decode() if name_node else ""

    def _get_modifiers(self, node: Node) -> list[str]:
        modifiers: list[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    if mod.type != "marker_annotation" and mod.type != "annotation":
                        modifiers.append(mod.text.decode())
        return modifiers

    def _get_annotations(self, node: Node) -> list[str]:
        annotations: list[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    if mod.type in ("marker_annotation", "annotation"):
                        annotations.append(mod.text.decode().strip())
        return annotations

    def _extract_constructor_params(self, node: Node) -> list[str]:
        """Extract params from the first constructor_declaration in the class body."""
        params: list[str] = []
        body = node.child_by_field_name("body")
        if not body:
            return params
        for child in body.children:
            if child.type == "constructor_declaration":
                for param_child in child.children:
                    if param_child.type == "formal_parameters":
                        for p in param_child.children:
                            if p.type == "formal_parameter":
                                params.append(p.text.decode().strip())
                break  # only first constructor
        return params

    def _extract_methods(self, node: Node) -> list[FunctionInfo]:
        functions: list[FunctionInfo] = []
        body = node.child_by_field_name("body")
        if not body:
            return functions
        for child in body.children:
            if child.type == "method_declaration":
                functions.append(self._parse_method(child))
            elif child.type == "enum_body_declarations":
                # Enum methods sit one level deeper, after the constant list.
                for sub in child.children:
                    if sub.type == "method_declaration":
                        functions.append(self._parse_method(sub))
        return functions

    def _parse_method(self, node: Node) -> FunctionInfo:
        name = self._get_name(node)
        modifiers = self._get_modifiers(node)
        annotations = self._get_annotations(node)
        is_private = "private" in modifiers

        params: list[str] = []
        for child in node.children:
            if child.type == "formal_parameters":
                for p in child.children:
                    if p.type == "formal_parameter":
                        params.append(p.text.decode().strip())

        return_type = ""
        type_node = node.child_by_field_name("type")
        if type_node:
            return_type = type_node.text.decode()

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            is_private=is_private,
            is_suspend=False,
            annotations=annotations,
        )
