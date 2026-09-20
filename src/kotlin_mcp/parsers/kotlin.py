import tree_sitter_kotlin as tskotlin
from tree_sitter import Language, Parser, Node

from .base import BaseParser, ClassInfo, FileSummary, FunctionInfo, PropertyInfo

KT_LANGUAGE = Language(tskotlin.language())


class KotlinParser(BaseParser):
    def __init__(self) -> None:
        self._parser = Parser(KT_LANGUAGE)

    def parse(self, source: str, file_name: str = "") -> FileSummary:
        tree = self._parser.parse(source.encode())
        root = tree.root_node

        errors: list[str] = []
        self._collect_errors(root, source, errors)

        package = self._extract_package(root)
        imports = self._extract_imports(root)
        classes = self._extract_classes(root)
        top_funcs = self._extract_top_level_functions(root)
        top_props = self._extract_top_level_properties(root)

        return FileSummary(
            file_name=file_name,
            package=package,
            classes=classes,
            top_level_functions=top_funcs,
            imports=imports,
            has_syntax_errors=len(errors) > 0,
            error_snippets=errors,
            top_level_properties=top_props,
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
            if child.type == "package_header":
                for sub in child.children:
                    if sub.type == "qualified_identifier":
                        return sub.text.decode()
        return ""

    def _extract_imports(self, root: Node) -> list[str]:
        imports: list[str] = []
        for child in root.children:
            if child.type == "import":
                for sub in child.children:
                    if sub.type == "qualified_identifier":
                        imports.append(sub.text.decode())
        return imports

    def _extract_classes(self, root: Node) -> list[ClassInfo]:
        classes: list[ClassInfo] = []
        for child in root.children:
            if child.type == "class_declaration":
                classes.append(self._parse_class(child))
            elif child.type == "object_declaration":
                classes.append(self._parse_object(child))
        return classes

    def _parse_class(self, node: Node) -> ClassInfo:
        name = ""
        kind = "class"
        annotations: list[str] = []
        constructor_params: list[str] = []
        functions: list[FunctionInfo] = []

        # Check modifiers for data/enum/interface
        modifiers = self._get_modifiers(node)
        if "data" in modifiers:
            kind = "data class"
        elif "enum" in modifiers:
            kind = "enum"
        elif "interface" in modifiers or "interface" in (node.type or ""):
            kind = "interface"
        elif "abstract" in modifiers:
            kind = "abstract class"

        # Check if it's actually an interface
        for child in node.children:
            if child.type == "interface":
                kind = "interface"

        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()

        # Annotations
        annotations = self._get_annotations(node)

        # Constructor params
        for child in node.children:
            if child.type == "primary_constructor":
                constructor_params = self._extract_constructor_params(child)

        # Functions from class body
        properties: list[PropertyInfo] = []
        for child in node.children:
            if child.type == "class_body":
                functions = self._extract_functions_from_body(child)
                properties = self._extract_properties_from_body(child)

        return ClassInfo(
            name=name,
            kind=kind,
            constructor_params=constructor_params,
            functions=functions,
            annotations=annotations,
            properties=properties,
        )

    def _parse_object(self, node: Node) -> ClassInfo:
        name = ""
        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()

        annotations = self._get_annotations(node)
        functions: list[FunctionInfo] = []
        properties: list[PropertyInfo] = []
        for child in node.children:
            if child.type == "class_body":
                functions = self._extract_functions_from_body(child)
                properties = self._extract_properties_from_body(child)

        return ClassInfo(
            name=name,
            kind="object",
            constructor_params=[],
            functions=functions,
            annotations=annotations,
            properties=properties,
        )

    def _get_modifiers(self, node: Node) -> list[str]:
        modifiers: list[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    if mod.type in ("class_modifier", "visibility_modifier",
                                    "inheritance_modifier", "member_modifier"):
                        modifiers.append(mod.text.decode())
                    elif mod.type == "annotation":
                        pass  # handled separately
                    else:
                        modifiers.append(mod.text.decode())
        return modifiers

    def _get_annotations(self, node: Node) -> list[str]:
        annotations: list[str] = []
        for child in node.children:
            if child.type == "modifiers":
                for mod in child.children:
                    if mod.type == "annotation":
                        annotations.append(mod.text.decode().strip())
        return annotations

    def _extract_constructor_params(self, node: Node) -> list[str]:
        params: list[str] = []
        for child in node.children:
            if child.type == "class_parameters":
                for param in child.children:
                    if param.type == "class_parameter":
                        params.append(param.text.decode().strip())
            elif child.type == "class_parameter":
                params.append(child.text.decode().strip())
        return params

    def _extract_functions_from_body(self, body: Node) -> list[FunctionInfo]:
        functions: list[FunctionInfo] = []
        for child in body.children:
            if child.type == "function_declaration":
                functions.append(self._parse_function(child))
        return functions

    def _extract_top_level_functions(self, root: Node) -> list[FunctionInfo]:
        functions: list[FunctionInfo] = []
        for child in root.children:
            if child.type == "function_declaration":
                functions.append(self._parse_function(child))
        return functions

    def _extract_properties_from_body(self, body: Node) -> list[PropertyInfo]:
        properties: list[PropertyInfo] = []
        for child in body.children:
            if child.type == "property_declaration":
                properties.append(self._parse_property(child))
        return properties

    def _extract_top_level_properties(self, root: Node) -> list[PropertyInfo]:
        properties: list[PropertyInfo] = []
        for child in root.children:
            if child.type == "property_declaration":
                properties.append(self._parse_property(child))
        return properties

    def _parse_property(self, node: Node) -> PropertyInfo:
        modifiers = self._get_modifiers(node)
        is_private = "private" in modifiers
        is_mutable = any(child.type == "var" for child in node.children)

        name = ""
        prop_type = ""
        for child in node.children:
            if child.type != "variable_declaration":
                continue
            for sub in child.children:
                if sub.type in ("identifier", "simple_identifier") and not name:
                    name = sub.text.decode()
                elif sub.type in ("user_type", "nullable_type", "function_type",
                                  "type_identifier"):
                    prop_type = sub.text.decode()
            break

        return PropertyInfo(
            name=name,
            type=prop_type,
            is_private=is_private,
            is_mutable=is_mutable,
        )

    def _parse_function(self, node: Node) -> FunctionInfo:
        name = ""
        params: list[str] = []
        return_type = ""
        is_private = False
        is_suspend = False
        annotations: list[str] = []

        modifiers = self._get_modifiers(node)
        is_private = "private" in modifiers
        is_suspend = "suspend" in modifiers
        annotations = self._get_annotations(node)

        name_node = node.child_by_field_name("name")
        if name_node:
            name = name_node.text.decode()

        # Parameters
        for child in node.children:
            if child.type == "function_value_parameters":
                for param in child.children:
                    if param.type == "parameter":
                        params.append(param.text.decode().strip())

        # Return type — find user_type/nullable_type after ":"
        found_colon = False
        for child in node.children:
            if child.type == ":" and found_colon is False:
                # First colon after function_value_parameters is return type
                found_colon = True
            elif found_colon and child.type in ("user_type", "nullable_type",
                                                  "type_identifier"):
                return_type = child.text.decode()
                break
            elif child.type == "function_body":
                break

        return FunctionInfo(
            name=name,
            params=params,
            return_type=return_type,
            is_private=is_private,
            is_suspend=is_suspend,
            annotations=annotations,
        )
