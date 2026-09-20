from dataclasses import dataclass, field


@dataclass
class FunctionInfo:
    name: str
    params: list[str]
    return_type: str
    is_private: bool
    is_suspend: bool = False
    annotations: list[str] = field(default_factory=list)


@dataclass
class PropertyInfo:
    name: str
    type: str
    is_private: bool
    is_mutable: bool = False  # var vs val


@dataclass
class ClassInfo:
    name: str
    kind: str  # class, data class, interface, object, enum
    constructor_params: list[str]
    functions: list[FunctionInfo]
    annotations: list[str] = field(default_factory=list)
    properties: list[PropertyInfo] = field(default_factory=list)


@dataclass
class FileSummary:
    file_name: str
    package: str
    classes: list[ClassInfo]
    top_level_functions: list[FunctionInfo]
    imports: list[str]
    has_syntax_errors: bool = False
    error_snippets: list[str] = field(default_factory=list)
    top_level_properties: list[PropertyInfo] = field(default_factory=list)


class BaseParser:
    def parse(self, source: str, file_name: str = "") -> FileSummary:
        raise NotImplementedError
