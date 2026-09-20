"""Module discovery and selection for the task-based evaluation (RQ2).

The unit of analysis is a Gradle module, not a whole repository. No context
window holds a multi-thousand-file Android project, so comparing the Full
condition against Summary at repository level is not possible for the larger
datasets — the request is simply rejected.

Selection rule: per project, take the largest module whose full source fits the
token budget. Largest makes the comparison as demanding as it can be; fitting
the budget is what lets all three conditions run over identical material. When
no module fits, the caller records a measured failure rather than silently
dropping the project.
"""

import logging
from pathlib import Path

from kotlin_mcp.summarizer import SKIP_DIRS
from token_counter import count_tokens_tiktoken

logger = logging.getLogger(__name__)

BUILD_SCRIPTS = ("build.gradle", "build.gradle.kts")


def find_modules(project_root: Path) -> list[Path]:
    """Directories holding a Gradle build script, excluding the root project.

    The root is excluded because its build script usually configures
    subprojects rather than owning source of its own. Build output, VCS and IDE
    directories are skipped with the same denylist the file collector uses, so
    a stray build script under build/ or a nested checkout under .claude/ does
    not register as a module.
    """
    modules = [
        path.parent
        for name in BUILD_SCRIPTS
        for path in project_root.rglob(name)
        if path.parent != project_root
        and not _is_ignorable(path.parent.relative_to(project_root))
    ]
    return sorted(set(modules))


def _is_ignorable(rel: Path) -> bool:
    return any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts)


def module_own_files(module: Path, all_modules: list[Path], collect) -> list[Path]:
    """Source files belonging to `module` itself, not to a nested module.

    `collect` is the file collector to use (kotlin_mcp.summarizer._collect_files),
    injected so this module stays independent of how filtering is configured.
    """
    nested = [m for m in all_modules if m != module and _is_under(m, module)]
    own = []
    for f in collect(module, None):
        if not any(_is_under(f, n) for n in nested):
            own.append(f)
    return own


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def measure_modules(project_root: Path, collect) -> list[dict]:
    """Token cost of the full source of every module, largest first."""
    modules = find_modules(project_root)
    measured = []
    for module in modules:
        files = module_own_files(module, modules, collect)
        if not files:
            continue
        tokens = sum(
            count_tokens_tiktoken(f.read_text(encoding="utf-8", errors="replace"))
            for f in files
        )
        measured.append(
            {
                "path": module,
                "name": str(module.relative_to(project_root)),
                "files": len(files),
                "full_tokens": tokens,
            }
        )
    measured.sort(key=lambda m: (-m["full_tokens"], m["name"]))
    return measured


def select_module(project_root: Path, collect, budget: int) -> tuple[dict | None, list[dict]]:
    """Largest module whose full source fits `budget`.

    Returns (chosen, all_measured). `chosen` is None when nothing fits, which
    the caller should record as a measured failure — the token cost of the
    smallest module is itself the finding.
    """
    measured = measure_modules(project_root, collect)
    for module in measured:  # already sorted largest first
        if module["full_tokens"] <= budget:
            return module, measured
    return None, measured
