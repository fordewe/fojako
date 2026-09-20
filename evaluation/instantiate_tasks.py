"""Instantiate task templates with real class/function names from dataset projects.

Elements are drawn from the same module the task runner will use as context —
the largest module that fits the token budget. Drawing from the whole
repository would produce questions about classes the model is never shown,
turning the evaluation into a test of refusing to guess.

Usage:
    uv run python evaluation/instantiate_tasks.py \
        --datasets evaluation/datasets.json \
        --templates evaluation/tasks.json \
        --output evaluation/tasks_instantiated.json

This script scans each project, picks real class names, interfaces, and functions,
then fills in the task templates to create concrete evaluation tasks.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kotlin_mcp.summarizer import _collect_files
from kotlin_mcp.parsers.kotlin import KotlinParser
from kotlin_mcp.parsers.java import JavaParser
from modules import find_modules, module_own_files, select_module

MAX_CONTEXT_TOKENS = 180_000  # must match task_runner.MAX_CONTEXT_TOKENS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_kotlin_parser = KotlinParser()
_java_parser = JavaParser()


def extract_module_elements(module_path: Path, files: list[Path]) -> dict:
    """Extract classes, interfaces, functions, packages from one module."""
    classes = []
    interfaces = []
    functions = []
    packages = set()
    class_pairs = []  # (class_a, class_b) from same package

    for f in files:
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
            parser = _kotlin_parser if f.suffix == ".kt" else _java_parser
            summary = parser.parse(source, file_name=f.name)

            if summary.package:
                packages.add(summary.package)

            for cls in summary.classes:
                entry = {
                    "name": cls.name,
                    "kind": cls.kind,
                    "file": f.name,
                    "package": summary.package,
                    "has_constructor": bool(cls.constructor_params),
                    "has_functions": bool(cls.functions),
                    "function_names": [fn.name for fn in cls.functions if fn.name],
                }
                if cls.kind == "interface":
                    interfaces.append(entry)
                else:
                    classes.append(entry)

            for fn in summary.top_level_functions:
                functions.append(
                    {
                        "name": fn.name,
                        "file": f.name,
                        "package": summary.package,
                    }
                )

            # Track class pairs in same file
            if len(summary.classes) >= 2:
                for i in range(len(summary.classes)):
                    for j in range(i + 1, len(summary.classes)):
                        class_pairs.append(
                            (summary.classes[i].name, summary.classes[j].name)
                        )
        except Exception as e:
            logger.warning("Error parsing %s: %s", f, e)

    return {
        "classes": classes,
        "interfaces": interfaces,
        "functions": functions,
        "packages": sorted(packages),
        "class_pairs": class_pairs,
    }


def pick(items: list, n: int = 1) -> list:
    """Randomly pick n items from list."""
    if not items:
        return []
    return random.sample(items, min(n, len(items)))


def infer_layers(packages: list[str]) -> dict[str, str]:
    """Guess architectural layers from package names."""
    layers = {}
    for pkg in packages:
        parts = pkg.lower().split(".")
        for part in parts:
            if part in ("ui", "view", "screen", "compose", "fragment", "activity"):
                layers["UI"] = pkg
            elif part in ("domain", "usecase", "interactor"):
                layers["domain"] = pkg
            elif part in ("data", "repository", "repo", "api", "network", "db", "database"):
                layers["data"] = pkg
            elif part in ("di", "injection", "module"):
                layers["DI"] = pkg
    return layers


def instantiate_tasks(
    project_name: str,
    elements: dict,
    templates: list[dict],
) -> list[dict]:
    """Create concrete tasks from templates using extracted elements."""
    tasks = []

    classes_with_fns = [c for c in elements["classes"] if c["has_functions"]]
    classes_with_named_fns = [c for c in elements["classes"] if c["function_names"]]
    classes_with_ctor = [c for c in elements["classes"] if c["has_constructor"]]
    layers = infer_layers(elements["packages"])

    for tmpl in templates:
        tid = tmpl["id"]
        category = tmpl["category"]
        template = tmpl["template"]

        try:
            if tid == "nav_01" and elements["classes"]:
                cls = pick(elements["classes"])[0]
                question = template.format(class_name=cls["name"])
            elif tid == "nav_02" and elements["interfaces"]:
                iface = pick(elements["interfaces"])[0]
                question = template.format(interface_name=iface["name"])
            elif tid == "nav_03" and classes_with_fns:
                cls = pick(classes_with_fns)[0]
                question = template.format(class_name=cls["name"])
            elif tid == "nav_04" and classes_with_ctor:
                cls = pick(classes_with_ctor)[0]
                question = template.format(class_name=cls["name"])
            elif tid == "nav_05" and layers:
                layer_name = pick(list(layers.keys()))[0]
                question = template.format(layer_name=layer_name)
            elif tid == "und_01" and layers:
                src = "API" if "data" in layers else "data source"
                dst = "UI" if "UI" in layers else "screen"
                question = template.format(source=src, destination=dst)
            elif tid == "und_02":
                question = template  # no substitution needed
            elif tid == "und_03" and elements["class_pairs"]:
                pair = pick(elements["class_pairs"])[0]
                question = template.format(class_a=pair[0], class_b=pair[1])
            elif tid == "und_04" and classes_with_named_fns:
                cls = pick(classes_with_named_fns)[0]
                fn_name = pick(cls["function_names"])[0]
                question = template.format(
                    function_name=fn_name, class_name=cls["name"]
                )
            elif tid == "und_05":
                question = template  # no substitution needed
            elif tid == "mod_01" and elements["classes"]:
                cls = pick(elements["classes"])[0]
                question = template.format(
                    feature_type="API endpoint", entity_name=cls["name"]
                )
            elif tid == "mod_02" and elements["classes"]:
                cls = pick(elements["classes"])[0]
                question = template.format(entity_name=cls["name"])
            elif tid == "mod_03" and elements["classes"]:
                cls = pick(elements["classes"])[0]
                question = template.format(
                    class_name=cls["name"],
                    new_name=cls["name"] + "V2",
                )
            elif tid == "mod_04":
                question = template.format(
                    feature_description="a caching layer for network responses"
                )
            elif tid == "mod_05":
                question = template  # no substitution needed
            else:
                logger.warning(
                    "Skipping task %s for %s: insufficient data", tid, project_name
                )
                continue

            tasks.append(
                {
                    "id": f"{project_name}_{tid}",
                    "project": project_name,
                    "category": category,
                    "template_id": tid,
                    "question": question,
                    "difficulty": tmpl["difficulty"],
                }
            )
        except Exception as e:
            logger.warning("Error instantiating %s for %s: %s", tid, project_name, e)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Instantiate evaluation tasks")
    parser.add_argument("--datasets", type=str, default="evaluation/datasets.json")
    parser.add_argument("--templates", type=str, default="evaluation/tasks.json")
    parser.add_argument(
        "--output", type=str, default="evaluation/tasks_instantiated.json"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.datasets) as f:
        datasets = json.load(f)
    with open(args.templates) as f:
        templates_data = json.load(f)

    all_tasks = []
    for project in datasets["projects"]:
        project_path = Path(project["path"]).expanduser()
        if not project_path.exists():
            logger.warning("Project not found: %s", project_path)
            continue

        chosen, measured = select_module(
            project_path, _collect_files, MAX_CONTEXT_TOKENS
        )
        if chosen is None:
            logger.warning(
                "  %s: no module fits %d tokens (%d considered) — no tasks generated",
                project["name"], MAX_CONTEXT_TOKENS, len(measured),
            )
            continue

        files = module_own_files(
            chosen["path"], find_modules(project_path), _collect_files
        )
        logger.info(
            "Extracting from %s / module %s (%d files, %d tok)...",
            project["name"], chosen["name"], len(files), chosen["full_tokens"],
        )
        elements = extract_module_elements(chosen["path"], files)
        logger.info(
            "  Found %d classes, %d interfaces, %d functions",
            len(elements["classes"]),
            len(elements["interfaces"]),
            len(elements["functions"]),
        )

        tasks = instantiate_tasks(
            project["name"], elements, templates_data["task_templates"]
        )
        for task in tasks:
            task["module"] = chosen["name"]
        all_tasks.extend(tasks)
        logger.info("  Generated %d tasks", len(tasks))

    output = {
        "version": "1.0",
        "seed": args.seed,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "total_tasks": len(all_tasks),
        "tasks": all_tasks,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Total tasks: %d, saved to %s", len(all_tasks), args.output)


if __name__ == "__main__":
    main()
