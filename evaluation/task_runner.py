"""RQ2: Task-based evaluation harness.

Runs tasks against Claude API under three conditions:
  1. Full Code — all source files as context
  2. Summary Only — structural summaries from kotlin-mcp
  3. Summary + On-demand — summary first, can request specific files

The unit of analysis is one Gradle module per project, not the whole
repository: no context window holds a multi-thousand-file Android project, so
the Full condition cannot run at repository scale. Per project we take the
largest module whose full source fits MAX_CONTEXT_TOKENS, so all three
conditions see identical material. A project where nothing fits is written out
as a measured failure, not dropped.

Usage:
    uv run python evaluation/task_runner.py \
        --tasks evaluation/tasks_instantiated.json \
        --datasets evaluation/datasets.json \
        --output evaluation/results/rq2_responses.json \
        --condition full|summary|hybrid \
        --model claude-sonnet-5

    # Build every context and report its token cost without calling the API:
    ... --condition full --dry-run
"""

import argparse
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anthropic

from cli_backend import CliBackend, verify_no_file_access

from kotlin_mcp.summarizer import _collect_files, _format_summary
from kotlin_mcp.parsers.java import JavaParser
from kotlin_mcp.parsers.kotlin import KotlinParser
from modules import module_own_files, find_modules, select_module
from token_counter import count_tokens_tiktoken

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are evaluating a Kotlin/Java codebase. Answer the question based only on the provided context.
Be specific: mention exact class names, file names, package names, and method signatures when relevant.
If the provided context is insufficient to answer, say so explicitly and explain what information is missing."""

MAX_CONTEXT_TOKENS = 180_000  # Leave room for the response
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0  # seconds, doubled each attempt
TEMPERATURE = 0.0  # responses must be reproducible

_kotlin_parser = KotlinParser()
_java_parser = JavaParser()


def load_full_source(module_path: Path, files: list[Path]) -> str:
    """Concatenate the full text of every source file in the module."""
    parts = []
    for f in sorted(files):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            parts.append(f"=== {f.relative_to(module_path)} ===\n{content}")
        except OSError as e:
            logger.warning("Failed to read %s: %s", f, e)
    return "\n\n".join(parts)


def load_summary(module_path: Path, files: list[Path]) -> str:
    """Structural summary of the module, grouped by package.

    Built here rather than through summarize_module so the evaluation reads
    exactly the files it measured, and so nothing is written into the dataset
    checkout while measuring it.
    """
    summaries = []
    for f in sorted(files):
        source = f.read_text(encoding="utf-8", errors="replace")
        parser = _kotlin_parser if f.suffix == ".kt" else _java_parser
        summaries.append(parser.parse(source, file_name=f.name))

    packages: dict[str, list[str]] = {}
    for s in summaries:
        packages.setdefault(s.package or "(default)", []).append(s.file_name)

    lines = [f"Module: {module_path.name} ({len(summaries)} files)", "", "Packages:"]
    for pkg, names in sorted(packages.items()):
        lines.append(f"  {pkg} ({len(names)} files)")
        for n in sorted(names):
            lines.append(f"    - {n}")
    lines += ["", "--- Per-file summaries below ---", ""]
    for s in summaries:
        lines.append(_format_summary(s))
        lines.append("")
    return "\n".join(lines)


def build_context(project_path: Path) -> dict:
    """Pick the module for this project and build both contexts once.

    Returned dict carries either both contexts, or the reason no module fits.
    """
    chosen, measured = select_module(project_path, _collect_files, MAX_CONTEXT_TOKENS)
    if chosen is None:
        smallest = min((m["full_tokens"] for m in measured), default=0)
        return {
            "ok": False,
            "reason": "context_budget_exceeded",
            "budget": MAX_CONTEXT_TOKENS,
            "smallest_module_tokens": smallest,
            "modules_considered": len(measured),
        }

    all_modules = find_modules(project_path)
    files = module_own_files(chosen["path"], all_modules, _collect_files)
    full = load_full_source(chosen["path"], files)
    summary = load_summary(chosen["path"], files)

    logger.info(
        "  module %s: %d files, full %d tok, summary %d tok",
        chosen["name"], len(files), chosen["full_tokens"],
        count_tokens_tiktoken(summary),
    )
    return {
        "ok": True,
        "module": chosen["name"],
        "module_path": chosen["path"],
        "module_files": len(files),
        "full": full,
        "full_tokens": chosen["full_tokens"],
        "summary": summary,
        "summary_tokens": count_tokens_tiktoken(summary),
    }


def run_task_full(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    source_context: str,
) -> dict:
    """Condition 1: Full source code as context."""
    messages = [
        {
            "role": "user",
            "content": f"Here is the full source code of the project:\n\n{source_context}\n\n---\n\nQuestion: {question}",
        }
    ]
    return _call_api(client, model, messages)


def run_task_summary(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    summary_context: str,
) -> dict:
    """Condition 2: Summary only as context."""
    messages = [
        {
            "role": "user",
            "content": f"Here is a structural summary of the project (signatures, classes, methods — no implementation bodies):\n\n{summary_context}\n\n---\n\nQuestion: {question}",
        }
    ]
    return _call_api(client, model, messages)


def run_task_hybrid(
    client: anthropic.Anthropic,
    model: str,
    question: str,
    summary_context: str,
    module_path: Path,
    module_files: list[Path],
) -> dict:
    """Condition 3: Summary first, then allow requesting specific files.

    Uses a two-turn conversation:
    1. Provide summary + question, ask LLM if it needs any full files
    2. If it requests files, provide them and re-ask
    """
    messages = [
        {
            "role": "user",
            "content": (
                f"Here is a structural summary of the project (signatures, classes, methods — no implementation bodies):\n\n"
                f"{summary_context}\n\n---\n\n"
                f"Question: {question}\n\n"
                f"If you need to see the full source of specific files to answer accurately, "
                f"list them as: REQUEST_FILES: file1.kt, file2.java\n"
                f"Otherwise, answer the question directly."
            ),
        }
    ]

    result = _call_api(client, model, messages)

    # Check if the model requested files
    response_text = result["response"]
    if "REQUEST_FILES:" in response_text:
        requested_files = _parse_requested_files(response_text)

        if requested_files:
            file_contents, resolution = _load_requested_files(
                module_path, module_files, requested_files
            )

            messages.append({"role": "assistant", "content": response_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"Here are the requested files:\n\n{file_contents}\n\nNow please answer the original question: {question}",
                }
            )

            result2 = _call_api(client, model, messages)
            # Combine token usage
            # billed_input_tokens and thinking_tokens were left out here, so a
            # two-call Hybrid task reported only its second call and the
            # condition looked as cheap as Summary.
            for k in ("input_tokens", "output_tokens", "cache_read_tokens",
                      "cache_write_tokens", "billed_input_tokens",
                      "thinking_tokens"):
                result2[k] += result[k]
            result2["files_requested"] = requested_files
            result2["turns"] = 2
            result2["attempts"] += result["attempts"]
            result2.update(resolution)
            return result2

    result["files_requested"] = []
    result["turns"] = 1
    result["files_resolved"] = []
    result["files_ambiguous"] = []
    result["files_missing"] = []
    return result


_FILE_TOKEN = re.compile(r"[\w./\-]+\.(?:kt|java)\b")


def _parse_requested_files(text: str) -> list[str]:
    """Pull file names out of a REQUEST_FILES reply.

    Models decorate the marker with markdown, spread the list over several
    lines, and append prose: "** `Scrollbar.kt` (full implementation)". Taking
    one line and splitting it on commas captured "**" and nothing else, so the
    second turn arrived without the file the model had asked for and Hybrid
    degraded into Summary plus a wasted round trip. Scanning the text after the
    marker for things that actually look like Kotlin or Java paths survives the
    decoration, the line breaks and the prose alike.
    """
    idx = text.upper().find("REQUEST_FILES")
    if idx < 0:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _FILE_TOKEN.finditer(text[idx:]):
        name = m.group(0).lstrip("./")
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _load_requested_files(
    module_path: Path, files: list[Path], requested: list[str]
) -> tuple[str, dict]:
    """Load the files the model asked for, matching on relative path.

    Matching on the bare file name collapses duplicates: Android modules are
    full of same-named files (MainActivity.kt, Module.kt, Screen.kt) in
    different packages, and a name-keyed dict silently returns whichever came
    last. A request that names only the file is honoured when exactly one file
    carries that name; when several do, every candidate is sent and the
    ambiguity is recorded.
    """
    by_rel = {str(f.relative_to(module_path)): f for f in files}
    by_name: dict[str, list[Path]] = {}
    for f in files:
        by_name.setdefault(f.name, []).append(f)

    parts: list[str] = []
    resolved: list[str] = []
    ambiguous: list[str] = []
    missing: list[str] = []

    def emit(path: Path) -> None:
        rel = str(path.relative_to(module_path))
        try:
            parts.append(f"=== {rel} ===\n{path.read_text(encoding='utf-8', errors='replace')}")
            resolved.append(rel)
        except OSError as e:
            parts.append(f"=== {rel} === (error: {e})")

    for raw in requested:
        name = raw.strip().lstrip("./")
        # A model often gives a path rooted somewhere other than this module,
        # so the bare file name is the fallback the exact path cannot cover.
        base = Path(name).name
        if name in by_rel:
            emit(by_rel[name])
        elif len(by_name.get(base, [])) == 1:
            emit(by_name[base][0])
        elif by_name.get(base):
            ambiguous.append(base)
            for candidate in by_name[base]:
                emit(candidate)
        else:
            missing.append(name)
            parts.append(f"=== {name} === (file not found in this module)")

    return "\n\n".join(parts), {
        "files_resolved": resolved,
        "files_ambiguous": ambiguous,
        "files_missing": missing,
    }


def _call_api(
    client: anthropic.Anthropic,
    model: str,
    messages: list[dict],
) -> dict:
    """Call Claude API and return the response with token usage.

    Retries on rate limits and overload so one throttled call does not drop a
    task from the run. latency_seconds is wall-clock for the whole call, not
    time-to-first-token — the request is not streamed.
    """
    if isinstance(client, CliBackend):
        return client.call(model, messages)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        start = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=TEMPERATURE,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            status = getattr(e, "status_code", None)
            if status is not None and status not in (429, 500, 502, 503, 529):
                raise
            last_error = e
            delay = RETRY_BASE_DELAY * (2**attempt) + random.uniform(0, 1)
            logger.warning(
                "  API %s, percobaan %d/%d, tunggu %.1fs",
                status or type(e).__name__, attempt + 1, MAX_RETRIES, delay,
            )
            time.sleep(delay)
            continue

        elapsed = time.time() - start
        text = "".join(b.text for b in response.content if b.type == "text")
        # Recorded so the paper's "no prompt cache" claim can be checked from
        # the results rather than taken on trust: both stay zero while
        # cache_control is absent from the request.
        usage = response.usage
        return {
            "response": text,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cache_write_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "latency_seconds": round(elapsed, 2),
            "model": model,
            "temperature": TEMPERATURE,
            "stop_reason": response.stop_reason,
            "attempts": attempt + 1,
        }

    raise RuntimeError(f"gagal setelah {MAX_RETRIES} percobaan: {last_error}")


def main():
    parser = argparse.ArgumentParser(description="RQ2 task-based evaluation")
    parser.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Path to instantiated tasks JSON",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="evaluation/datasets.json",
        help="Path to datasets JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/rq2_responses.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=["full", "summary", "hybrid"],
        required=True,
        help="Experiment condition",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-5",
        help="Claude model id, e.g. claude-haiku-4-5 / claude-sonnet-5 / claude-opus-5",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Run only for a specific project name (optional)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run tasks already present in the output file (default: skip them)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build every context and report its token cost without calling the API",
    )
    parser.add_argument(
        "--backend",
        choices=("api", "cli"),
        default="api",
        help=(
            "api: Messages API, needs ANTHROPIC_API_KEY, sets temperature. "
            "cli: claude -p on a subscription, no temperature control and "
            "the CLI's own scaffolding lands in the billed token counts"
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        client = None
    elif args.backend == "cli":
        client = CliBackend(SYSTEM_PROMPT)
        # The whole design rests on the subject seeing only the context we
        # built. Check it before spending a run rather than after.
        verify_no_file_access(client, args.model)
    else:
        client = anthropic.Anthropic()

    with open(args.tasks) as f:
        tasks_data = json.load(f)

    with open(args.datasets) as f:
        datasets = json.load(f)

    project_map = {p["name"]: Path(p["path"]).expanduser() for p in datasets["projects"]}

    output_path = Path(args.output)
    existing: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)

    # Keyed so a re-run replaces a row instead of appending a duplicate. The
    # model is part of the key: the same task under the same condition is a
    # different observation on a different model, and the evaluation sweeps
    # several capability tiers.
    def key(row):
        return (row["task_id"], row["condition"], row.get("model", ""))

    by_key = {key(r): r for r in existing}
    already = set(by_key)

    contexts: dict[str, dict] = {}  # built once per project, reused by every task
    results = []
    skipped = 0

    def flush() -> None:
        """Persist what has been collected so far.

        Called after every task. Writing once at the end of a condition means a
        process that dies on the last task loses every response before it, and
        those responses cost real quota. Written to a temporary file and then
        renamed so a crash during the write cannot leave a truncated file
        behind.
        """
        if args.dry_run:
            return
        for row in results:
            by_key[key(row)] = row
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(list(by_key.values()), f, indent=2)
        tmp.replace(output_path)

    for task in tasks_data["tasks"]:
        project_name = task["project"]
        if args.project and project_name != args.project:
            continue

        task_id = task["id"]
        if (task_id, args.condition, args.model) in already and not args.overwrite:
            skipped += 1
            continue

        project_path = project_map.get(project_name)
        if not project_path or not project_path.exists():
            logger.warning("Project %s not found, skipping", project_name)
            continue

        if project_name not in contexts:
            logger.info("Building context for %s...", project_name)
            contexts[project_name] = build_context(project_path)
        ctx = contexts[project_name]

        base = {
            "task_id": task_id,
            "project": project_name,
            "category": task["category"],
            "question": task["question"],
            "condition": args.condition,
            "model": args.model,
        }

        if not ctx["ok"]:
            # No module fits the budget. Recorded with its measurement so the
            # failure can be reported, rather than vanishing from the results.
            logger.warning("  %s: %s", task_id, ctx["reason"])
            results.append({**base, "error": ctx["reason"], **{
                k: v for k, v in ctx.items() if k not in ("ok", "reason")
            }})
            flush()
            continue

        base |= {
            "module": ctx["module"],
            "module_files": ctx["module_files"],
            "context_tokens_full": ctx["full_tokens"],
            "context_tokens_summary": ctx["summary_tokens"],
        }

        if args.dry_run:
            results.append({**base, "dry_run": True})
            continue

        question = task["question"]
        logger.info("Running %s [%s]", task_id, args.condition)
        try:
            if args.condition == "full":
                result = run_task_full(client, args.model, question, ctx["full"])
            elif args.condition == "summary":
                result = run_task_summary(client, args.model, question, ctx["summary"])
            else:
                module_files = module_own_files(
                    ctx["module_path"], find_modules(project_path), _collect_files
                )
                result = run_task_hybrid(
                    client, args.model, question, ctx["summary"],
                    ctx["module_path"], module_files,
                )
            results.append({**base, **result})
        except Exception as e:
            logger.error("Error on task %s: %s", task_id, e)
            results.append({**base, "error": f"{type(e).__name__}: {e}"})
        flush()

    if args.dry_run:
        for row in results:
            by_key[key(row)] = row
        _report(results, args.condition, skipped, dry_run=True)
        return

    flush()
    logger.info("Wrote %d rows to %s", len(by_key), output_path)

    _report(results, args.condition, skipped)


def _report(results: list[dict], condition: str, skipped: int, dry_run: bool = False) -> None:
    failed = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]
    print(f"\n=== {condition}{' (dry run)' if dry_run else ''} ===")
    print(f"Tasks handled: {len(results)}  (skipped as already scored: {skipped})")
    if failed:
        print(f"Failed: {len(failed)}")
        reasons: dict[str, int] = {}
        for r in failed:
            reasons[r["error"]] = reasons.get(r["error"], 0) + 1
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {n:3}  {reason}")
    if dry_run:
        key = "context_tokens_full" if condition == "full" else "context_tokens_summary"
        tokens = [r[key] for r in ok if key in r]
        if tokens:
            print(f"Context tokens per task: min {min(tokens):,}  max {max(tokens):,}")
            print(f"Total input tokens if run: {sum(tokens):,}")
        return
    # The CLI backend puts nearly the whole request into the cache fields, so
    # input_tokens alone reports a few hundred tokens for a run that carried
    # half a million. Sum what was actually billed.
    billed = sum(
        r.get("billed_input_tokens")
        or (r.get("input_tokens", 0) + r.get("cache_write_tokens", 0)
            + r.get("cache_read_tokens", 0))
        for r in ok
    )
    print(f"Billed input tokens: {billed:,}")
    print(f"Total output tokens: {sum(r.get('output_tokens', 0) for r in ok):,}")
    thinking = sum(r.get("thinking_tokens", 0) or 0 for r in ok)
    if thinking:
        print(f"  of which thinking: {thinking:,}")


if __name__ == "__main__":
    main()
