"""Turn RQ2's collected responses into the numbers Section 5.3 (RQ3) reports.

Reads ``rq2_responses.json``, already collected by ``task_runner.py`` with no
new API calls, and:

  * rebuilds Hybrid's missing input-token figure, which ``analyze_rq2.py``
    leaves ``NaN`` in Table 3. A Hybrid task's first turn costs exactly what
    Summary costs; its second turn sends whatever files the model asked for,
    a set that varies per task and was never tokenized as a single number at
    collection time. This script reads those files back off disk and counts
    them the same way RQ1 counted everything else, with ``tiktoken``.
  * prints Table 5 (mean input/output/total tokens per condition) and
    Table 6 (tasks a fixed token budget covers per condition).

Two token vocabularies exist in the data, and only one belongs in these
tables. ``billed_input_tokens`` is what the CLI subscription backend actually
billed, inflated by its own scaffolding (see the Section 5 backend decision
in RUN-LOG.md) — comparing conditions on it would mostly compare how much
scaffolding each call happened to re-send. ``context_tokens_full`` /
``context_tokens_summary`` (and the Hybrid figure computed here) are the size
of the module material this evaluation itself built, comparable to RQ1.
Tables 5 and 6 use the second vocabulary; the first belongs in the
methodology's cost caveat, not in a strategy comparison.

Usage:
    uv run python evaluation/analyze_rq3.py
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_rq2 import complete_tasks, load_scores  # noqa: E402
from token_counter import count_tokens_tiktoken  # noqa: E402

CONDITIONS = ["full", "summary", "hybrid"]
LABEL = {"full": "Full", "summary": "Summary", "hybrid": "Hybrid"}
BUDGETS = [100_000, 1_000_000, 10_000_000]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def hybrid_context_tokens(
    context_tokens_summary: int, resolved_relpaths: list[str], file_contents: dict[str, str],
) -> int:
    """Context a Hybrid task actually saw: the summary plus any files it asked for.

    A task that never requested a file (``resolved_relpaths == []``) costs
    exactly what Summary costs, which is the right floor: Hybrid can only add
    to Summary's context, never subtract from it.
    """
    extra = sum(count_tokens_tiktoken(file_contents[p]) for p in resolved_relpaths)
    return context_tokens_summary + extra


def load_resolved_file_contents(module_path: Path, relpaths: list[str]) -> dict[str, str]:
    """Read the files a Hybrid task's second turn actually sent, off disk.

    ``files_resolved`` already carries paths relative to ``module_path`` and
    already survived the ambiguity resolution ``task_runner.py`` did at
    collection time, so this only reads — it does not re-resolve.
    """
    return {
        rel: (module_path / rel).read_text(encoding="utf-8", errors="replace")
        for rel in relpaths
    }


def fill_hybrid_context_tokens(
    responses: list[dict], project_paths: dict[str, Path],
) -> dict[str, int]:
    """context_tokens_hybrid per Hybrid task_id, reading each module once."""
    module_paths: dict[str, Path] = {}
    out: dict[str, int] = {}
    for r in responses:
        if r["condition"] != "hybrid":
            continue
        project = r["project"]
        if project not in module_paths:
            module_paths[project] = project_paths[project] / r["module"]
        contents = load_resolved_file_contents(module_paths[project], r["files_resolved"])
        out[r["task_id"]] = hybrid_context_tokens(
            r["context_tokens_summary"], r["files_resolved"], contents
        )
    return out


def token_consumption(
    responses: list[dict], done: set[str], hybrid_context: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Mean input/output/total tokens per condition, over the balanced task set."""
    by_cond: dict[str, dict[str, list[float]]] = {
        c: {"input": [], "output": []} for c in CONDITIONS
    }
    for r in responses:
        if r["task_id"] not in done:
            continue
        c = r["condition"]
        if c == "hybrid":
            by_cond[c]["input"].append(hybrid_context[r["task_id"]])
        else:
            field = "context_tokens_full" if c == "full" else "context_tokens_summary"
            by_cond[c]["input"].append(r[field])
        by_cond[c]["output"].append(r["output_tokens"])

    result = {}
    for c in CONDITIONS:
        inp = mean(by_cond[c]["input"])
        out = mean(by_cond[c]["output"])
        result[c] = {"input": inp, "output": out, "total": inp + out}
    return result


def table_token_consumption(consumption: dict[str, dict[str, float]]) -> str:
    full_total = consumption["full"]["total"]
    lines = []
    for c in CONDITIONS:
        v = consumption[c]
        vs_full = "---" if c == "full" else f"{v['total'] / full_total * 100:.1f}\\%"
        lines.append(
            f"        {LABEL[c]:<7} & {v['input']:,.0f} & {v['output']:,.0f} "
            f"& {v['total']:,.0f} & {vs_full} \\\\ \\hline"
        )
    return "\n".join(lines)


def table_budget(consumption: dict[str, dict[str, float]]) -> str:
    lines = []
    for budget in BUDGETS:
        cells = [f"{int(budget // consumption[c]['total']):,}" for c in CONDITIONS]
        label = f"{budget // 1000}K" if budget < 1_000_000 else f"{budget // 1_000_000}M"
        lines.append(f"        {label:<6} & " + " & ".join(cells) + " \\\\ \\hline")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ2 responses into RQ3 token tables")
    parser.add_argument("--responses", default="evaluation/results/rq2_responses.json")
    parser.add_argument("--scores", default="evaluation/results/rq2_judge_scores.csv")
    parser.add_argument("--subset", default="evaluation/results/rq2_judge_subset.json")
    parser.add_argument("--datasets", default="evaluation/datasets.json")
    args = parser.parse_args()

    responses = json.load(open(args.responses))

    keep = None
    subset = Path(args.subset)
    if subset.exists():
        keep = {r["task_id"] for r in json.load(open(subset))}
    done = complete_tasks(load_scores(Path(args.scores), keep))
    if not done:
        print("Belum ada tugas yang lengkap di tiga kondisi. Tabel tidak bisa dihitung.")
        return

    project_paths = {
        p["name"]: Path(p["path"]).expanduser()
        for p in json.load(open(args.datasets))["projects"]
    }
    hybrid_context = fill_hybrid_context_tokens(
        [r for r in responses if r["task_id"] in done], project_paths,
    )

    consumption = token_consumption(responses, done, hybrid_context)

    print("\n% ── Tabel: konsumsi token per kondisi " + "─" * 29)
    print(table_token_consumption(consumption))
    print("\n% ── Tabel: tugas per anggaran token " + "─" * 31)
    print(table_budget(consumption))


if __name__ == "__main__":
    main()
