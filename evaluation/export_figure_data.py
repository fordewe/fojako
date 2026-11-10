"""Write the three CSVs behind the paper's RQ2 and RQ3 figures.

The paper repository renders its RQ2 and RQ3 figures from three small tables
and reconciles them with the manuscript text (``make_rq2_rq3_figures.py
--check``). Until this script, two of those tables were typed by hand from
the manuscript, so a check that passed only proved that two copies of the
same numbers agreed. This script derives all three from the evaluation data:

  * ``rq2_aggregates.csv``: mean and SD per rater x category x condition x
    metric. Judge cells cover the 45 tasks judged under all three conditions
    (15 per category), as ``analyze_rq2.py`` does; human cells cover the 36
    validation scores (4 per cell) and leave SD blank, because the figure
    draws no error bar on four scores.
  * ``rq3_token_totals.csv``: mean input, output and total tokens per
    condition over the same 45 tasks, as ``analyze_rq3.py`` prints them.
  * ``rq3_per_task.csv``: one row per task with the context each condition
    built, plus how many files Hybrid requested and how many it was sent.

Hybrid's input is rebuilt from the module files on disk, so the dataset
checkouts named in ``datasets.json`` must be present.

    uv run python evaluation/export_figure_data.py \\
        --output-dir ../latex-paper/2026/maret/j_ucs_structure_code_summarization/figures
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_rq2 import (  # noqa: E402
    CATEGORIES,
    CONDITIONS,
    LABEL,
    complete_tasks,
    load_scores,
    mean,
    stdev,
)
from analyze_rq3 import fill_hybrid_context_tokens, token_consumption  # noqa: E402

METRICS = [("accuracy_score", "Accuracy"), ("completeness_score", "Completeness")]
AGGREGATE_FIELDS = ["rater", "category", "condition", "metric", "mean", "sd", "n"]
TOTAL_FIELDS = ["condition", "input", "output", "total"]
PER_TASK_FIELDS = [
    "task_id", "project", "category", "full", "summary", "hybrid",
    "files_requested", "files_sent",
]


def aggregate_rows(rater: str, rows: list[dict], with_sd: bool) -> list[dict]:
    """One row per category x metric x condition, in the figure's order.

    An empty cell raises rather than printing ``nan``: the figure would draw
    nothing there and the check would compare against a number that is not
    in the manuscript.
    """
    out = []
    for category in CATEGORIES:
        for field, metric in METRICS:
            for condition in CONDITIONS:
                scores = [
                    int(r[field]) for r in rows
                    if r["category"] == category and r["condition"] == condition
                ]
                if not scores:
                    raise ValueError(
                        f"{rater}: no {metric} scores for {category}/{condition}"
                    )
                out.append({
                    "rater": rater,
                    "category": category.capitalize(),
                    "condition": LABEL[condition],
                    "metric": metric,
                    "mean": f"{mean(scores):.2f}",
                    "sd": f"{stdev(scores):.2f}" if with_sd else "",
                    "n": len(scores),
                })
    return out


def figure_aggregates(judge: list[dict], human: list[dict] | None) -> list[dict]:
    """Judge rows, then human rows unless ``human`` is None.

    None leaves the human markers out of the figure. The clean rerun's human
    sample is scored after the judge, and first-run human means beside
    clean-run judge bars would compare different responses.
    """
    rows = aggregate_rows("judge", judge, with_sd=True)
    if human is not None:
        rows += aggregate_rows("human", human, with_sd=False)
    return rows


def total_rows(consumption: dict[str, dict[str, float]]) -> list[dict]:
    """Condition means rounded to whole tokens, as the manuscript prints them."""
    return [
        {
            "condition": LABEL[c],
            "input": round(consumption[c]["input"]),
            "output": round(consumption[c]["output"]),
            "total": round(consumption[c]["total"]),
        }
        for c in CONDITIONS
    ]


def per_task_rows(
    responses: list[dict], done: set[str], hybrid_context: dict[str, int],
) -> list[dict]:
    """Per-task context under each condition, sorted by task id.

    Raises ``ValueError`` naming any task in ``done`` that lacks a condition,
    since a half-filled row would plot as a task that cost nothing.
    """
    rows: dict[str, dict] = {}
    for r in responses:
        if r["task_id"] not in done:
            continue
        row = rows.setdefault(r["task_id"], {
            "task_id": r["task_id"], "project": r["project"],
            "category": r["category"],
        })
        if r["condition"] == "full":
            row["full"] = r["context_tokens_full"]
        elif r["condition"] == "summary":
            row["summary"] = r["context_tokens_summary"]
        elif r["condition"] == "hybrid":
            row["hybrid"] = hybrid_context[r["task_id"]]
            row["files_requested"] = len(r["files_requested"])
            row["files_sent"] = len(r["files_resolved"])

    incomplete = sorted(t for t, row in rows.items() if set(PER_TASK_FIELDS) - row.keys())
    if incomplete:
        raise ValueError(f"tasks missing a condition: {', '.join(incomplete)}")
    return [rows[t] for t in sorted(rows)]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """CSV with LF line ends; the csv module defaults to CRLF."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Write the RQ2/RQ3 figure CSVs")
    parser.add_argument("--scores", default="evaluation/results/rq2_judge_scores.csv")
    parser.add_argument("--subset", default="evaluation/results/rq2_judge_subset.json")
    parser.add_argument("--responses", default="evaluation/results/rq2_responses.json")
    parser.add_argument("--human", default="evaluation/results/rq2_human_scores.csv")
    parser.add_argument(
        "--no-human", action="store_true",
        help="leave human means out of rq2_aggregates.csv (and the figure)",
    )
    parser.add_argument("--datasets", default="evaluation/datasets.json")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    needed = [args.scores, args.subset, args.responses, args.datasets]
    if not args.no_human:
        needed.append(args.human)
    for path in needed:
        if not Path(path).exists():
            raise SystemExit(f"Tidak ditemukan: {path}")
    out_dir = Path(args.output_dir)
    if not out_dir.is_dir():
        raise SystemExit(f"Direktori keluaran tidak ada: {out_dir}")

    keep = {r["task_id"] for r in json.loads(Path(args.subset).read_text())}
    judge = load_scores(Path(args.scores), keep)
    done = complete_tasks(judge)
    balanced = [r for r in judge if r["task_id"] in done]
    human = None
    if not args.no_human:
        with open(args.human) as f:
            human = list(csv.DictReader(f))

    responses = json.loads(Path(args.responses).read_text())
    project_paths = {
        p["name"]: Path(p["path"]).expanduser()
        for p in json.loads(Path(args.datasets).read_text())["projects"]
    }
    hybrid = fill_hybrid_context_tokens(
        [r for r in responses if r["task_id"] in done], project_paths,
    )

    try:
        aggregates = figure_aggregates(balanced, human)
        per_task = per_task_rows(responses, done, hybrid)
    except ValueError as e:
        raise SystemExit(str(e))
    totals = total_rows(token_consumption(responses, done, hybrid))

    write_csv(out_dir / "rq2_aggregates.csv", AGGREGATE_FIELDS, aggregates)
    write_csv(out_dir / "rq3_token_totals.csv", TOTAL_FIELDS, totals)
    write_csv(out_dir / "rq3_per_task.csv", PER_TASK_FIELDS, per_task)
    print(
        f"{len(done)} tugas lengkap; {len(aggregates)} sel RQ2, "
        f"{len(per_task)} baris per tugas -> {out_dir}"
    )


if __name__ == "__main__":
    main()
