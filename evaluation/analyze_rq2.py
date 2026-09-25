"""Turn the judge's scores into the numbers Section 5.2 reports.

Reads ``rq2_judge_scores.csv`` and prints the two RQ2 tables plus the tests
behind them, in LaTeX rows ready to paste.

Three things here are decisions, not defaults:

*Pairing.* Every task is answered under all three conditions, so Full against
Summary is a paired comparison on the same question. Wilcoxon signed-rank
matches that; Mann-Whitney would throw the pairing away and lose most of the
power on a five-point scale.

*Ties.* A five-point scale on 45 tasks produces many equal pairs. ``scipy``
drops them, which is the standard treatment, so the effective n is smaller than
the number of tasks and both are printed. A comparison whose pairs are almost
all ties is reported as such rather than as a p-value nobody can interpret.

*Multiplicity.* Three condition pairs times two dimensions is six tests on one
dataset. Holm-Bonferroni across the whole family, which Section 4 pre-registers.

    uv run python evaluation/analyze_rq2.py
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CONDITIONS = ["full", "summary", "hybrid"]
LABEL = {"full": "Full", "summary": "Summary", "hybrid": "Hybrid"}
CATEGORIES = ["navigation", "understanding", "modification"]
DIMENSIONS = [("accuracy_score", "accuracy"), ("completeness_score", "completeness")]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: list[float]) -> float:
    """Sample standard deviation, which is what a table of means reports."""
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def cliffs_delta(a: list[float], b: list[float]) -> tuple[float, str]:
    """Non-parametric effect size, with the Romano et al. reading of it.

    Reported next to every p-value because on a five-point scale a difference
    can reach significance while being too small for a developer to notice.
    """
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    delta = (gt - lt) / (len(a) * len(b))

    size = abs(delta)
    if size < 0.147:
        reading = "negligible"
    elif size < 0.330:
        reading = "small"
    elif size < 0.474:
        reading = "medium"
    else:
        reading = "large"
    return delta, reading


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    n = len(pvalues)
    order = sorted(range(n), key=lambda i: pvalues[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def load_scores(path: Path, keep: set[str] | None) -> list[dict]:
    """Scores for the tasks that were actually in the judging queue.

    Seven rows were graded before the queue was cut to 45 tasks. They are left
    in the file as a record of what was spent, and excluded here, because a
    condition that carries them and a condition that does not are no longer
    balanced.
    """
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if keep is not None and row["task_id"] not in keep:
                continue
            row["accuracy_score"] = int(row["accuracy_score"])
            row["completeness_score"] = int(row["completeness_score"])
            rows.append(row)
    return rows


def complete_tasks(rows: list[dict]) -> set[str]:
    """Tasks graded under all three conditions.

    A task missing one condition cannot enter a paired test, and letting it
    into the means but not the tests would make the table and the statistics
    describe different samples.
    """
    seen: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        seen[r["task_id"]].add(r["condition"])
    return {t for t, cs in seen.items() if set(CONDITIONS) <= cs}


def by_condition(rows: list[dict], field: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {c: [] for c in CONDITIONS}
    for r in rows:
        out[r["condition"]].append(r[field])
    return out


def paired(rows: list[dict], field: str, a: str, b: str) -> tuple[list[int], list[int]]:
    index = {(r["task_id"], r["condition"]): r[field] for r in rows}
    tasks = sorted({r["task_id"] for r in rows})
    xs = [index[(t, a)] for t in tasks if (t, a) in index and (t, b) in index]
    ys = [index[(t, b)] for t in tasks if (t, a) in index and (t, b) in index]
    return xs, ys


def table_quality(rows: list[dict], tokens: dict[str, float]) -> str:
    lines = []
    for c in CONDITIONS:
        acc = [r["accuracy_score"] for r in rows if r["condition"] == c]
        comp = [r["completeness_score"] for r in rows if r["condition"] == c]
        lines.append(
            f"        {LABEL[c]:<7} & ${mean(acc):.2f} \\pm {stdev(acc):.2f}$ "
            f"& ${mean(comp):.2f} \\pm {stdev(comp):.2f}$ "
            f"& {tokens.get(c, float('nan')):,.0f} \\\\ \\hline"
        )
    return "\n".join(lines)


def table_by_category(rows: list[dict]) -> str:
    lines = []
    for category in CATEGORIES:
        for c in CONDITIONS:
            sub = [
                r for r in rows
                if r["condition"] == c and r["category"] == category
            ]
            acc = [r["accuracy_score"] for r in sub]
            comp = [r["completeness_score"] for r in sub]
            lines.append(
                f"        {category.capitalize():<13} & {LABEL[c]:<7} "
                f"& ${mean(acc):.2f} \\pm {stdev(acc):.2f}$ "
                f"& ${mean(comp):.2f} \\pm {stdev(comp):.2f}$ \\\\ \\hline"
            )
    return "\n".join(lines)


def run_tests(rows: list[dict]) -> list[dict]:
    pairs = [("full", "summary"), ("full", "hybrid"), ("summary", "hybrid")]
    results = []

    for field, name in DIMENSIONS:
        for a, b in pairs:
            xs, ys = paired(rows, field, a, b)
            nonzero = sum(1 for x, y in zip(xs, ys) if x != y)
            entry = {
                "dimension": name, "a": a, "b": b,
                "n_pairs": len(xs), "n_nonzero": nonzero,
                "mean_a": mean(xs), "mean_b": mean(ys),
            }
            if nonzero == 0:
                # Every pair is a tie. Wilcoxon has nothing to rank, and
                # reporting a p-value here would suggest a test happened.
                entry.update(statistic=float("nan"), p=float("nan"))
            else:
                stat, p = wilcoxon(xs, ys, zero_method="wilcox")
                entry.update(statistic=float(stat), p=float(p))
            entry["delta"], entry["reading"] = cliffs_delta(xs, ys)
            results.append(entry)

    testable = [r for r in results if r["n_nonzero"] > 0]
    adjusted = holm([r["p"] for r in testable])
    for r, p in zip(testable, adjusted):
        r["p_holm"] = p
    for r in results:
        r.setdefault("p_holm", float("nan"))
    return results


def main():
    parser = argparse.ArgumentParser(description="Aggregate RQ2 judge scores")
    parser.add_argument("--scores", default="evaluation/results/rq2_judge_scores.csv")
    parser.add_argument("--subset", default="evaluation/results/rq2_judge_subset.json")
    parser.add_argument("--responses", default="evaluation/results/rq2_responses.json")
    args = parser.parse_args()

    keep = None
    subset = Path(args.subset)
    if subset.exists():
        keep = {r["task_id"] for r in json.load(open(subset))}

    rows = load_scores(Path(args.scores), keep)
    done = complete_tasks(rows)
    balanced = [r for r in rows if r["task_id"] in done]

    logger.info(
        "%d skor dibaca, %d tugas lengkap di tiga kondisi, %d baris dipakai",
        len(rows), len(done), len(balanced),
    )
    partial = {r["task_id"] for r in rows} - done
    if partial:
        logger.warning(
            "%d tugas belum lengkap dan dikeluarkan: %s",
            len(partial), ", ".join(sorted(partial)[:6]),
        )
    if not balanced:
        logger.error("Belum ada tugas yang lengkap. Tabel tidak bisa dihitung.")
        return

    # Context we built and counted locally, not what the provider billed.
    responses = json.load(open(args.responses))
    ctx = {r["task_id"]: r for r in responses if r["condition"] == "full"}
    tokens = {
        "full": mean([ctx[t]["context_tokens_full"] for t in done if t in ctx]),
        "summary": mean([ctx[t]["context_tokens_summary"] for t in done if t in ctx]),
    }
    tokens["hybrid"] = float("nan")  # two turns; analyze_rq3.py resolves it

    print("\n% ── Tabel: skor kualitas per kondisi " + "─" * 30)
    print(table_quality(balanced, tokens))
    print("\n% ── Tabel: skor kualitas per kategori " + "─" * 29)
    print(table_by_category(balanced))

    print("\n% ── Uji statistik " + "─" * 48)
    for r in run_tests(balanced):
        p = "tak terdefinisi (semua pasangan seri)" if r["n_nonzero"] == 0 else (
            f"p = {r['p']:.4g}, Holm p = {r['p_holm']:.4g}"
        )
        stat = "---" if r["n_nonzero"] == 0 else f"{r['statistic']:.1f}"
        print(
            f"  {r['dimension']:<12} {LABEL[r['a']]:>7} vs {LABEL[r['b']]:<7} "
            f"mean {r['mean_a']:.2f} vs {r['mean_b']:.2f}  "
            f"n = {r['n_pairs']} ({r['n_nonzero']} tidak seri)  "
            f"W = {stat}  {p}  "
            f"delta = {r['delta']:+.3f} ({r['reading']})"
        )

    print("\n% Kappa penilai-vs-manusia menunggu validate_ui.py.")


if __name__ == "__main__":
    main()
