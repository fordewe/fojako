"""Judge-vs-human agreement for the RQ2 validation sample.

Section 4.5 of the paper makes two claims that only this script can settle:
the LLM judge agrees with a human reader well enough to stand in for one, and
the human reader agrees with themselves a week later. Both are reported as
quadratic-weighted Cohen's kappa, overall and per task category.

The weighting matters. Scores are ordinal on a 1-5 scale, so a judge that says
5 where the human says 4 has made a smaller error than one that says 5 where
the human says 1. Quadratic weights encode that, and linear or unweighted
kappa would punish the small disagreement as hard as the large one.

Kappa is undefined, not zero, when the two raters use a single value in common:
the expected disagreement goes to zero and the ratio has no meaning. That case
returns ``None`` so a caller cannot mistake a degenerate sample for a bad
judge.

    uv run python evaluation/analyze_kappa.py

    uv run python evaluation/analyze_kappa.py \
        --rescore evaluation/results/rq2_human_rescore.csv

The judge scores live in two files. ``rq2_judge_scores.csv`` holds the 142
responses judged for RQ2. ``rq2_judge_validation_extra.csv`` holds the eight
responses that the validation sample drew outside that set, judged after the
human pass with the same model, prompt and rubric. ``--judge`` reads both by
default and refuses a response that appears in more than one file, since
silently keeping either score would change kappa.

The rescore file is the same schema as the human file, collected in a second
sitting. ``validate_ui.py`` skips a response it has already scored in the same
output file, so the second pass has to be written to its own path.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

FLOOR = 0.61
DIMENSIONS = ("accuracy", "completeness")
KEY = ("task_id", "condition", "model")
DEFAULT_JUDGE = (
    "evaluation/results/rq2_judge_scores.csv",
    "evaluation/results/rq2_judge_validation_extra.csv",
)


def weighted_kappa(rater_a, rater_b, min_rating=1, max_rating=5):
    """Quadratic-weighted Cohen's kappa for two ordinal rating lists.

    Returns ``None`` when the ratings cannot support a kappa: no pairs, or
    zero expected disagreement.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError("rater_a and rater_b must have the same length")
    n = len(rater_a)
    if n == 0:
        return None

    span = max_rating - min_rating + 1
    observed = [[0] * span for _ in range(span)]
    for a, b in zip(rater_a, rater_b):
        observed[a - min_rating][b - min_rating] += 1

    row_sums = [sum(row) for row in observed]
    col_sums = [sum(observed[i][j] for i in range(span)) for j in range(span)]

    numerator = 0.0
    denominator = 0.0
    for i in range(span):
        for j in range(span):
            weight = (i - j) ** 2 / (span - 1) ** 2
            numerator += weight * observed[i][j]
            denominator += weight * row_sums[i] * col_sums[j] / n

    if denominator == 0:
        return None
    return 1.0 - numerator / denominator


def agreement(rows_a, rows_b):
    """Kappa per dimension over the rows both raters scored.

    Rows are matched on ``(task_id, condition, model)``. A row scored by one
    rater and not the other is dropped rather than imputed. The two directions
    are counted separately because they mean different things: a judge row the
    human never scored is expected, since the human sample is a subset, while a
    human row with no judge row behind it is a broken join and should be zero.
    """
    b_index = {tuple(r[k] for k in KEY): r for r in rows_b}
    a_index = {tuple(r[k] for k in KEY): r for r in rows_a}
    pairs = [(r, b_index[tuple(r[k] for k in KEY)]) for r in rows_a
             if tuple(r[k] for k in KEY) in b_index]
    human_only = sum(1 for k in b_index if k not in a_index)

    out = {}
    for dimension in DIMENSIONS:
        column = f"{dimension}_score"
        a = [int(r[column]) for r, _ in pairs]
        b = [int(s[column]) for _, s in pairs]
        out[dimension] = {"kappa": weighted_kappa(a, b), "n": len(pairs)}
    out["n_matched"] = len(pairs)
    out["n_judge_only"] = len(a_index) - len(pairs)
    out["n_human_only"] = human_only
    return out


def by_category(rows_a, rows_b):
    """Same agreement, split by the category recorded on each row."""
    groups_a = defaultdict(list)
    for r in rows_a:
        groups_a[r["category"]].append(r)

    out = {}
    for category in sorted(groups_a):
        out[category] = agreement(groups_a[category], rows_b)
    return out


def load_scores(path):
    with open(path) as f:
        return [r for r in csv.DictReader(f)]


def load_judge_files(paths):
    """Judge rows from several score files, read as one table.

    Raises ``ValueError`` when one response is scored in two files: which
    score should count is a decision, not something a join can make.
    """
    rows = []
    source = {}
    for path in paths:
        for r in load_scores(path):
            key = tuple(r[k] for k in KEY)
            if key in source:
                raise ValueError(
                    f"{key} is scored in both {source[key]} and {path}"
                )
            source[key] = path
            rows.append(r)
    return rows


def verdict(kappa):
    if kappa is None:
        return "undefined"
    if kappa < FLOOR:
        return "below floor"
    return "substantial or better"


def report(title, result):
    print(f"\n% ── {title} " + "─" * max(0, 56 - len(title)))
    print(
        f"% matched {result['n_matched']}, judge-only {result['n_judge_only']}, "
        f"human-only {result['n_human_only']}"
    )
    if result["n_human_only"]:
        print("% PERINGATAN: ada skor manusia tanpa pasangan juri; periksa kunci.")
    for dimension in DIMENSIONS:
        cell = result[dimension]
        kappa = cell["kappa"]
        shown = "undef." if kappa is None else f"{kappa:.3f}"
        print(f"  {dimension:<13} kappa = {shown:<7} n = {cell['n']:<3} "
              f"{verdict(kappa)}")


def report_categories(result):
    print("\n% ── Per kategori " + "─" * 41)
    for category, cell in result.items():
        parts = []
        for dimension in DIMENSIONS:
            kappa = cell[dimension]["kappa"]
            shown = "undef." if kappa is None else f"{kappa:.3f}"
            parts.append(f"{dimension}={shown}")
        print(f"  {category:<15} n = {cell['n_matched']:<3} "
              f"{'  '.join(parts)}")


def main():
    parser = argparse.ArgumentParser(description="Judge-vs-human agreement")
    parser.add_argument(
        "--judge", nargs="+", default=list(DEFAULT_JUDGE),
        help="one or more judge score files, read as one table",
    )
    parser.add_argument(
        "--human", default="evaluation/results/rq2_human_scores.csv"
    )
    parser.add_argument(
        "--rescore", default=None,
        help="second human pass for intra-rater agreement",
    )
    parser.add_argument("--json", default=None, help="write the numbers here")
    args = parser.parse_args()

    judge_paths = [Path(p) for p in args.judge]
    missing = [str(p) for p in judge_paths if not p.exists()]
    if missing:
        raise SystemExit(f"Berkas juri tidak ditemukan: {', '.join(missing)}")
    try:
        judge_rows = load_judge_files(judge_paths)
    except ValueError as e:
        raise SystemExit(f"Skor juri ganda: {e}")
    human_path = Path(args.human)
    if not human_path.exists():
        raise SystemExit(
            f"{human_path} belum ada. Jalankan validate_ui.py sampai selesai."
        )
    human_rows = load_scores(human_path)

    overall = agreement(judge_rows, human_rows)
    categories = by_category(judge_rows, human_rows)

    report("Juri vs manusia, seluruh sampel", overall)
    report_categories(categories)

    payload = {"overall": overall, "by_category": categories, "floor": FLOOR}

    if args.rescore:
        rescore_path = Path(args.rescore)
        if not rescore_path.exists():
            raise SystemExit(f"{rescore_path} tidak ditemukan.")
        intra = agreement(human_rows, load_scores(rescore_path))
        report("Intra-penilai, sampel yang dinilai ulang", intra)
        payload["intra_rater"] = intra

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nDitulis ke {args.json}")


if __name__ == "__main__":
    main()
