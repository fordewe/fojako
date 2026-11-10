"""Equivalence tests and paired effect sizes for RQ2.

The plan was fixed before any of this was computed (paper RUN-LOG 42, commit
9ecd6d9 in the paper repository):

  * TOST for the two equivalence claims the paper makes: Hybrid vs Full on all
    45 paired tasks, and Summary vs Full on navigation, each on accuracy and
    completeness. Margins of 0.5 rubric points (primary) and 0.25
    (sensitivity). Each TOST is two one-sided Wilcoxon signed-rank tests on
    the shifted paired differences, matching the main analysis; its p-value
    is the larger of the two, and the four primary p-values are Holm-adjusted.
  * A 90% percentile bootstrap interval of the mean paired difference
    (10,000 resamples, seed 42) beside each TOST.
  * The matched-pairs rank-biserial correlation beside Cliff's delta for the
    six condition comparisons.
  * Wilcoxon tests for the three condition pairs within each task category,
    Holm-adjusted per (category, metric).

    uv run python evaluation/analyze_equivalence.py \\
        --scores evaluation/results/rq2_clean_judge_scores.csv
"""

import argparse
import json
import random
import sys
from pathlib import Path

from scipy.stats import rankdata, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_rq2 import (  # noqa: E402
    CATEGORIES,
    DIMENSIONS,
    cliffs_delta,
    complete_tasks,
    holm,
    load_scores,
    mean,
    paired,
)

PRIMARY_MARGIN = 0.5
SENSITIVITY_MARGIN = 0.25
ALPHA = 0.05
BOOTSTRAP_LEVEL = 0.90
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
# (condition compared with Full, category or None for all tasks)
CLAIMS = (("hybrid", None), ("summary", "navigation"))
PAIRS = (("full", "summary"), ("full", "hybrid"), ("summary", "hybrid"))


def differences(rows: list[dict], field: str, condition: str,
                category: str | None = None) -> list[float]:
    """Per-task ``condition - full`` on one dimension, tasks sorted by id."""
    subset = [r for r in rows if category is None or r["category"] == category]
    xs, ys = paired(subset, field, condition, "full")
    return [x - y for x, y in zip(xs, ys)]


def tost(d: list[float], margin: float) -> dict:
    """Two one-sided Wilcoxon signed-rank tests of |location| < margin.

    The lower test asks whether the differences sit above ``-margin``, the
    upper whether they sit below ``+margin``. Equivalence needs both, so the
    TOST p-value is the larger of the two.
    """
    lower = wilcoxon([x + margin for x in d], alternative="greater").pvalue
    upper = wilcoxon([x - margin for x in d], alternative="less").pvalue
    return {"p_lower": float(lower), "p_upper": float(upper),
            "p": float(max(lower, upper))}


def bootstrap_ci(d: list[float], level: float = BOOTSTRAP_LEVEL,
                 resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float, float]:
    """Mean of ``d`` and its percentile bootstrap interval."""
    rng = random.Random(seed)
    n = len(d)
    means = sorted(sum(rng.choice(d) for _ in range(n)) / n
                   for _ in range(resamples))
    tail = (1 - level) / 2
    low = means[int(tail * resamples)]
    high = means[int((1 - tail) * resamples) - 1]
    return float(mean(d)), float(low), float(high)


def rank_biserial(d: list[float]) -> float | None:
    """Matched-pairs rank-biserial correlation (Kerby's simple difference).

    Zero differences are dropped, as the Wilcoxon test drops them. ``None``
    when every difference is zero.
    """
    nonzero = [x for x in d if x != 0]
    if not nonzero:
        return None
    ranks = rankdata([abs(x) for x in nonzero])
    positive = sum(r for r, x in zip(ranks, nonzero) if x > 0)
    negative = sum(r for r, x in zip(ranks, nonzero) if x < 0)
    return float((positive - negative) / (positive + negative))


def category_tests(rows: list[dict]) -> list[dict]:
    """Wilcoxon for each condition pair within each category, Holm per family.

    A family is one (category, metric): three tests. A pair whose differences
    are all zero gets NaN rather than a p-value, since nothing was tested.
    """
    results = []
    for category in CATEGORIES:
        subset = [r for r in rows if r["category"] == category]
        for field, metric in DIMENSIONS:
            family = []
            for a, b in PAIRS:
                xs, ys = paired(subset, field, a, b)
                d = [x - y for x, y in zip(xs, ys)]
                entry = {"category": category, "metric": metric, "a": a, "b": b,
                         "n": len(d), "mean_a": mean(xs), "mean_b": mean(ys),
                         "r_rb": rank_biserial(d)}
                if any(d):
                    test = wilcoxon(xs, ys, zero_method="wilcox")
                    entry.update(statistic=float(test.statistic), p=float(test.pvalue))
                else:
                    entry.update(statistic=float("nan"), p=float("nan"))
                family.append(entry)
            tested = [e for e in family if e["p"] == e["p"]]
            for e, p in zip(tested, holm([e["p"] for e in tested])):
                e["p_holm"] = p
            for e in family:
                e.setdefault("p_holm", float("nan"))
            results += family
    return results


def equivalence(rows: list[dict], margin: float) -> list[dict]:
    """TOST and bootstrap interval for each claim and dimension, Holm across."""
    results = []
    for condition, category in CLAIMS:
        for field, metric in DIMENSIONS:
            d = differences(rows, field, condition, category)
            mean_d, low, high = bootstrap_ci(d)
            results.append({"condition": condition, "category": category or "all",
                            "metric": metric, "n": len(d), "margin": margin,
                            "mean_diff": mean_d, "ci90": [low, high],
                            **tost(d, margin)})
    for r, p in zip(results, holm([r["p"] for r in results])):
        r["p_holm"] = p
        r["equivalent"] = p < ALPHA
    return results


def effect_sizes(rows: list[dict]) -> list[dict]:
    """Cliff's delta and matched-pairs rank-biserial for the six comparisons."""
    out = []
    for field, metric in DIMENSIONS:
        for a, b in PAIRS:
            xs, ys = paired(rows, field, a, b)
            delta, reading = cliffs_delta(xs, ys)
            out.append({"metric": metric, "a": a, "b": b, "cliffs_delta": delta,
                        "reading": reading,
                        "r_rb": rank_biserial([x - y for x, y in zip(xs, ys)])})
    return out


def show(value: float | None, digits: int = 3) -> str:
    if value is None or value != value:
        return "--"
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description="RQ2 equivalence tests")
    parser.add_argument("--scores", default="evaluation/results/rq2_clean_judge_scores.csv")
    parser.add_argument("--subset", default="evaluation/results/rq2_judge_subset.json")
    parser.add_argument("--json", default=None, help="write the numbers here")
    args = parser.parse_args()

    keep = {r["task_id"] for r in json.loads(Path(args.subset).read_text())}
    rows = load_scores(Path(args.scores), keep)
    done = complete_tasks(rows)
    rows = [r for r in rows if r["task_id"] in done]
    print(f"{len(done)} tugas lengkap, {len(rows)} skor")

    payload = {}
    for name, margin in (("primary", PRIMARY_MARGIN), ("sensitivity", SENSITIVITY_MARGIN)):
        payload[name] = equivalence(rows, margin)
        print(f"\n== TOST, margin +/-{margin} ({name}; Holm atas 4)")
        for r in payload[name]:
            print(f"  {r['condition']:<7} vs full  {r['category']:<10} {r['metric']:<12} "
                  f"n={r['n']:<2} mean d={r['mean_diff']:+.3f}  "
                  f"90% CI [{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]  "
                  f"p_low={show(r['p_lower'], 4)} p_up={show(r['p_upper'], 4)}  "
                  f"p={show(r['p'], 4)} Holm={show(r['p_holm'], 4)}  "
                  f"{'SETARA' if r['equivalent'] else 'belum terbukti setara'}")

    payload["effect_sizes"] = effect_sizes(rows)
    print("\n== Effect size, 45 tugas berpasangan")
    for e in payload["effect_sizes"]:
        print(f"  {e['metric']:<12} {e['a']}-{e['b']:<8} Cliff's delta {e['cliffs_delta']:+.3f} "
              f"({e['reading']})  rank-biserial {show(e['r_rb'])}")

    payload["categories"] = category_tests(rows)
    print("\n== Wilcoxon per kategori (Holm per kategori-metrik, 3 uji)")
    for e in payload["categories"]:
        print(f"  {e['category']:<13} {e['metric']:<12} {e['a']}-{e['b']:<8} "
              f"{e['mean_a']:.2f} vs {e['mean_b']:.2f}  W={show(e['statistic'], 1)} "
              f"p={show(e['p'], 4)} Holm={show(e['p_holm'], 4)} r_rb={show(e['r_rb'])}")

    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nDitulis ke {args.json}")


if __name__ == "__main__":
    main()
