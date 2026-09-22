"""Render the RQ1 figures from rq1_metrics.csv.

Two figures, each answering a question Finding 1 raises and a table cannot:

  * Spread per project. The table reports one aggregate number per project, so
    it cannot show that Tivi is not uniformly worse but has a different
    distribution. A box plot does.
  * File size against reduction. Finding 1 reports Spearman 0.468 and a
    monotone quartile progression; the scatter shows the shape behind the
    coefficient, including where compression fails.

Serif at 9pt to sit inside a J.UCS page without looking pasted in.

Usage:
    uv run python evaluation/make_figures.py \\
        --metrics evaluation/results/rq1_metrics.csv \\
        --output-dir ../latex-paper/2026/maret/j_ucs_structure_code_summarization/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

LABEL = {
    "sunflower": "Sunflower",
    "architecture-components-samples": "Arch\nComponents",
    "nowinandroid": "Now in\nAndroid",
    "compose-samples": "Compose\nSamples",
    "tivi": "Tivi",
}
# Ordered by file count, matching Table 2, so a reader moving between them does
# not have to re-sort mentally.
ORDER = [
    "sunflower", "architecture-components-samples", "nowinandroid",
    "compose-samples", "tivi",
]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })


def box_by_project(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    data = [df.loc[df.project == p, "token_reduction_rate"] * 100 for p in ORDER]

    bp = ax.boxplot(
        data, tick_labels=[LABEL[p] for p in ORDER], widths=0.55,
        showfliers=True, flierprops={"marker": ".", "markersize": 2.5,
                                     "markerfacecolor": "0.4",
                                     "markeredgecolor": "0.4"},
        medianprops={"color": "black", "linewidth": 1.3},
        boxprops={"color": "0.25"}, whiskerprops={"color": "0.25"},
        capprops={"color": "0.25"},
    )
    for patch in bp["boxes"]:
        patch.set_linewidth(1.0)

    # The overall rate goes in as a reference line only. Labelling it inside the
    # axes collides with whichever box happens to sit near 63%, so the caption
    # names it instead.
    overall = (1 - df.summary_tokens.sum() / df.original_tokens.sum()) * 100
    ax.axhline(overall, linestyle="--", linewidth=0.8, color="0.45")

    ax.set_ylabel("Token reduction (%)")
    ax.set_ylim(-40, 105)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def scatter_size_vs_reduction(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 2.8))

    for lang, marker, color, size in [
        ("kotlin", "o", "0.55", 5), ("java", "^", "0.1", 11),
    ]:
        g = df[df.language == lang]
        ax.scatter(
            g.original_tokens, g.token_reduction_rate * 100, s=size, marker=marker,
            facecolors="none" if lang == "kotlin" else color,
            edgecolors=color, linewidths=0.5, alpha=0.55 if lang == "kotlin" else 0.9,
            label=f"Kotlin (n={len(g)})" if lang == "kotlin" else f"Java (n={len(g)})",
        )

    # Aggregate reduction per size quartile: the trend Finding 1 quotes, drawn
    # over the cloud so the coefficient has a visible shape.
    q = pd.qcut(df.original_tokens, 4)
    line = df.groupby(q, observed=True).apply(
        lambda g: pd.Series({
            "x": g.original_tokens.median(),
            "y": (1 - g.summary_tokens.sum() / g.original_tokens.sum()) * 100,
        }),
        include_groups=False,
    )
    ax.plot(line.x, line.y, color="black", linewidth=1.2, marker="s",
            markersize=3.5, label="quartile aggregate", zorder=5)

    ax.axhline(0, linewidth=0.7, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("File size (tokens, log scale)")
    ax.set_ylabel("Token reduction (%)")
    ax.set_ylim(-40, 105)
    ax.grid(linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    parser = argparse.ArgumentParser(description="Render the RQ1 figures")
    parser.add_argument("--metrics", default="evaluation/results/rq1_metrics.csv")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    style()
    print(f"{len(df)} berkas -> {out_dir}")
    box_by_project(df, out_dir / "rq1-reduction-by-project.pdf")
    scatter_size_vs_reduction(df, out_dir / "rq1-size-vs-reduction.pdf")


if __name__ == "__main__":
    main()
