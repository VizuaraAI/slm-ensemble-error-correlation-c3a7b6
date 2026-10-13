"""Regenerate the paper's Figure 2 from real analysis output.

Usage:
    python -m src.plots --analysis runs/default/analysis --out figures/

Reads table1_main.csv and scaling.csv, writes fig2_results.png with the same
layout as the placeholder figure in the draft, so dropping the real numbers
into the paper is a file copy.
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK = "#1b1f23"
MUTED = "#4a5157"
EDGE = "#99a1a9"
NOMINAL = "#d7dce1"
# Ordered light to dark, so the figure still reads when printed in greyscale.
RAMP = ["#a8c0d8", "#6f93b8", "#426d96", "#1f3f63"]
MARKERS = ["o", "s", "^", "D"]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color(EDGE)
    ax.tick_params(labelsize=8.0, length=3, width=0.8, color=EDGE, labelcolor=MUTED)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 2 from analysis output.")
    parser.add_argument("--analysis", default="runs/default/analysis")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    analysis = Path(args.analysis)
    main_rows = read_csv(analysis / "table1_main.csv")
    scaling_rows = read_csv(analysis / "scaling.csv")
    if not main_rows:
        raise SystemExit("table1_main.csv is empty; run `python -m src.analyze` first")

    labels = [r["ensemble_construction"] for r in main_rows]
    ess = [float(r["ess"]) for r in main_rows]
    nominal = [float(r["n_members"]) for r in main_rows]
    colours = [RAMP[i % len(RAMP)] for i in range(len(labels))]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK})
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.6, 4.4), gridspec_kw={"wspace": 0.26})
    fig.suptitle("Nominal ensemble size versus effective ensemble size",
                 fontsize=11.0, weight="bold", y=0.975)

    # ---- panel (a) ---------------------------------------------------------
    x = np.arange(len(labels))
    width = 0.36
    bars_nom = axa.bar(x - width / 2, nominal, width, color=NOMINAL,
                       edgecolor=EDGE, linewidth=0.8, label="nominal $N$")
    bars_ess = axa.bar(x + width / 2, ess, width, color=colours,
                       edgecolor="none", label="effective ESS")
    top = max(max(nominal), max(ess))
    for bar, value in list(zip(bars_nom, nominal)) + list(zip(bars_ess, ess)):
        axa.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
                 f"{value:.1f}", ha="center", va="bottom", fontsize=7.6, color=MUTED)
    axa.set_xticks(x)
    # Wrap on width, not on commas: strategy names vary in shape and long
    # unwrapped labels collide with their neighbours.
    axa.set_xticklabels([textwrap.fill(l, 18) for l in labels], fontsize=7.6, linespacing=1.5)
    axa.set_ylim(0, top * 1.16)
    axa.set_ylabel("ensemble size", fontsize=9.0)
    axa.legend(frameon=False, fontsize=8.2, ncol=2, loc="lower left",
               bbox_to_anchor=(-0.015, 1.0), handlelength=1.4, borderpad=0.0)
    tidy(axa)

    # ---- panel (b) ---------------------------------------------------------
    by_strategy: dict[str, list[tuple[int, float]]] = {}
    for row in scaling_rows:
        by_strategy.setdefault(row["ensemble_construction"], []).append(
            (int(row["n_members"]), float(row["mean_accuracy_pct"]))
        )
    for i, (label, points) in enumerate(sorted(by_strategy.items())):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        axb.plot(xs, ys, marker=MARKERS[i % len(MARKERS)], ms=4.0, lw=1.5,
                 color=RAMP[i % len(RAMP)], markeredgecolor="white",
                 markeredgewidth=0.6, label=label)
    if by_strategy:
        all_sizes = sorted({n for pts in by_strategy.values() for n, _ in pts})
        axb.set_xscale("log", base=2)
        axb.set_xticks(all_sizes)
        axb.set_xticklabels([str(v) for v in all_sizes])
    axb.set_xlabel("number of ensemble members $N$", fontsize=9.0, labelpad=5)
    axb.set_ylabel("majority-vote accuracy (%)", fontsize=9.0)
    axb.legend(frameon=False, fontsize=7.8, loc="lower right", handlelength=1.7, borderpad=0.0)
    tidy(axb)

    fig.text(0.2747, 0.088, "(a) effective sample size by construction strategy",
             ha="center", va="bottom", fontsize=8.6, color=MUTED)
    fig.text(0.7853, 0.088, "(b) measured accuracy against member count",
             ha="center", va="bottom", fontsize=8.6, color=MUTED)
    fig.subplots_adjust(top=0.845, bottom=0.265, left=0.072, right=0.988)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "fig2_results.png"
    fig.savefig(destination, dpi=330, facecolor="white")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
