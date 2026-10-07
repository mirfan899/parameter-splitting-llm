#!/usr/bin/env python3
"""
Render the ACL paper's benchmark summary figure (multi_model_benchmark_comparison.png)
from the per-seed results written by the Kaggle benchmark notebook.

    python generate_benchmark_figure.py results/full_2026-10-06 [more result dirs ...]

reads <dir>/benchmark_results.csv from every directory (one per Kaggle model group) and writes a full-width, two-panel figure:
  (a) relative validation-perplexity reduction per growth method,
      mean +/- std over seeds with the individual seeds as dots;
  (b) per-seed paired difference of each noisy split from exact duplication
      (naive split), with the mean and the paired t-test p-value.
Any number of models and datasets is supported; groups follow the CSV order.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent
OUTPUTS = [ROOT / "paper" / "multi_model_benchmark_comparison.png",
           ROOT / "multi_model_benchmark_comparison.png"]

METHODS = [  # CSV label -> (legend label, colour); colours shared with generate_loss_convergence_figure.py
    ("Base (no split)", "Base (no split)", "#6D28D9"),
    ("Base (2x W_down lr)", "Base, 2$\\times$ lr on $W_{down}$", "#64748B"),
    ("Naive split", "Naive split ($C=0$)", "#D97706"),
    ("Gaussian split", "Gaussian split", "#0891B2"),
    ("DPS-LLM", "DPS-LLM (ours)", "#2563EB"),
]
NOISY = [("DPS-LLM", "DPS-LLM $-$ naive", "#2563EB"), ("Gaussian split", "Gaussian $-$ naive", "#0891B2")]
INK, INK_MUTED, GRID = "#1E293B", "#64748B", "#E2E8F0"


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#94A3B8")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK, labelsize=9, length=3, width=0.8)


def main(*results_dirs: str) -> None:
    df = pd.concat([pd.read_csv(Path(d) / "benchmark_results.csv") for d in results_dirs], ignore_index=True)
    groups = list(dict.fromkeys(zip(df.Model, df.Dataset)))
    groups.sort(key=lambda g: list(dict.fromkeys(df.Dataset)).index(g[1]))   # datasets together
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(14.0, 4.6), dpi=300,
                                   gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.30})

    # ---- (a) relative PPL reduction, mean +/- std with seeds as dots ------------
    methods = [m for m in METHODS if m[0] in set(df.Method)]
    xa = np.arange(len(groups))
    bw = 0.8 / len(methods)
    for k, (key, label, colour) in enumerate(methods):
        pos = xa + (k - (len(methods) - 1) / 2) * bw
        per_group = [df[(df.Model == m) & (df.Dataset == d) & (df.Method == key)].PPLDrop.to_numpy() for m, d in groups]
        means = [v.mean() for v in per_group]
        stds = [v.std(ddof=1) if len(v) > 1 else 0.0 for v in per_group]
        axa.bar(pos, means, bw * 0.92, yerr=stds, label=label, color=colour, linewidth=0, zorder=3,
                error_kw=dict(ecolor=INK, elinewidth=0.9, capsize=2.5, capthick=0.9))
        for p, v in zip(pos, per_group):
            axa.scatter(np.full(len(v), p), v, s=7, color="white", edgecolor=INK, linewidth=0.5, zorder=4)
        for p, mu, sd in zip(pos, means, stds):
            axa.annotate(f"{mu:.1f}", xy=(p, mu + sd), xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7, color=INK, zorder=5)
    axa.set_xticks(xa)
    axa.set_xticklabels([f"{m}\n({d})" for m, d in groups], fontsize=9.5, color=INK)
    axa.set_ylabel("Relative PPL reduction (%)", fontsize=10, color=INK)
    axa.set_ylim(0, df.PPLDrop.max() * 1.18)
    axa.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8, zorder=0)
    axa.set_axisbelow(True)
    axa.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=2, handlelength=1.2, handleheight=0.9)
    n_seeds = df.Seed.nunique()
    axa.set_title(f"(a) Validation perplexity reduction (mean $\\pm$ std, {n_seeds} seeds)",
                  fontsize=11, fontweight="bold", color=INK, loc="left", pad=10)
    clean_axes(axa)

    # ---- (b) paired difference from exact duplication --------------------------------
    yb = np.arange(len(groups))[::-1]
    off = 0.16
    for j, (key, label, colour) in enumerate(NOISY):
        for y, (m, d) in zip(yb, groups):
            sub = df[(df.Model == m) & (df.Dataset == d)].pivot(index="Seed", columns="Method", values="PPLDrop")
            diff = (sub[key] - sub["Naive split"]).dropna().to_numpy()
            yy = y + (off if j == 0 else -off)
            axb.scatter(diff, np.full(len(diff), yy), s=16, color=colour, alpha=0.45, linewidth=0, zorder=3)
            axb.scatter([diff.mean()], [yy], s=46, marker="D", color=colour, edgecolor="white", linewidth=0.6,
                        zorder=4, label=label if y == yb[0] else None)
            if key == "DPS-LLM" and len(diff) > 1:
                p = ttest_rel(sub.loc[sub[key].notna(), key], sub.loc[sub[key].notna(), "Naive split"]).pvalue
                axb.text(1.02, y, f"p={p:.2f}", transform=axb.get_yaxis_transform(), ha="left", va="center",
                         fontsize=9, color=INK)
    axb.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    axb.set_yticks(yb)
    axb.set_yticklabels([f"{m} ({d})" for m, d in groups], fontsize=9.5, color=INK)
    lim = max(abs(np.array(axb.get_xlim()))) * 1.05
    axb.set_xlim(-lim, lim)
    axb.set_xlabel("Difference from naive split (percentage points of PPL reduction)", fontsize=9.5, color=INK)
    axb.grid(axis="x", color=GRID, linestyle="-", linewidth=0.8, zorder=0)
    axb.set_axisbelow(True)
    axb.set_title("(b) Noisy splits vs. exact duplication, per seed", fontsize=11, fontweight="bold",
                  color=INK, loc="left", pad=10)
    axb.text(1.02, 1.0, "DPS vs naive", transform=axb.transAxes, ha="left", va="bottom",
             fontsize=8.5, color=INK_MUTED, fontweight="bold")
    axb.legend(loc="upper right", frameon=False, fontsize=8.5, handletextpad=0.3)
    clean_axes(axb)
    axb.tick_params(axis="y", length=0)

    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.17, top=0.88)
    for out in OUTPUTS:
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:])
