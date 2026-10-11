#!/usr/bin/env python3
"""
Render the ACL paper's benchmark summary figure (multi_model_benchmark_comparison.png)
from the per-seed results written by the Kaggle benchmark notebook.

    python generate_benchmark_figure.py results/full_2026-10-06 [more result dirs ...]

reads <dir>/benchmark_results.csv from every directory (one per Kaggle model group) and writes a full-width, two-panel figure:
  (a) relative validation-perplexity reduction per growth method,
      mean +/- std over seeds with the individual seeds as dots;
  (b) per-seed paired difference of each split from the matched control (the unsplit model with a
      2x learning rate on W_down; exact duplication for older results without it), with the mean and
      the paired t-test p-value of DPS-LLM.
Any number of models and datasets is supported; models are ordered by size, datasets kept together.
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
SPLITS = [("Naive split", "Naive split", "#D97706"), ("Gaussian split", "Gaussian split", "#0891B2"),
          ("DPS-LLM", "DPS-LLM", "#2563EB")]
ORDER = ["SmolLM-360M", "SmolLM2-360M", "Qwen2.5-0.5B", "TinyLlama-1.1B", "Llama-3.2-1B", "OLMo-2-1B",
         "Qwen2.5-1.5B", "SmolLM2-1.7B", "SmolLM-360M-bf16SR"]
SHOWN = {"SmolLM-360M-bf16SR": "SmolLM-360M$^\\dagger$"}
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
    rank = lambda m: ORDER.index(m) if m in ORDER else len(ORDER)
    groups.sort(key=lambda g: (list(dict.fromkeys(df.Dataset)).index(g[1]), rank(g[0])))   # datasets together
    shown = lambda m: SHOWN.get(m, m)
    ref = "Base (2x W_down lr)" if "Base (2x W_down lr)" in set(df.Method) else "Naive split"
    ref_name = "matched control" if ref != "Naive split" else "naive split"
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(max(14.0, 1.25 * len(groups) + 4), 4.6), dpi=300,
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
        for p, mu, sd in zip(pos, means, stds) if len(groups) <= 6 else ():
            axa.annotate(f"{mu:.1f}", xy=(p, mu + sd), xytext=(0, 2), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7, color=INK, zorder=5)
    axa.set_xticks(xa)
    axa.set_xticklabels([shown(m) for m, _ in groups], fontsize=8.5, color=INK, rotation=25, ha="right",
                        rotation_mode="anchor")
    datasets = list(dict.fromkeys(d for _, d in groups))
    for i, ds in enumerate(datasets):                    # dataset name over its block, separator between blocks
        idx = [k for k, (_, d) in enumerate(groups) if d == ds]
        axa.text(np.mean(idx), 1.0, ds, transform=axa.get_xaxis_transform(), ha="center", va="bottom",
                 fontsize=9.5, fontweight="bold", color=INK_MUTED)
        if i:
            axa.axvline(idx[0] - 0.5, color="#CBD5E1", linewidth=0.9, zorder=1)
    axa.set_ylabel("Relative PPL reduction (%)", fontsize=10, color=INK)
    axa.set_ylim(0, df.PPLDrop.max() * 1.30)
    axa.grid(axis="y", color=GRID, linestyle="-", linewidth=0.8, zorder=0)
    axa.set_axisbelow(True)
    axa.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3, handlelength=1.2, handleheight=0.9)
    n_seeds = df.Seed.nunique()
    axa.set_title(f"(a) Validation perplexity reduction (mean $\\pm$ std, {n_seeds} seeds)",
                  fontsize=11, fontweight="bold", color=INK, loc="left", pad=18)
    clean_axes(axa)

    # ---- (b) paired difference of each split from the reference -------------------------
    yb = np.arange(len(groups))[::-1]
    splits = [t for t in SPLITS if t[0] != ref and t[0] in set(df.Method)]
    offs = np.linspace(0.22, -0.22, len(splits))
    for (key, label, colour), off in zip(splits, offs):
        for y, (m, d) in zip(yb, groups):
            sub = df[(df.Model == m) & (df.Dataset == d)].pivot(index="Seed", columns="Method", values="PPLDrop")
            diff = (sub[key] - sub[ref]).dropna().to_numpy()
            axb.scatter(diff, np.full(len(diff), y + off), s=14, color=colour, alpha=0.45, linewidth=0, zorder=3)
            axb.scatter([diff.mean()], [y + off], s=40, marker="D", color=colour, edgecolor="white", linewidth=0.6,
                        zorder=4, label=label if y == yb[0] else None)
            if key == "DPS-LLM" and len(diff) > 1:
                ok = sub[key].notna() & sub[ref].notna()
                p = ttest_rel(sub.loc[ok, key], sub.loc[ok, ref]).pvalue
                axb.text(1.02, y, f"p={p:.2f}", transform=axb.get_yaxis_transform(), ha="left", va="center",
                         fontsize=8.5, color=INK)
    axb.axvline(0, color=INK_MUTED, linewidth=1.0, zorder=2)
    axb.set_yticks(yb)
    axb.set_yticklabels([f"{shown(m)} ({d})" for m, d in groups], fontsize=8.5 if len(groups) > 6 else 9.5, color=INK)
    lim = max(abs(np.array(axb.get_xlim()))) * 1.05
    axb.set_xlim(-lim, lim)
    axb.set_xlabel(f"Difference from {ref_name} (percentage points of PPL reduction)", fontsize=9.5, color=INK)
    axb.grid(axis="x", color=GRID, linestyle="-", linewidth=0.8, zorder=0)
    axb.set_axisbelow(True)
    axb.set_title(f"(b) Splits vs. {ref_name}, per seed", fontsize=11, fontweight="bold",
                  color=INK, loc="left", pad=10)
    axb.text(1.02, 1.0, "DPS", transform=axb.transAxes, ha="left", va="bottom",
             fontsize=8.5, color=INK_MUTED, fontweight="bold")
    axb.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5, handletextpad=0.3)
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
