#!/usr/bin/env python3
"""
Render the neuron-pair divergence figure of the ACL paper (pair_divergence.png): the mean relative distance
||a-b|| / (0.5(||a||+||b||)) between the two copies of every split neuron during fine-tuning, for one model
under the fp32 protocol and the bf16 stochastic-rounding protocol.

    python generate_divergence_figure.py results/part1_medium_2026-10-07/symmetry_histories.json

Lines are the mean over seeds and both datasets. Exact duplication stays at exactly 0 in fp32, which a log
axis cannot show, so it is noted in the panel instead.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = [ROOT / "paper" / "pair_divergence.png", ROOT / "pair_divergence.png"]
PANELS = [("SmolLM-360M", "(a) fp32"), ("SmolLM-360M-bf16SR", "(b) bf16, stochastic rounding")]
STYLE = {"Naive split": ("Naive split", "#D97706", "--"), "Gaussian split": ("Gaussian split", "#0891B2", "-."),
         "DPS-LLM": ("DPS-LLM", "#2563EB", "-")}
INK = "#1E293B"


def main(path):
    hist = json.loads(Path(path).read_text())
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.5), dpi=300, sharey=True)
    for ax, (model, title) in zip(axes, PANELS):
        for method, (label, colour, ls) in STYLE.items():
            runs = [v for k, v in hist.items() if v and k.split(" | ")[0] == model and k.split(" | ")[2] == method]
            if not runs:
                continue
            steps = sorted(int(s) for s in runs[0])
            rel = np.array([[r[str(s)]["rel"] for s in steps] for r in runs]).mean(0)
            if np.all(rel == 0):
                ax.text(0.03, 0.55, f"{label}: exactly 0", transform=ax.transAxes, fontsize=7.5, color=colour)
                continue
            ax.plot(steps, np.where(rel > 0, rel, np.nan), color=colour, ls=ls, lw=1.8, marker="o", ms=2.5, label=label)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", color=INK)
        ax.set_xlabel("Fine-tuning step", fontsize=8.5)
        ax.grid(True, color="#E2E8F0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Relative distance\nbetween copies", fontsize=8.5)
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    for out in OUTPUTS:
        fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1])
