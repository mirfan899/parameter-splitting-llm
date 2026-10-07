#!/usr/bin/env python3
"""
Render the per-step fine-tuning loss figure (paper Figure "loss convergence")
from the loss histories written by the Kaggle benchmark notebook.

The notebooks (Parameter_Splitting_LLM_Kaggle_part1/part2.ipynb, benchmark cell) write
``loss_histories.json`` with keys of the form
    "<model> | <task> | <method> | seed<N>"
and a list of per-step training losses as values. Seeds are averaged and
drawn as a mean line with a +/- 1 std band; each run is first smoothed with
a moving average (default 10 steps) because every step is a single sequence. Run:

    python generate_loss_convergence_figure.py loss_histories.json [model] [window]

The figure is written to paper/benchmark_loss_convergence.png and to the
repository root at 300 DPI as a two-panel, full-width figure (one panel per
dataset) so that the three growth conditions are readable in print.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUTS = [ROOT / "paper" / "benchmark_loss_convergence.png",
           ROOT / "benchmark_loss_convergence.png"]

STYLE = {  # method substring -> (label, colour, linestyle, linewidth)
    "Base": ("Base (no split)", "#6D28D9", "-", 2.0),
    "Naive": ("Naive split ($C=0$)", "#D97706", "--", 2.0),
    "Gaussian": ("Gaussian split", "#0891B2", "-.", 2.0),
    "DPS": ("DPS-LLM (ours)", "#2563EB", "-", 2.4),
}
TASKS = [("GSM8K", "(a) GSM8K"), ("Alpaca", "(b) Alpaca-Cleaned")]


def main(path: str, model_substr: str = "Qwen2.5-0.5B", window: int = 10) -> None:
    hist = json.loads(Path(path).read_text())
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=300, sharey=False)
    for ax, (task_key, title) in zip(axes, TASKS):
        for m_key, (label, colour, ls, lw) in STYLE.items():
            # one run per seed: keys look like "<model> | <task> | <method> | seed42"
            runs = [np.asarray(v, dtype=float) for k, v in hist.items()
                    if model_substr in k and task_key in k and m_key in k.split("|")[2]]
            if not runs:
                continue
            n = min(map(len, runs))
            kernel = np.ones(window) / window
            arr = np.stack([np.convolve(r[:n], kernel, mode="valid") for r in runs])
            mu, sd = arr.mean(0), arr.std(0)
            xs = np.arange(window - 1, n)                      # step at the end of each window
            ax.plot(xs, mu, label=f"{label}" + (f" (n={len(runs)})" if len(runs) > 1 else ""),
                    color=colour, linestyle=ls, linewidth=lw, alpha=0.95, zorder=3)
            if len(runs) > 1:
                ax.fill_between(xs, mu - sd, mu + sd, color=colour, alpha=0.12, linewidth=0, zorder=2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Fine-tuning step (one sequence per step)", fontsize=9.5)
        ax.set_ylabel(f"Training loss ({window}-step moving avg.)", fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
        ax.tick_params(labelsize=8.5)
    axes[0].legend(fontsize=8.5, framealpha=0.92)
    fig.suptitle(f"Per-step training loss on {model_substr}: base model vs. split variants",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    for out in OUTPUTS:
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], *sys.argv[2:3], *map(int, sys.argv[3:4]))
