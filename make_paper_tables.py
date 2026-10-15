#!/usr/bin/env python3
"""
Write the LaTeX tables of the ACL paper from one or more Kaggle result folders.

    python make_paper_tables.py results/part1_medium_2026-10-07 [results/part2a_... results/part2b_...]

Reads benchmark_results.csv, lr_sweep_results.csv and (if present) ablation_results.csv from every
folder and writes paper/tables/{main,accuracy,lr,ablation}.tex, which paper_acl.tex \\input s.
Adding a result folder adds its models to every table; no manual editing of numbers.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper" / "tables"
# display order (by size) and names; bf16-SR models are marked with a dagger
ORDER = ["SmolLM-360M", "SmolLM2-360M", "Qwen2.5-0.5B", "TinyLlama-1.1B", "Llama-3.2-1B", "OLMo-2-1B",
         "Qwen2.5-1.5B", "SmolLM2-1.7B", "SmolLM-360M-bf16SR"]
DISPLAY = {"SmolLM-360M-bf16SR": "SmolLM-360M"}
METHODS = ["Base (no split)", "Base (2x W_down lr)", "Naive split", "Gaussian split", "DPS-LLM"]
CONTROL = "Base (2x W_down lr)"


def name(model, precision):
    shown = DISPLAY.get(model, model)
    return shown + (r"$^\dagger$" if precision == "bf16_sr" else "")


def ordered(models):
    return sorted(models, key=lambda m: ORDER.index(m) if m in ORDER else len(ORDER))


def load(dirs, fname):
    frames = [pd.read_csv(Path(d) / fname) for d in dirs if (Path(d) / fname).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main_table(df):
    rows = []
    for ds in ("GSM8K", "Alpaca"):
        for model in ordered(df[df.Dataset == ds].Model.unique()):
            sub = df[(df.Model == model) & (df.Dataset == ds)]
            prec = sub.Precision.iloc[0]
            stats = {m: sub[sub.Method == m].PPLDrop for m in METHODS}
            best = round(max(stats[m].mean() for m in METHODS), 2)
            cells = []
            for m in METHODS:
                v = stats[m]
                txt = f"{v.mean():.2f}{{\\scriptsize$\\pm${v.std(ddof=1):.2f}}}"
                cells.append(f"\\textbf{{{txt}}}" if round(v.mean(), 2) == best else txt)   # ties all bold
            piv = sub.pivot(index="Seed", columns="Method", values="PPLDrop")
            diff = (piv["DPS-LLM"] - piv[CONTROL]).dropna()
            p = ttest_rel(piv.loc[diff.index, "DPS-LLM"], piv.loc[diff.index, CONTROL]).pvalue
            rows.append(f"{name(model, prec)} & {ds} & " + " & ".join(cells)
                        + f" & {signed(diff.mean())} ({p:.2f}) \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    header = (r"\textbf{Model} & \textbf{Data} & \textbf{Base} & \textbf{Base, 2$\times$ lr $W_{down}$} & "
              r"\textbf{Naive split} & \textbf{Gaussian split} & \textbf{DPS-LLM} & \textbf{DPS $-$ ctrl ($p$)} \\")
    return "\n".join([r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule", header,
                      r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def accuracy_table(df):
    g = df[df.Dataset == "GSM8K"]
    rows = []
    for model in ordered(g.Model.unique()):
        sub = g[g.Model == model]
        init = sub.InitAcc.dropna()
        cells = [f"{init.iloc[0]:.0f}" if len(init) else "--"]
        cells += [f"{sub[sub.Method == m].Acc.mean():.1f}" for m in METHODS]
        rows.append(f"{name(model, sub.Precision.iloc[0])} & " + " & ".join(cells) + r" \\")
    header = (r"\textbf{Model} & \textbf{Before} & \textbf{Base} & \textbf{2$\times$ ctrl} & \textbf{Naive} & "
              r"\textbf{Gauss.} & \textbf{DPS} \\")
    return "\n".join([r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule", header, r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])


def lr_table(df, lr_df):
    seed = int(lr_df.Seed.iloc[0])
    main = df[(df.Dataset == "GSM8K") & (df.Seed == seed) & df.Method.isin(["Base (no split)", "DPS-LLM"])]
    main = main.assign(LR=2e-5)
    both = pd.concat([lr_df, main[lr_df.columns.intersection(main.columns)]], ignore_index=True)
    rows = []
    for model in ordered(both.Model.unique()):
        sub = both[both.Model == model]
        prec = df[df.Model == model].Precision.iloc[0]
        best = {m: sub[sub.Method == m].sort_values("PPLDrop").iloc[-1] for m in ("Base (no split)", "DPS-LLM")}
        b, d = best["Base (no split)"], best["DPS-LLM"]
        rows.append(f"{name(model, prec)} & {sci(b.LR)} & {b.PPLDrop:.2f} & {sci(d.LR)} & {d.PPLDrop:.2f} & "
                    f"{signed(d.PPLDrop - b.PPLDrop)} \\\\")
    header = r"\textbf{Model} & \multicolumn{2}{c}{\textbf{Base}} & \multicolumn{2}{c}{\textbf{DPS-LLM}} & $\Delta$ \\"
    sub_header = r" & lr & red. & lr & red. & \\"
    return "\n".join([r"\begin{tabular}{@{}lccccc@{}}", r"\toprule", header, r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
                      sub_header, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])


def signed(x):
    return f"{x:+.2f}".replace("-", "$-$")


def pow10(x):
    """1e-05 -> 10^{-5}, 5e-05 -> 5{\\times}10^{-5} (inside math mode)."""
    mant, exp = f"{x:.0e}".split("e")
    return f"10^{{{int(exp)}}}" if mant == "1" else f"{mant}{{\\times}}10^{{{int(exp)}}}"


def sci(x):
    return f"${pow10(x)}$"


def tex_sci(x, signed=False):
    """Two significant digits in scientific notation, e.g. -1.5e-02 -> $-1.5{\\times}10^{-2}$."""
    if x == 0:
        return "0"
    mant, exp = f"{x:{'+' if signed else ''}.1e}".split("e")
    return f"${mant}{{\\times}}10^{{{int(exp)}}}$"


def ablation_label(v):
    if " C=" in v:
        kind, c = v.split(" C=")
        return f"{kind}, $C{{=}}{pow10(float(c))}$"
    return {"DPS symmetric (+d,+d)": r"DPS, symmetric $(+\delta,+\delta)$",
            "DPS without 1/2 scaling": r"DPS, no $\tfrac{1}{2}$ factor"}.get(v, v)


def ablation_table(abl):
    base = abl[abl.Variant == "Base (no split)"].iloc[0]
    rows = []
    for _, r in abl.iterrows():
        v = r.Variant
        dist = r.get("FinalPairRelDist", np.nan)
        dist = "--" if pd.isna(dist) else tex_sci(dist)
        drift = r.InitPPL - base.InitPPL
        drift = "0" if abs(drift) < 5e-7 else (signed(drift) if abs(drift) >= 0.1 else tex_sci(drift, signed=True))
        rows.append(f"{ablation_label(v)} & {drift} & {r.PPLDrop:.2f} & {dist} \\\\")
        if v in ("Naive split", "Gaussian C=0.1"):
            rows.append(r"\midrule")
    header = r"\textbf{Variant} & \textbf{Init drift} & \textbf{Red.\ (\%)} & \textbf{Pair dist.} \\"
    return "\n".join([r"\begin{tabular}{@{}lccc@{}}", r"\toprule", header, r"\midrule", *rows,
                      r"\bottomrule", r"\end{tabular}"])


def main(dirs):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load(dirs, "benchmark_results.csv")
    (OUT / "main.tex").write_text(main_table(df) + "\n")
    (OUT / "accuracy.tex").write_text(accuracy_table(df) + "\n")
    lr = load(dirs, "lr_sweep_results.csv")
    if len(lr):
        (OUT / "lr.tex").write_text(lr_table(df, lr) + "\n")
    abl = load(dirs, "ablation_results.csv")
    if len(abl):
        (OUT / "ablation.tex").write_text(ablation_table(abl) + "\n")
    n_models = df.Model.nunique()
    print(f"wrote {sorted(p.name for p in OUT.glob('*.tex'))} for {n_models} model entries, {len(df)} runs")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
