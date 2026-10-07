# 📄 Discriminative Parameter Splitting for LLMs (DPS-LLM): Publication Kit

This directory contains **publication-ready LaTeX manuscripts**, **BibTeX bibliographies**, and **embedded figures** for submitting our empirical findings to premier peer-reviewed machine learning and NLP venues ($0 submission fees).

---

## 🎯 Target Venue Comparison Matrix

| Venue | Focus Track | Submission Deadline | Review Model | Target LaTeX File |
| :--- | :--- | :---: | :---: | :---: |
| **ACL Rolling Review (ARR)** | Efficient NLP, Large Language Models | **15th of Every Month** | Fast-Track Double Blind | [`paper_acl.tex`](paper_acl.tex) |
| **COLM (Conference on Language Modeling)** | Foundation Model Architectures, Scaling Laws | Annual (Spring) | Double Blind (OpenReview) | [`paper_colm.tex`](paper_colm.tex) |
| **ICLR 2026** | Foundation Models, Deep Learning Theory | Annual (October) | Double Blind (OpenReview) | [`paper_iclr.tex`](paper_iclr.tex) |
| **TMLR (Transactions on ML Research)** | Efficient Deep Learning, Continual Learning | **Rolling (Submit Anytime)** | Single/Double Blind (OpenReview) | [`paper_tmlr.tex`](paper_tmlr.tex) |

*All four venues have **$0 submission fees** and **Core A\* / Premier journal status**.*

---

## 📊 Core Empirical Findings Included in the ACL Paper

Source: `results/full_2026-10-06/` (Kaggle 2×T4, notebook as of commit `47fe37b`, `PRESET="FULL"`). All 48 runs share one recipe: pure bfloat16, AdamW (lr 2e-5), 200 steps (batch 1, accumulation 2), 256-token sequences, only the SwiGLU MLP trainable, 512 fine-tuning / 200 validation sequences, seeds 42–43–44. *PPL red.* is the relative perplexity reduction (mean ± std); *Acc.* is GSM8K exact match on 100 test problems.

| Model | Dataset | Base | Naive split | Gaussian split | DPS-LLM | DPS vs naive (paired) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-0.5B | GSM8K PPL red. | 11.18 ± 1.12 | 12.67 ± 0.31 | 11.82 ± 0.41 | 11.94 ± 0.58 | −0.73 pp, 1/3 wins, p=0.29 |
| Qwen2.5-0.5B | GSM8K Acc. | **23.0 ± 2.6** | 13.7 ± 6.7 | 12.7 ± 4.2 | 12.3 ± 9.0 | |
| SmolLM-360M | GSM8K PPL red. | 5.25 ± 0.14 | 10.52 ± 0.29 | 10.51 ± 0.29 | 10.50 ± 0.39 | −0.02 pp, 2/3 wins, p=0.73 |
| SmolLM-360M | GSM8K Acc. | 3.7 ± 1.5 | 2.7 ± 0.6 | 3.3 ± 1.2 | 3.3 ± 1.2 | |
| Qwen2.5-0.5B | Alpaca PPL red. | **27.74 ± 0.28** | 25.57 ± 0.28 | 25.69 ± 0.23 | 25.68 ± 0.34 | +0.11 pp, 3/3 wins, p=0.09 |
| SmolLM-360M | Alpaca PPL red. | 4.57 ± 0.05 | 11.39 ± 0.18 | 11.45 ± 0.09 | 11.36 ± 0.16 | −0.03 pp, 1/3 wins, p=0.56 |

Takeaways:
* **Invariance holds**: init PPL is within 0.012 of the base model; dropping the ½ factor raises it from 5.27 to 14.85.
* **Symmetry breaking does not happen in bfloat16**: the two copies of every neuron keep cosine similarity 1.0000 for the whole run at C=1e-4, so DPS-LLM, Gaussian noise and naive duplication are statistically indistinguishable.
* **Splitting ≠ capacity here**: SmolLM's ~2× larger PPL reduction appears for all three split variants. Under AdamW an exact split trains like the unsplit model with a **2× learning rate on W_down** (each down-projection copy gets the full gradient and the copies add up; verified on a toy SwiGLU block and in the notebook pipeline), and this protocol is learning-rate-limited. The notebook now runs that matched control.
* **Accuracy drops**: splitting lowers Qwen2.5-0.5B GSM8K accuracy from 23% to 12–14% (every seed).

The ICLR, COLM and TMLR manuscripts and the two single-column figures they use (`benchmark_ppl_comparison.png`, `benchmark_symmetry_margin.png`) still describe the earlier single-seed, 25-step run and have not been updated.

### Regenerating the figures
```bash
python generate_benchmark_figure.py results/full_2026-10-06            # multi_model_benchmark_comparison.png
python generate_loss_convergence_figure.py results/full_2026-10-06/loss_histories.json   # benchmark_loss_convergence.png
```

---

## 📁 Files in This Directory

* **[`paper_iclr.tex`](paper_iclr.tex)**: Master manuscript in official ICLR format.
* **[`paper_acl.tex`](paper_acl.tex)**: Two-column ACL Rolling Review manuscript (long paper). Compiled with `\usepackage[review]{acl}` (anonymous, line-numbered) for submission; switch to `[final]` for the camera-ready. Includes Related Work, the invariance proof, the sign-preservation and drift-cancellation propositions, the full fine-tuning recipe (Table 2), the mandatory **Limitations** section, an Ethics Statement, and appendices (MoE/LoRA regimes, prompt templates, reproducibility).
* **[`paper_colm.tex`](paper_colm.tex)**: Manuscript formatted for Conference on Language Modeling.
* **[`paper_tmlr.tex`](paper_tmlr.tex)**: Fast-track journal format for TMLR on OpenReview.
* **[`references.bib`](references.bib)**: Shared BibTeX bibliography (51 entries: Transformer/SwiGLU, function-preserving growth, neuron splitting, MoE upcycling, PEFT, datasets, optimisation, software).
* **[`math_commands.tex`](math_commands.tex)**: Standard mathematical shorthand and notation macros.
* **[`dps_llm_training_curve.png`](dps_llm_training_curve.png)**: Embedded 300 DPI training convergence plot.
* **[`benchmark_ppl_comparison.png`](benchmark_ppl_comparison.png)**: Single-column 300 DPI perplexity improvement comparison plot.
* **[`benchmark_symmetry_margin.png`](benchmark_symmetry_margin.png)**: Single-column 300 DPI symmetry-breaking advantage and capacity relief plot.
* **[`multi_model_benchmark_comparison.png`](multi_model_benchmark_comparison.png)**: Wide two-panel figure: PPL reduction per method (mean ± std, seeds as dots) and per-seed difference of the noisy splits from naive duplication, rendered from `results/full_2026-10-06/benchmark_results.csv`.
* **[`benchmark_loss_convergence.png`](benchmark_loss_convergence.png)**: Training loss on Qwen2.5-0.5B (10-step moving average, mean ± std over seeds), rendered from `results/full_2026-10-06/loss_histories.json`.

---

## 🚀 How to Compile or Edit on Overleaf

1. Compress the sources of this `paper/` folder into a ZIP file (build artefacts excluded):
   ```bash
   cd parameter-splitting-llm/paper
   zip dps_llm_paper.zip *.tex *.bib *.sty *.bst *.png
   ```
   Or build locally: `pdflatex paper_acl && bibtex paper_acl && pdflatex paper_acl && pdflatex paper_acl`.
2. Go to **[overleaf.com](https://www.overleaf.com)** $\to$ **New Project** $\to$ **Upload Project**.
3. Select `dps_llm_paper.zip`.
4. Choose the `.tex` file for your desired venue (`paper_iclr.tex`, `paper_acl.tex`, etc.) and click **Recompile**.
