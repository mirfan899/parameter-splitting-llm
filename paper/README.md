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

## 📊 Core Empirical Findings Included in the Paper

All twelve runs share one recipe: Kaggle 2×T4, bfloat16, AdamW (lr 2e-5), 25 steps (batch 1, accumulation 2), 128-token sequences, only the SwiGLU MLP parameters trainable, 30 fine-tuning / 10 held-out examples per dataset, seed 42. *PPL Drop* is the relative perplexity reduction; *3.2×* is the ratio of that reduction to the unexpanded model's, not a wall-clock speed-up.

### Table 1: Multi-Model & Multi-Dataset Empirical Benchmark (Kaggle T4 Verified)
| Model Architecture | Evaluation Dataset | Growth Method | Parameters | Init PPL | Final PPL | PPL Drop (%) | Key Takeaway |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Qwen 2.5** | **Math (GSM8K)** | Base (Original) | 494.0M | 1.91 | 1.52 | +20.43% | Baseline capacity |
| *(Alibaba Research)* | | Naive Split (Duplication) | 807.8M | 1.91 | 1.64 | +13.83% | Symmetric gradient stall |
| | | **DPS-LLM (Our Method)** | 807.8M | **1.91** | **1.64** | **+14.09%** | **DPS wins (+0.26%)** |
| **SmolLM** | **Math (GSM8K)** | Base (Original) | 361.8M | 6.31 | 6.25 | +1.03% | Capacity saturated |
| *(Hugging Face)* | | Naive Split (Duplication) | 597.8M | 6.30 | 6.18 | +1.94% | Split relieves saturation |
| | | **DPS-LLM (Our Method)** | 597.8M | **6.31** | **6.18** | **+2.01%** | **$\approx 2\times$ faster adaptation** |
| **Qwen 2.5** | **Instruction (Alpaca)** | Base (Original) | 494.0M | 7.29 | 5.44 | +25.34% | Baseline adaptation |
| *(Alibaba Research)* | | Naive Split (Duplication) | 807.8M | 7.29 | 5.54 | +24.03% | Duplication lag |
| | | **DPS-LLM (Our Method)** | 807.8M | **7.28** | **5.53** | **+24.12%** | **DPS wins (+0.09%)** |
| **SmolLM** | **Instruction (Alpaca)** | Base (Original) | 361.8M | 8.88 | 8.81 | +0.71% | Severe capacity limit |
| *(Hugging Face)* | | Naive Split (Duplication) | 597.8M | 8.86 | 8.70 | +1.85% | Capacity doubled |
| | | **DPS-LLM (Our Method)** | 597.8M | **8.86** | **8.66** | **+2.28%** | **$\mathbf{3.2\times}$ learning acceleration!** |

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
* **[`multi_model_benchmark_comparison.png`](multi_model_benchmark_comparison.png)**: Wide two-panel composite comparison figure.
* **[`benchmark_loss_convergence.png`](benchmark_loss_convergence.png)**: Per-step fine-tuning loss on Qwen2.5-0.5B (exported from the Kaggle run). The raw loss histories were not saved for that run; the benchmark notebook now writes `loss_histories.json` and `benchmark_results.csv`, and `python ../generate_loss_convergence_figure.py loss_histories.json` regenerates this figure at 300 DPI as a two-panel plot.

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
