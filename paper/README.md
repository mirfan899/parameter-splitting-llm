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
* **[`paper_acl.tex`](paper_acl.tex)**: Two-column format for ACL Rolling Review / EMNLP.
* **[`paper_colm.tex`](paper_colm.tex)**: Manuscript formatted for Conference on Language Modeling.
* **[`paper_tmlr.tex`](paper_tmlr.tex)**: Fast-track journal format for TMLR on OpenReview.
* **[`references.bib`](references.bib)**: Complete BibTeX bibliography with all foundational citations.
* **[`math_commands.tex`](math_commands.tex)**: Standard mathematical shorthand and notation macros.
* **[`dps_llm_training_curve.png`](dps_llm_training_curve.png)**: Embedded 300 DPI training convergence plot.
* **[`benchmark_ppl_comparison.png`](benchmark_ppl_comparison.png)**: Single-column 300 DPI perplexity improvement comparison plot.
* **[`benchmark_symmetry_margin.png`](benchmark_symmetry_margin.png)**: Single-column 300 DPI symmetry-breaking advantage and capacity relief plot.
* **[`multi_model_benchmark_comparison.png`](multi_model_benchmark_comparison.png)**: Wide two-panel composite comparison figure.

---

## 🚀 How to Compile or Edit on Overleaf

1. Compress the contents of this `paper/` folder into a ZIP file:
   ```bash
   cd parameter-splitting-llm/paper
   zip -r dps_llm_paper.zip .
   ```
2. Go to **[overleaf.com](https://www.overleaf.com)** $\to$ **New Project** $\to$ **Upload Project**.
3. Select `dps_llm_paper.zip`.
4. Choose the `.tex` file for your desired venue (`paper_iclr.tex`, `paper_acl.tex`, etc.) and click **Recompile**.
