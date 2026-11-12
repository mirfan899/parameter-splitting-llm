# Function-Preserving Width Splitting of Pre-trained LLMs

Code, notebooks and per-seed results for the anonymous submission *"Function-Preserving Width Splitting of
Pre-trained LLMs: Exact Expansion, a Hidden Learning-Rate Change, and the Limits of Symmetry-Breaking Noise"*.

The paper studies Discriminative Parameter Splitting for LLMs (DPS-LLM): doubling the intermediate width of every
SwiGLU feed-forward block by duplicating the rows of `W_gate` and `W_up` and the columns of `1/2 W_down`, with an
anti-symmetric, polarity-preserving perturbation to break the symmetry of the copies. It proves the split is exactly
function-preserving, shows that under Adam an exact split trains like its seed with a doubled learning rate on
`W_down`, and benchmarks the split against a matched learning-rate control on eight open-weight models.

## Repository layout

| Path | Contents |
|---|---|
| `split_huggingface_llm.py` | The split operator applied in place to a Hugging Face checkpoint |
| `llm_parameter_splitting/` | `SplitSwiGLU`, `SplitMoE`, `SplitLoRALinear` layers and the noise generators (`noise.py`) |
| `test_llm_parameter_splitting.py` | Unit tests for the noise generators, the split layers (including SwiGLU output invariance), MoE upcycling, LoRA rank doubling and an end-to-end split model |
| `run_experiment_dps_vs_scratch.py` | Synthetic experiment behind the paper's first figure |
| `Parameter_Splitting_LLM_Kaggle_part{1,2a,2b,2c}.ipynb` | The benchmark notebooks (Kaggle, 2x T4); identical except for `RUN_GROUP` |
| `results/part*_medium_*/` | Raw per-seed results, learning-rate sweeps, ablations, loss and neuron-pair distance traces, run logs |
| `results/full_2026-10-06/` | The pure-bfloat16 pilot run discussed in the paper |
| `make_paper_tables.py` | Regenerates every table of the paper from the result folders |
| `generate_benchmark_figure.py`, `generate_divergence_figure.py` | Regenerate the paper's figures |
| `paper/` | LaTeX source (`body.tex`, `appendix.tex`, one wrapper per venue) |
| `llm_adaptation/` | Earlier prototype of the same code |

## Reproducing the results

1. Run the notebooks on Kaggle with two T4 GPUs and `PRESET = "MEDIUM"`. Each writes `results_<group>/`.
   Part 2c (Llama-3.2-1B) needs a Hugging Face token with access to the gated model, provided as a Kaggle secret.
2. Regenerate the tables and the benchmark figure from the result folders:

   ```bash
   python make_paper_tables.py results/part1_medium_2026-10-07 results/part2a_medium_2026-10-11 \
       results/part2b_medium_2026-10-11 results/part2c_medium_2026-10-14
   python generate_benchmark_figure.py results/part1_medium_2026-10-07 results/part2a_medium_2026-10-11 \
       results/part2b_medium_2026-10-11 results/part2c_medium_2026-10-14
   python generate_divergence_figure.py results/part1_medium_2026-10-07/symmetry_histories.json
   ```

3. Build the paper: `cd paper && ./build_all.sh`.

The float32 models take about 0.8 h each and the bfloat16 models 2.5-3 h for 30 runs on a free Kaggle session.
