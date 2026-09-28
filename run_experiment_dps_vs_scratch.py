"""
Empirical Experiment: Discriminative Parameter Splitting (DPS-LLM) vs Scratch vs Non-Discriminative Splitting.

Compares:
1. Full Model trained from scratch (Full Capacity Baseline)
2. Seed Model + Non-Discriminative Splitting (Plain Duplication)
3. Seed Model + Discriminative Parameter Splitting (DPS-LLM with thesis noise)
"""

import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from llm_parameter_splitting.transformer import SplitTransformerLLM


def generate_synthetic_language_data(vocab_size=128, seq_len=32, num_samples=1000):
    """Generates synthetic sequence data with Markovian dependencies."""
    torch.manual_seed(42)
    # Transition matrix with structure
    transition = torch.softmax(torch.randn(vocab_size, vocab_size), dim=-1)
    
    data = []
    for _ in range(num_samples):
        seq = [torch.randint(0, vocab_size, (1,)).item()]
        for _ in range(seq_len):
            next_token = torch.multinomial(transition[seq[-1]], num_samples=1).item()
            seq.append(next_token)
        data.append(seq)
        
    tensor_data = torch.tensor(data, dtype=torch.long)
    inputs = tensor_data[:, :-1].contiguous()
    targets = tensor_data[:, 1:].contiguous()
    return inputs, targets


def train_steps(model, inputs, targets, optimizer, steps=100, batch_size=32):
    losses = []
    num_samples = inputs.shape[0]
    
    for s in range(steps):
        idx = torch.randint(0, num_samples, (batch_size,))
        b_x, b_y = inputs[idx], targets[idx]
        
        optimizer.zero_grad()
        out = model(b_x, targets=b_y)
        loss = out["ce_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        
    return losses


def run_experiment():
    print("=" * 70)
    print("Running Empirical Evaluation: LLM Discriminative Parameter Splitting")
    print("=" * 70)
    
    vocab_size = 128
    seq_len = 24
    d_model = 64
    n_heads = 4
    n_layers = 2
    seed_d_ffn = 64
    full_d_ffn = 128 # 2x seed
    
    train_x, train_y = generate_synthetic_language_data(vocab_size, seq_len, num_samples=1200)
    val_x, val_y = generate_synthetic_language_data(vocab_size, seq_len, num_samples=300)
    
    seed_steps = 80
    split_steps = 120
    total_steps = seed_steps + split_steps
    
    # -------------------------------------------------------------
    # 1. Full Model from Scratch
    # -------------------------------------------------------------
    print("[1/3] Training Full Model from Scratch (Random Init)...")
    torch.manual_seed(101)
    model_scratch = SplitTransformerLLM(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=full_d_ffn, max_seq_len=seq_len + 4
    )
    opt_scratch = optim.AdamW(model_scratch.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    losses_scratch = train_steps(model_scratch, train_x, train_y, opt_scratch, steps=total_steps)
    t_scratch = time.time() - t0
    
    with torch.no_grad():
        val_scratch = model_scratch(val_x, targets=val_y)["ce_loss"].item()
    print(f"    Scratch Final Val Loss: {val_scratch:.4f} | Time: {t_scratch:.2f}s")

    # -------------------------------------------------------------
    # 2. Seed Model -> Non-Discriminative Splitting (Plain Duplication)
    # -------------------------------------------------------------
    print("[2/3] Training Seed Model -> Non-Discriminative Splitting...")
    torch.manual_seed(101)
    model_nondisc = SplitTransformerLLM(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=seed_d_ffn, max_seq_len=seq_len + 4
    )
    opt_nondisc = optim.AdamW(model_nondisc.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    losses_seed_nd = train_steps(model_nondisc, train_x, train_y, opt_nondisc, steps=seed_steps)
    
    # Perform Non-Discriminative Splitting (noise=False)
    model_nondisc.split_all_ffn_widths(discriminative=False)
    # Re-instantiate optimizer for expanded parameters
    opt_nondisc = optim.AdamW(model_nondisc.parameters(), lr=1e-3, weight_decay=0.01)
    losses_split_nd = train_steps(model_nondisc, train_x, train_y, opt_nondisc, steps=split_steps)
    t_nondisc = time.time() - t0
    losses_nondisc = losses_seed_nd + losses_split_nd
    
    with torch.no_grad():
        val_nondisc = model_nondisc(val_x, targets=val_y)["ce_loss"].item()
    print(f"    Non-Disc Final Val Loss: {val_nondisc:.4f} | Time: {t_nondisc:.2f}s")

    # -------------------------------------------------------------
    # 3. Seed Model -> Discriminative Parameter Splitting (DPS-LLM)
    # -------------------------------------------------------------
    print("[3/3] Training Seed Model -> Discriminative Parameter Splitting (DPS-LLM)...")
    torch.manual_seed(101)
    model_dps = SplitTransformerLLM(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=seed_d_ffn, max_seq_len=seq_len + 4
    )
    opt_dps = optim.AdamW(model_dps.parameters(), lr=1e-3, weight_decay=0.01)
    t0 = time.time()
    losses_seed_dps = train_steps(model_dps, train_x, train_y, opt_dps, steps=seed_steps)
    
    # Perform Discriminative Parameter Splitting (noise=True, thesis Eq 5.7 - 5.8)
    model_dps.split_all_ffn_widths(discriminative=True, c_scale=1e-3)
    opt_dps = optim.AdamW(model_dps.parameters(), lr=1e-3, weight_decay=0.01)
    losses_split_dps = train_steps(model_dps, train_x, train_y, opt_dps, steps=split_steps)
    t_dps = time.time() - t0
    losses_dps = losses_seed_dps + losses_split_dps
    
    with torch.no_grad():
        val_dps = model_dps(val_x, targets=val_y)["ce_loss"].item()
    print(f"    DPS-LLM Final Val Loss: {val_dps:.4f} | Time: {t_dps:.2f}s")

    print("\n" + "=" * 70)
    print("EMPIRICAL COMPARISON SUMMARY:")
    print("=" * 70)
    print(f"Strategy                          | Val Loss | Val Perplexity | Training Time")
    print(f"----------------------------------+----------+----------------+--------------")
    print(f"Full Model (Scratch)              | {val_scratch:.4f}   | {math.exp(val_scratch):.2f}          | {t_scratch:.2f}s")
    print(f"Non-Discriminative Splitting       | {val_nondisc:.4f}   | {math.exp(val_nondisc):.2f}          | {t_nondisc:.2f}s")
    print(f"Discriminative Splitting (DPS-LLM)| {val_dps:.4f}   | {math.exp(val_dps):.2f}          | {t_dps:.2f}s")
    print("=" * 70)
    
    # Calculate compute savings during seed phase
    seed_params = sum(p.numel() for p in model_dps.tok_embeddings.parameters()) + \
                  len(model_dps.layers) * (d_model * seed_d_ffn * 3)
    full_params = sum(p.numel() for p in model_scratch.tok_embeddings.parameters()) + \
                  len(model_scratch.layers) * (d_model * full_d_ffn * 3)
    param_saving = (1.0 - seed_params / full_params) * 100
    print(f"[+] Parameter footprint during Phase 1: {seed_params} vs {full_params} ({param_saving:.1f}% reduction).")
    
    # Generate visualization plot
    plt.figure(figsize=(10, 6))
    plt.plot(losses_scratch, label="Full Model (Trained from Scratch)", color="#e74c3c", alpha=0.8, linewidth=2)
    plt.plot(losses_nondisc, label="Non-Discriminative Splitting (Duplication)", color="#f39c12", alpha=0.8, linewidth=2)
    plt.plot(losses_dps, label="Discriminative Splitting (DPS-LLM)", color="#2ecc71", alpha=0.9, linewidth=2.5)
    plt.axvline(x=seed_steps, color="#7f8c8d", linestyle="--", label="Parameter Split Event (Width 2x)")
    plt.title("LLM Parameter Splitting: Training Loss Trajectories", fontsize=14, fontweight="bold")
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plot_path = r"C:\Users\Minahil Aman\.gemini\antigravity-ide\scratch\dps_llm_training_curve.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved comparative training plot to: {plot_path}")


if __name__ == "__main__":
    run_experiment()
