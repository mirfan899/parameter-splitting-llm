"""
Unit Test Suite for LLM Parameter Splitting.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_parameter_splitting.noise import (
    ThesisDiscriminativeNoise,
    AntiSymmetricDiscriminativeNoise,
    MultiExpertDiscriminativeNoise
)
from llm_parameter_splitting.layers import (
    SplitLinear,
    SplitSwiGLU,
    SplitMoE,
    SplitLoRALinear
)
from llm_parameter_splitting.transformer import SplitTransformerLLM


class TestLLMParameterSplitting(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_01_thesis_noise(self):
        """Verify direct thesis formulation matches sign boosting."""
        w = torch.tensor([-2.0, -0.5, 0.0, 1.5, 3.0])
        noise_gen = ThesisDiscriminativeNoise(c_constant=1e-4)
        delta = noise_gen.generate(w)
        
        self.assertEqual(delta.shape, w.shape)
        # Perturbation should be positive for w >= 0 and negative for w < 0
        for i in range(len(w)):
            if w[i] >= 0:
                self.assertGreaterEqual(delta[i].item(), 0.0)
            else:
                self.assertLessEqual(delta[i].item(), 0.0)
        print("[+] Test 1 passed: Thesis discriminative noise matches sign-boosting equations.")

    def test_02_antisymmetric_noise(self):
        """Verify anti-symmetric noise satisfies zero drift (Delta1 + Delta2 = 0)."""
        w = torch.randn(64, 128)
        noise_gen = AntiSymmetricDiscriminativeNoise(scale_ratio=1e-3)
        d1, d2 = noise_gen.generate_pair(w)
        
        diff = torch.abs(d1 + d2).max().item()
        self.assertAlmostEqual(diff, 0.0, places=6)
        print("[+] Test 2 passed: Anti-symmetric zero-drift property verified.")

    def test_03_split_linear(self):
        """Verify row and column extensions on SplitLinear."""
        layer = SplitLinear(in_features=32, out_features=16)
        x = torch.randn(2, 32)
        out1 = layer(x)
        self.assertEqual(out1.shape, (2, 16))
        
        # Test row expansion
        layer.split_rows(discriminative=True)
        self.assertEqual(layer.out_features, 32)
        out2 = layer(x)
        self.assertEqual(out2.shape, (2, 32))
        
        # Test column expansion
        x_expanded = torch.cat([x, x], dim=1)
        layer.split_columns(discriminative=True, scale_factor=0.5)
        self.assertEqual(layer.in_features, 64)
        out3 = layer(x_expanded)
        self.assertEqual(out3.shape, (2, 32))
        print("[+] Test 3 passed: SplitLinear row and column splitting verified.")

    def test_04_split_swiglu_invariance(self):
        """Verify SplitSwiGLU intermediate width splitting preserves output invariance."""
        d_model = 64
        d_ffn = 128
        mlp = SplitSwiGLU(d_model=d_model, d_ffn=d_ffn)
        
        max_diff = mlp.split_width(discriminative=True, c_scale=1e-4)
        self.assertEqual(mlp.d_ffn, 256)
        self.assertLess(max_diff, 1e-2, f"Invariance drift too high: {max_diff}")
        print(f"[+] Test 4 passed: SplitSwiGLU width splitting invariant (max diff = {max_diff:.6f}).")

    def test_05_dense_to_moe_upcycling(self):
        """Verify dense SwiGLU can be upcycled into an MoE layer."""
        d_model = 64
        d_ffn = 128
        dense_mlp = SplitSwiGLU(d_model=d_model, d_ffn=d_ffn)
        
        moe = SplitMoE(dense_mlp, num_experts=4, top_k=2, discriminative=True)
        x = torch.randn(2, 8, d_model)
        out, aux_loss = moe(x)
        
        self.assertEqual(out.shape, (2, 8, d_model))
        self.assertGreater(aux_loss.item(), 0.0)
        self.assertEqual(len(moe.experts), 4)
        print("[+] Test 5 passed: Dense-to-MoE upcycling verified.")

    def test_06_split_lora_rank(self):
        """Verify LoRA rank doubling preserves forward pass output."""
        base = nn.Linear(64, 64)
        lora = SplitLoRALinear(base, r=8, lora_alpha=16.0)
        
        max_diff = lora.split_rank(discriminative=True, c_scale=1e-4)
        self.assertEqual(lora.r, 16)
        self.assertLess(max_diff, 1e-2, f"LoRA rank split drift too high: {max_diff}")
        print(f"[+] Test 6 passed: LoRA rank splitting invariant (max diff = {max_diff:.6f}).")

    def test_07_transformer_llm_end_to_end(self):
        """Verify full Transformer LLM forward, split, and loss continuity."""
        model = SplitTransformerLLM(
            vocab_size=256,
            d_model=64,
            n_heads=2,
            n_layers=2,
            d_ffn=128,
            max_seq_len=32
        )
        
        input_ids = torch.randint(0, 256, (2, 16))
        targets = torch.randint(0, 256, (2, 16))
        
        # Step 1: Forward pass before split
        res1 = model(input_ids, targets=targets)
        loss_before = res1["loss"].item()
        params_before = model.count_parameters()
        
        # Step 2: Split all layer intermediate widths
        diffs = model.split_all_ffn_widths(discriminative=True, c_scale=1e-4)
        params_after = model.count_parameters()
        
        # Step 3: Forward pass after split
        res2 = model(input_ids, targets=targets)
        loss_after = res2["loss"].item()
        
        self.assertGreater(params_after, params_before)
        # Loss should be virtually unchanged immediately after splitting
        self.assertAlmostEqual(loss_before, loss_after, delta=0.05)
        
        # Step 4: Verify backpropagation works seamlessly on expanded model
        loss_after_val = res2["loss"]
        loss_after_val.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")
                
        print(f"[+] Test 7 passed: End-to-end LLM Parameter Splitting verified. "
              f"Params: {params_before} -> {params_after}, Loss: {loss_before:.4f} -> {loss_after:.4f}.")


if __name__ == "__main__":
    unittest.main()
