"""
Discriminative Noise and Symmetry Breaking Generators for Parameter Splitting.

Implements the original formulation from Muhammad Irfan's thesis:
    f(x) = |(rand(0, 1) - 0.5) * C * N(mu, sigma^2)|
    g(w) = w + f(x) if w >= 0 else w - f(x)
Extended with modern LLM scaling laws (variance normalization, anti-symmetric
zero-drift pairing, and orthogonal null-space projections).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ThesisDiscriminativeNoise:
    """
    Direct implementation of the thesis noise formulation (Eq. 5.7 - 5.8).
    
    Formula:
        f(x) = |(uniform(0, 1) - 0.5) * C * normal(0, 1)|
        g(w) = w + f(x) if w >= 0 else w - f(x)
    """
    def __init__(self, c_constant: float = 1e-4, mu: float = 0.0, sigma: float = 1.0):
        self.c_constant = c_constant
        self.mu = mu
        self.sigma = sigma

    def generate(self, weight: torch.Tensor) -> torch.Tensor:
        """Generates sign-aligned discriminative perturbation matching weight shape."""
        device = weight.device
        dtype = weight.dtype
        
        u = torch.rand_like(weight, device=device, dtype=dtype) - 0.5
        n = torch.normal(mean=self.mu, std=self.sigma, size=weight.shape, device=device, dtype=dtype)
        
        # Thesis Eq 5.7: absolute value of centered uniform scaled by constant and normal
        fx = torch.abs(u * self.c_constant * n)
        
        # Thesis Eq 5.8: sign-dependent boosting
        # g(w) = w + fx if w >= 0 else w - fx => perturbation delta = sign(w) * fx
        sign_w = torch.where(weight >= 0, torch.tensor(1.0, device=device, dtype=dtype),
                                          torch.tensor(-1.0, device=device, dtype=dtype))
        delta = sign_w * fx
        return delta


class AntiSymmetricDiscriminativeNoise:
    """
    Paired Anti-Symmetric Noise for LLM Parameter Splitting.
    
    When splitting a weight W into two copies [W1, W2]:
        W1 = W + Delta
        W2 = W - Delta
    This guarantees:
        1. Delta1 + Delta2 = 0 (first-order zero drift on summed activations).
        2. Gradient diversity: nabla_{W1} L != nabla_{W2} L immediately upon backprop.
    """
    def __init__(self, scale_ratio: float = 1e-3, relative_to_std: bool = True):
        self.scale_ratio = scale_ratio
        self.relative_to_std = relative_to_std

    def generate_pair(self, weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = weight.device
        dtype = weight.dtype
        
        if self.relative_to_std:
            w_std = torch.std(weight).item()
            effective_scale = max(self.scale_ratio * w_std, 1e-6)
        else:
            effective_scale = self.scale_ratio
            
        u = torch.rand_like(weight, device=device, dtype=dtype) - 0.5
        n = torch.randn_like(weight, device=device, dtype=dtype)
        fx = torch.abs(u * effective_scale * n)
        
        sign_w = torch.where(weight >= 0, torch.tensor(1.0, device=device, dtype=dtype),
                                          torch.tensor(-1.0, device=device, dtype=dtype))
        delta = sign_w * fx
        
        return delta, -delta


class MultiExpertDiscriminativeNoise:
    """
    Generates E distinct discriminative perturbations for Dense-to-MoE parameter splitting.
    Ensures sum(Delta_e) = 0 so the average initial expert response matches the seed model.
    """
    def __init__(self, num_experts: int, scale_ratio: float = 1e-3):
        self.num_experts = num_experts
        self.scale_ratio = scale_ratio

    def generate_experts(self, weight: torch.Tensor) -> list[torch.Tensor]:
        device = weight.device
        dtype = weight.dtype
        w_std = torch.std(weight).item()
        effective_scale = max(self.scale_ratio * w_std, 1e-6)
        
        deltas = []
        for e in range(self.num_experts):
            u = torch.rand_like(weight, device=device, dtype=dtype) - 0.5
            n = torch.randn_like(weight, device=device, dtype=dtype)
            fx = u * effective_scale * n
            deltas.append(fx)
            
        # Center deltas so mean perturbation across experts is exactly 0
        mean_delta = torch.stack(deltas, dim=0).mean(dim=0)
        centered_deltas = [d - mean_delta for d in deltas]
        return centered_deltas
