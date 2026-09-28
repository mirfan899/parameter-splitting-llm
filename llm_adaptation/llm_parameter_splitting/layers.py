"""
Neural Network Layers for LLM Parameter Splitting.

Includes:
1. SplitLinear: Base linear layer supporting row/column extension with discriminative noise.
2. SplitSwiGLU: LLaMA/Mistral style SwiGLU MLP with dynamic width doubling and output invariance.
3. SplitMoE: Upcycling dense Transformer MLPs into Mixture-of-Experts via parameter splitting.
4. SplitLoRALinear: Low-Rank Adapter with dynamic rank expansion (r -> 2r) preserving function.
5. SplitMultiheadAttention: Attention layer supporting head doubling (Nh -> 2Nh).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .noise import AntiSymmetricDiscriminativeNoise, MultiExpertDiscriminativeNoise, ThesisDiscriminativeNoise


class SplitLinear(nn.Module):
    """Linear layer supporting row and column expansion with discriminative noise."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def split_rows(self, discriminative: bool = True, c_scale: float = 1e-3):
        """
        Extends rows: out_features -> 2 * out_features (Thesis Eq 5.6).
        Used for incoming projections where each output unit is split into two units.
        """
        old_w = self.weight.data
        if discriminative:
            noise_gen = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
            d1, d2 = noise_gen.generate_pair(old_w)
            new_w = torch.cat([old_w + d1, old_w + d2], dim=0)
        else:
            new_w = torch.cat([old_w, old_w], dim=0)

        new_bias = None
        if self.bias is not None:
            old_b = self.bias.data
            if discriminative:
                noise_gen_b = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
                db1, db2 = noise_gen_b.generate_pair(old_b)
                new_bias = torch.cat([old_b + db1, old_b + db2], dim=0)
            else:
                new_bias = torch.cat([old_b, old_b], dim=0)

        self.out_features = self.out_features * 2
        self.weight = nn.Parameter(new_w)
        if new_bias is not None:
            self.bias = nn.Parameter(new_bias)

    def split_columns(self, discriminative: bool = True, c_scale: float = 1e-3, scale_factor: float = 0.5):
        """
        Extends columns: in_features -> 2 * in_features (Thesis Eq 5.5).
        Used for outgoing projections. Scales original weights by scale_factor (default 0.5)
        to maintain output activation invariance.
        """
        old_w = self.weight.data * scale_factor
        if discriminative:
            noise_gen = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
            d1, d2 = noise_gen.generate_pair(old_w)
            new_w = torch.cat([old_w + d1, old_w + d2], dim=1)
        else:
            new_w = torch.cat([old_w, old_w], dim=1)

        self.in_features = self.in_features * 2
        self.weight = nn.Parameter(new_w)


class SplitSwiGLU(nn.Module):
    """
    LLaMA/Mistral style SwiGLU MLP with dynamic width splitting.
    
    Formula:
        FFN(x) = W_down * (SiLU(W_gate * x) * (W_up * x))
    
    When splitting width (d_ffn -> 2 * d_ffn):
        - W_gate: rows doubled (d_ffn -> 2*d_ffn)
        - W_up: rows doubled (d_ffn -> 2*d_ffn)
        - W_down: columns doubled (d_ffn -> 2*d_ffn) with 0.5x scaling
    Preserves exact forward output at initialization while allowing gradient divergence.
    """
    def __init__(self, d_model: int, d_ffn: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.w_gate = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_up = nn.Linear(d_model, d_ffn, bias=bias)
        self.w_down = nn.Linear(d_ffn, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        hidden = gate * up
        return self.w_down(hidden)

    def split_width(self, discriminative: bool = True, c_scale: float = 1e-4) -> float:
        """
        Splits intermediate MLP width by 2x.
        Returns the maximum absolute difference on unit test input (should be ~0 for invariance).
        """
        device = self.w_gate.weight.device
        dtype = self.w_gate.weight.dtype
        
        # Test input to verify invariance
        with torch.no_grad():
            dummy = torch.randn(1, 4, self.d_model, device=device, dtype=dtype)
            out_before = self.forward(dummy)

        # 1. Split W_gate (rows)
        wg = self.w_gate.weight.data
        if discriminative:
            noise_gen = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
            d1, d2 = noise_gen.generate_pair(wg)
            new_wg = torch.cat([wg + d1, wg + d2], dim=0)
        else:
            new_wg = torch.cat([wg, wg], dim=0)

        # 2. Split W_up (rows)
        wup = self.w_up.weight.data
        if discriminative:
            noise_gen = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
            d1, d2 = noise_gen.generate_pair(wup)
            new_wup = torch.cat([wup + d1, wup + d2], dim=0)
        else:
            new_wup = torch.cat([wup, wup], dim=0)

        # 3. Split W_down (columns) with 0.5x scaling
        wd = self.w_down.weight.data * 0.5
        if discriminative:
            noise_gen = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
            d1, d2 = noise_gen.generate_pair(wd)
            new_wd = torch.cat([wd + d1, wd + d2], dim=1)
        else:
            new_wd = torch.cat([wd, wd], dim=1)

        # Apply parameters
        new_d_ffn = self.d_ffn * 2
        self.w_gate = nn.Linear(self.d_model, new_d_ffn, bias=False, device=device, dtype=dtype)
        self.w_up = nn.Linear(self.d_model, new_d_ffn, bias=False, device=device, dtype=dtype)
        self.w_down = nn.Linear(new_d_ffn, self.d_model, bias=False, device=device, dtype=dtype)

        self.w_gate.weight = nn.Parameter(new_wg)
        self.w_up.weight = nn.Parameter(new_wup)
        self.w_down.weight = nn.Parameter(new_wd)
        self.d_ffn = new_d_ffn

        with torch.no_grad():
            out_after = self.forward(dummy)
            diff = (out_after - out_before).abs().max().item()

        return diff


class SplitMoE(nn.Module):
    """
    Dense-to-MoE Upcycling via Parameter Splitting.
    
    Transforms a single pretrained dense MLP into an E-expert Mixture-of-Experts.
    Each expert inherits the seed weights with discriminative perturbation.
    Router distributes tokens with Top-K gating.
    """
    def __init__(self, seed_mlp: SplitSwiGLU, num_experts: int = 4, top_k: int = 2,
                 discriminative: bool = True, c_scale: float = 1e-3):
        super().__init__()
        self.d_model = seed_mlp.d_model
        self.d_ffn = seed_mlp.d_ffn
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        
        # Router
        self.router = nn.Linear(self.d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        
        # Noise generator
        noise_gen = MultiExpertDiscriminativeNoise(num_experts, scale_ratio=c_scale)
        
        wg = seed_mlp.w_gate.weight.data
        wup = seed_mlp.w_up.weight.data
        wd = seed_mlp.w_down.weight.data
        
        if discriminative:
            wg_deltas = noise_gen.generate_experts(wg)
            wup_deltas = noise_gen.generate_experts(wup)
            wd_deltas = noise_gen.generate_experts(wd)
        else:
            wg_deltas = [torch.zeros_like(wg) for _ in range(num_experts)]
            wup_deltas = [torch.zeros_like(wup) for _ in range(num_experts)]
            wd_deltas = [torch.zeros_like(wd) for _ in range(num_experts)]
            
        # Instantiate expert list
        self.experts = nn.ModuleList()
        for e in range(num_experts):
            exp = SplitSwiGLU(self.d_model, self.d_ffn, bias=False)
            exp.w_gate.weight = nn.Parameter(wg + wg_deltas[e])
            exp.w_up.weight = nn.Parameter(wup + wup_deltas[e])
            exp.w_down.weight = nn.Parameter(wd + wd_deltas[e])
            self.experts.append(exp)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        # Compute router logits and Top-K weights
        logits = self.router(x_flat) # [N, E]
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True) # renormalize
        
        out_flat = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_indices[:, k]
            weight_k = topk_scores[:, k].unsqueeze(-1)
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_in = x_flat[mask]
                    expert_out = self.experts[e](expert_in)
                    out_flat[mask] += weight_k[mask] * expert_out
                    
        # Load balancing auxiliary loss
        router_prob_mean = scores.mean(dim=0)
        tokens_per_expert = (F.one_hot(topk_indices, self.num_experts).float().sum(dim=1)).mean(dim=0)
        aux_loss = self.num_experts * torch.sum(router_prob_mean * tokens_per_expert)
        
        return out_flat.view(batch_size, seq_len, d_model), aux_loss


class SplitLoRALinear(nn.Module):
    """
    Low-Rank Adapter (LoRA) with Dynamic Rank Splitting (r -> 2r).
    
    Delta W * x = (B * A) * (alpha / r) * x
    
    When splitting rank r -> 2r:
        A_split = [A; Delta_A]
        B_split = [B, 0 + Delta_B]
    Since the second half of B is ~0, the forward pass is initial-invariant:
        B_split * A_split = B * A + Delta_B * Delta_A ~= B * A
    Capacity is doubled to learn new tasks/reasoning paths without catastrophic interference.
    """
    def __init__(self, base_layer: nn.Linear, r: int = 8, lora_alpha: float = 16.0):
        super().__init__()
        self.base_layer = base_layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        
        in_dim = base_layer.in_features
        out_dim = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

    def split_rank(self, discriminative: bool = True, c_scale: float = 1e-4) -> float:
        """Splits rank from r -> 2*r preserving existing output."""
        device = self.lora_A.device
        dtype = self.lora_A.dtype
        dummy = torch.randn(1, 4, self.base_layer.in_features, device=device, dtype=dtype)
        
        with torch.no_grad():
            out_before = self.forward(dummy)
            
        old_A = self.lora_A.data
        old_B = self.lora_B.data
        
        # New A has old A on top, perturbed copy on bottom
        if discriminative:
            noise_gen_A = AntiSymmetricDiscriminativeNoise(scale_ratio=c_scale)
            _, dA2 = noise_gen_A.generate_pair(old_A)
            new_A = torch.cat([old_A, old_A + dA2], dim=0)
        else:
            new_A = torch.cat([old_A, old_A], dim=0)
            
        # New B has old B on left, near-zero perturbation on right
        if discriminative:
            # Perturb zero columns with very small scale to break symmetry
            dB = torch.randn_like(old_B) * c_scale * torch.std(old_B).clamp(min=1e-5)
            new_B = torch.cat([old_B, dB], dim=1)
        else:
            new_B = torch.cat([old_B, torch.zeros_like(old_B)], dim=1)
            
        self.r = self.r * 2
        # Keep scaling constant relative to new r or adapt
        self.scaling = self.lora_alpha / self.r
        # Rescale B so that (lora_alpha / (2r)) * (2 * B_old) matches previous scaling
        new_B[:, :old_B.shape[1]] = new_B[:, :old_B.shape[1]] * 2.0
        
        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        
        with torch.no_grad():
            out_after = self.forward(dummy)
            diff = (out_after - out_before).abs().max().item()
            
        return diff
