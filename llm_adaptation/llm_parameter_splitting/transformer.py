"""
Transformer Architecture with Native Parameter Splitting Support.

Implements a standard modern Decoder-only LLM (LLaMA/Mistral style):
- RMSNorm
- Multi-Head Attention (RoPE positional embeddings)
- SwiGLU MLP
- Native Parameter Splitting API:
  - Width Splitting (2x d_ffn)
  - Dense-to-MoE Upcycling (E experts with top-k gating)
  - LoRA Parameter Splitting
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .layers import SplitSwiGLU, SplitMoE, SplitLoRALinear
from .noise import AntiSymmetricDiscriminativeNoise


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ffn: int):
        super().__init__()
        self.d_model = d_model
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SplitSwiGLU(d_model, d_ffn)
        self.is_moe = False

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-LN Transformer
        attn_out = self.attn(self.attn_norm(x), mask=mask)
        x = x + attn_out
        
        aux_loss = torch.tensor(0.0, device=x.device)
        normed_x = self.ffn_norm(x)
        
        if self.is_moe:
            ffn_out, aux = self.ffn(normed_x)
            aux_loss = aux
        else:
            ffn_out = self.ffn(normed_x)
            
        x = x + ffn_out
        return x, aux_loss

    def split_ffn_width(self, discriminative: bool = True, c_scale: float = 1e-4) -> float:
        """Splits intermediate width of SwiGLU MLP."""
        if self.is_moe:
            diffs = [exp.split_width(discriminative, c_scale) for exp in self.ffn.experts]
            return max(diffs)
        else:
            return self.ffn.split_width(discriminative, c_scale)

    def upcycle_to_moe(self, num_experts: int = 4, top_k: int = 2,
                       discriminative: bool = True, c_scale: float = 1e-3):
        """Converts dense SwiGLU to SplitMoE using discriminative parameter splitting."""
        if not self.is_moe:
            self.ffn = SplitMoE(self.ffn, num_experts=num_experts, top_k=top_k,
                                discriminative=discriminative, c_scale=c_scale)
            self.is_moe = True


class SplitTransformerLLM(nn.Module):
    """
    Decoder-only language model supporting dynamic Parameter Splitting.
    """
    def __init__(self, vocab_size: int = 4096, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 4, d_ffn: int = 256, max_seq_len: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, d_ffn=d_ffn)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.tok_embeddings.weight

    def get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        h = self.tok_embeddings(input_ids) + self.pos_embeddings(positions)
        
        mask = self.get_causal_mask(seq_len, device)
        total_aux_loss = torch.tensor(0.0, device=device)
        
        for layer in self.layers:
            h, aux = layer(h, mask=mask)
            total_aux_loss = total_aux_loss + aux
            
        h = self.norm(h)
        logits = self.lm_head(h)
        
        result = {"logits": logits, "aux_loss": total_aux_loss}
        
        if targets is not None:
            # Reshape for cross-entropy
            flat_logits = logits.contiguous().view(-1, self.vocab_size)
            flat_targets = targets.contiguous().view(-1)
            loss = F.cross_entropy(flat_logits, flat_targets)
            result["loss"] = loss + 0.01 * total_aux_loss
            result["ce_loss"] = loss
            
        return result

    def split_all_ffn_widths(self, discriminative: bool = True, c_scale: float = 1e-4) -> list[float]:
        """Splits intermediate width of all transformer layers."""
        diffs = []
        for i, layer in enumerate(self.layers):
            diff = layer.split_ffn_width(discriminative=discriminative, c_scale=c_scale)
            diffs.append(diff)
        return diffs

    def upcycle_all_to_moe(self, num_experts: int = 4, top_k: int = 2,
                           discriminative: bool = True, c_scale: float = 1e-3):
        """Converts all dense layers into Mixture-of-Experts via parameter splitting."""
        for layer in self.layers:
            layer.upcycle_to_moe(num_experts=num_experts, top_k=top_k,
                                 discriminative=discriminative, c_scale=c_scale)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
