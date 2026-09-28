"""
LLM Parameter Splitting Package.
Adapts discriminative parameter splitting for Large Language Models.
"""

from .noise import (
    ThesisDiscriminativeNoise,
    AntiSymmetricDiscriminativeNoise,
    MultiExpertDiscriminativeNoise
)
from .layers import (
    SplitLinear,
    SplitSwiGLU,
    SplitMoE,
    SplitLoRALinear
)
from .transformer import (
    RMSNorm,
    MultiHeadAttention,
    TransformerBlock,
    SplitTransformerLLM
)

__all__ = [
    "ThesisDiscriminativeNoise",
    "AntiSymmetricDiscriminativeNoise",
    "MultiExpertDiscriminativeNoise",
    "SplitLinear",
    "SplitSwiGLU",
    "SplitMoE",
    "SplitLoRALinear",
    "RMSNorm",
    "MultiHeadAttention",
    "TransformerBlock",
    "SplitTransformerLLM"
]
