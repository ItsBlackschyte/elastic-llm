"""Model presets.

arch:
  gpt2  - LayerNorm, learned positional embeddings, GELU MLP (mpu-30m era)
  llama - RMSNorm, RoPE, SwiGLU, no biases (mpu-124m-coder onward)

tokenizer is recorded in the config (and therefore in every checkpoint's
config.json) so sampling/eval always uses the right vocabulary.

Param notes (tied embeddings):
  tiny / tiny-llama : CPU test shapes only
  10m               : ~30M total (GPT-2 BPE vocab dominates)
  100m              : GPT-2 small shape, ~124M with GPT-2 vocab
  124m-coder        : llama-style 12x12x768, StarCoder2 vocab 49152,
                      ~123M total
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    arch: str = "gpt2"            # "gpt2" | "llama"
    tokenizer: str = "gpt2"
    rope_theta: float = 10000.0
    mlp_hidden: int = 0           # 0 = auto (4x for gpt2, ~8/3x for llama)


PRESETS = {
    "tiny": ModelConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2, n_embd=64),
    "tiny-llama": ModelConfig(vocab_size=256, block_size=64, n_layer=2, n_head=2,
                              n_embd=64, arch="llama"),
    "10m": ModelConfig(block_size=1024, n_layer=6, n_head=6, n_embd=384),
    "100m": ModelConfig(block_size=1024, n_layer=12, n_head=12, n_embd=768),
    "124m-coder": ModelConfig(vocab_size=49152, block_size=1024, n_layer=12,
                              n_head=12, n_embd=768, arch="llama",
                              tokenizer="bigcode/starcoder2-15b"),
}
