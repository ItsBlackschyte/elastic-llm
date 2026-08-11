"""Decoder-only transformer supporting two recipes:

  arch="gpt2"  : LayerNorm, learned positional embeddings, GELU MLP
  arch="llama" : RMSNorm, rotary position embeddings (RoPE), SwiGLU MLP,
                 no biases (the modern Llama/Qwen/Mistral recipe)

Both share the skeleton: token embedding -> N pre-norm blocks -> final norm
-> tied LM head. Attention uses F.scaled_dot_product_attention (FlashAttention
kernels on Ampere+, efficient math fallback on T4).
"""
import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def make_norm(cfg):
    if cfg.arch == "llama":
        return RMSNorm(cfg.n_embd)
    return nn.LayerNorm(cfg.n_embd, bias=cfg.bias)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, nh, T, hd); cos,sin: (T, hd) -> broadcast over B, nh
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, rope=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, k, rope[0], rope[1])
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class GeluMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.mlp_hidden or 4 * cfg.n_embd
        self.c_fc = nn.Linear(cfg.n_embd, h, bias=cfg.bias)
        self.c_proj = nn.Linear(h, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        h = cfg.mlp_hidden or 64 * round(8 * cfg.n_embd / 3 / 64)
        self.w_gate = nn.Linear(cfg.n_embd, h, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, h, bias=False)
        self.c_proj = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.silu(self.w_gate(x)) * self.w_up(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = make_norm(cfg)
        self.mlp = SwiGLUMLP(cfg) if cfg.arch == "llama" else GeluMLP(cfg)

    def forward(self, x, rope=None):
        x = x + self.attn(self.ln_1(x), rope)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.arch in ("gpt2", "llama"), cfg.arch
        self.cfg = cfg
        parts = dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=make_norm(cfg),
        )
        if cfg.arch == "gpt2":
            parts["wpe"] = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.transformer = nn.ModuleDict(parts)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        if cfg.arch == "llama":
            hd = cfg.n_embd // cfg.n_head
            inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
            t = torch.arange(cfg.block_size).float()
            freqs = torch.outer(t, inv)                      # (T, hd/2)
            emb = torch.cat((freqs, freqs), dim=-1)          # (T, hd)
            self.register_buffer("rope_cos", emb.cos(), persistent=False)
            self.register_buffer("rope_sin", emb.sin(), persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.arch == "gpt2":
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.transformer.wte(idx)
        rope = None
        if self.cfg.arch == "gpt2":
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            x = x + self.transformer.wpe(pos)
        else:
            rope = (self.rope_cos[:T], self.rope_sin[:T])
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, rope)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            return logits, loss
        return self.lm_head(x[:, [-1], :]), None

    def configure_optimizer(self, weight_decay, lr, betas, device_type):
        decay = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused_ok = (
            "fused" in inspect.signature(torch.optim.AdamW).parameters
            and device_type == "cuda"
        )
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused_ok)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx
