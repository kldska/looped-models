"""
Looped Transformer — baseline (без модификаций для борьбы с сатурацией).

Один и тот же блок (attention + SwiGLU MLP, в стиле Qwen3: RMSNorm, RoPE,
GQA) применяется T раз с общими весами (weight tying по глубине).

Это ИСХОДНАЯ версия для Шага 1 — увидеть сатурацию своими глазами.
Никакого input injection / loop-index conditioning / random T здесь
намеренно нет: это чистый baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoopedConfig:
    vocab_size: int = 8000
    d_model: int = 256
    n_heads: int = 4
    n_kv_heads: int = 2          # GQA: n_kv_heads < n_heads экономит параметры/память
    n_layers_per_block: int = 2  # сколько transformer-слоёв в ОДНОМ переиспользуемом блоке
    d_ff: int = 672              # SwiGLU intermediate dim (~ 8/3 * d_model, округлено)
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    n_loops_train: int = 8       # T при обучении (baseline: фиксированное)
    dropout: float = 0.0
    tie_embeddings: bool = True  # экономит vocab_size * d_model параметров


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x.to(dtype)) * weight


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    # дублируем, чтобы покрыть весь head_dim (как в стандартной реализации RoPE)
    cos = torch.cat([cos, cos], dim=-1).to(dtype)
    sin = torch.cat([sin, sin], dim=-1).to(dtype)
    return cos, sin  # каждый: (seq_len, head_dim)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q, k: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


class GQAAttention(nn.Module):
    """Grouped-Query Attention с RoPE (в духе Qwen3)."""

    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.n_groups = cfg.n_heads // cfg.n_kv_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * self.head_dim, cfg.d_model, bias=False)

        # QK-norm (как в Qwen3) — нормализуем q/k по последней оси перед RoPE,
        # это помогает стабильности именно на больших эффективных глубинах.
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin, attn_mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        # повторяем kv-головы под каждую группу q-голов (GQA)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerLayer(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = GQAAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.attn_norm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LoopedBlock(nn.Module):
    """Несколько transformer-слоёв, которые вместе переиспользуются как ОДИН
    рекуррентный шаг looped-модели (weight tying по глубине)."""

    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(cfg) for _ in range(cfg.n_layers_per_block)]
        )

    def forward(self, x, cos, sin, attn_mask):
        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        return x


class LoopedTransformer(nn.Module):
    def __init__(self, cfg: LoopedConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.block = LoopedBlock(cfg)  # ОДИН блок, применяется T раз
        self.final_norm = RMSNorm(cfg.d_model)

        if cfg.tie_embeddings:
            self.lm_head = None  # используем embed.weight транспонированно
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        head_dim = cfg.d_model // cfg.n_heads
        cos, sin = build_rope_cache(cfg.max_seq_len, head_dim, cfg.rope_theta,
                                     device="cpu", dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, n_loops: int | None = None,
                return_all_loop_logits: bool = False):
        """
        input_ids: (B, T)
        n_loops: сколько раз применить блок. По умолчанию — cfg.n_loops_train.
                 На инференсе передавайте любое T, чтобы построить кривую
                 качество/T (см. eval.py).
        """
        cfg = self.cfg
        T = input_ids.shape[1]
        n_loops = n_loops or cfg.n_loops_train

        cos = self.rope_cos[:T].to(input_ids.device)
        sin = self.rope_sin[:T].to(input_ids.device)

        x = self.embed(input_ids)

        all_logits = [] if return_all_loop_logits else None
        for t in range(n_loops):
            x = self.block(x, cos, sin, attn_mask=None)
            if return_all_loop_logits:
                all_logits.append(self._project(self.final_norm(x)))

        logits = self._project(self.final_norm(x))
        if return_all_loop_logits:
            return logits, all_logits
        return logits

    def _project(self, x):
        if self.cfg.tie_embeddings:
            return x @ self.embed.weight.T
        return self.lm_head(x)


if __name__ == "__main__":
    cfg = LoopedConfig()
    model = LoopedTransformer(cfg)
    print(f"Параметров: {model.num_params():,}")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits = model(x, n_loops=8)
    print("logits shape:", logits.shape)
