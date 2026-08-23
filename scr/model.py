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

    # --- Шаг 2: адаптивная нормализация ---
    # Вариант A (проще, без истории/состояния) — масштаб зависит ТОЛЬКО от
    # номера итерации t, через синусоидальное кодирование (как RoPE) —
    # обобщается на T, не виденные при обучении, в отличие от lookup-таблицы.
    use_step_scale: bool = False
    step_scale_n_freqs: int = 8
    step_scale_hidden: int = 32

    # Вариант B (сложнее, гипотеза 4 из REPORT.md) — масштаб/gate зависит от
    # текущего состояния и предложенной блоком поправки (trajectory-aware).
    use_gating: bool = False     # обучаемый gate вместо безусловного h_next = block(h)
    gate_hidden_mult: float = 0.5  # размер скрытого слоя gate-сети относительно d_model

    # Вариант C — структурная нормализация residual stream после каждого
    # лупа (RMSNorm), НЕ зависит от t вообще — работает одинаково на любом
    # T по построению, а не по обучению (в отличие от use_step_scale).
    use_loop_norm: bool = False


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


class LearnedStepScale(nn.Module):
    """Обучаемый множитель на вклад блока, зависящий ТОЛЬКО от номера
    итерации t — без учёта состояния или истории (это тот самый более
    простой вариант, который тестируем в первую очередь).

    h_next = h + scale(t) * (block(h) - h)

    В отличие от жёсткого 1/sqrt(t), scale(t) не задаётся формулой
    вручную — она выучивается моделью. Но чтобы это работало и на T,
    сильно превышающих обученные (T=8 при трейне, T=64 при эвале), номер
    шага кодируется НЕПРЕРЫВНО через синусоиды разных частот (та же идея,
    что и в RoPE / классическом positional encoding), а не через
    lookup-таблицу — таблица работала бы только до максимального t,
    виденного при обучении, а синусоидальное кодирование гладко
    продолжается дальше.
    """

    def __init__(self, n_freqs: int = 8, hidden: int = 32):
        super().__init__()
        self.n_freqs = n_freqs
        self.net = nn.Sequential(
            nn.Linear(2 * n_freqs, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Старт близко к 1 (sigmoid(2.0) ≈ 0.88) — в начале обучения
        # ведёт себя почти как baseline, дальше модель сама учится, на
        # каких шагах убавлять вклад.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)

    def _step_embedding(self, t: int, device, dtype) -> torch.Tensor:
        i = torch.arange(self.n_freqs, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (i / self.n_freqs))
        angles = t * inv_freq
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb.to(dtype)

    def forward(self, t: int, device, dtype) -> torch.Tensor:
        emb = self._step_embedding(t, device, dtype)
        logit = self.net(emb)          # (1,)
        return torch.sigmoid(logit)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class AdaptiveResidualGate(nn.Module):
    """Обучаемый, зависящий от состояния gate — решает, сколько от
    предложенного блоком обновления реально применить на этом шаге.

    h_next = h + g * (block(h) - h),  g = sigmoid(MLP([norm(h), norm(delta)]))

    В отличие от фиксированного расписания (например 1/sqrt(t)), gate не
    привязан к номеру итерации явно — он смотрит на текущее состояние и
    на предложенную блоком поправку, и решает адаптивно, для каждого
    токена отдельно. Параметры gate не зависят от T (не растут с числом
    лупов) — это важно по ограничениям задания на масштабируемость.

    Упрощённая первая версия гипотезы 4 (trajectory-conditioned gating)
    из results/REPORT.md: без явной памяти о предыдущих дельтах, только
    текущий шаг. Полная версия с историей — следующий кандидат, если
    этой окажется недостаточно.
    """

    def __init__(self, d_model: int, hidden_mult: float = 0.5):
        super().__init__()
        hidden = max(8, int(d_model * hidden_mult))
        self.norm_x = RMSNorm(d_model)
        self.norm_delta = RMSNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # Инициализация: gate стартует близко к 1 (sigmoid(2.0) ≈ 0.88),
        # то есть в начале обучения ведёт себя почти как baseline (полное
        # принятие block(h)). Модель дообучается сама уменьшать gate там,
        # где это полезно, а не стартует с произвольного 0.5.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 2.0)

    def forward(self, x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([self.norm_x(x), self.norm_delta(delta)], dim=-1)
        gate_logit = self.net(feats)          # (B, T, 1)
        return torch.sigmoid(gate_logit)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


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

        if cfg.use_step_scale and cfg.use_gating:
            raise ValueError("use_step_scale и use_gating взаимоисключающие — "
                              "сначала тестируем вариант A (без истории), "
                              "потом отдельно вариант B (с состоянием)")
        if sum([cfg.use_step_scale, cfg.use_gating, cfg.use_loop_norm]) > 1:
            raise ValueError("use_step_scale / use_gating / use_loop_norm "
                              "взаимоисключающие — сравниваем варианты по "
                              "отдельности, не смешивая механизмы")

        if cfg.use_step_scale:
            self.step_scale = LearnedStepScale(cfg.step_scale_n_freqs, cfg.step_scale_hidden)
        else:
            self.step_scale = None

        if cfg.use_gating:
            self.gate = AdaptiveResidualGate(cfg.d_model, cfg.gate_hidden_mult)
        else:
            self.gate = None

        if cfg.use_loop_norm:
            self.loop_norm = RMSNorm(cfg.d_model)
        else:
            self.loop_norm = None

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
                return_all_loop_logits: bool = False, return_gates: bool = False):
        """
        input_ids: (B, T)
        n_loops: сколько раз применить блок. По умолчанию — cfg.n_loops_train.
                 На инференсе передавайте любое T, чтобы построить кривую
                 качество/T (см. eval.py).
        return_gates: если cfg.use_gating=True, дополнительно вернуть список
                 средних значений gate по итерациям — для диагностики того,
                 что gate реально выучивает (см. eval.py).
        """
        cfg = self.cfg
        T = input_ids.shape[1]
        n_loops = n_loops or cfg.n_loops_train

        cos = self.rope_cos[:T].to(input_ids.device)
        sin = self.rope_sin[:T].to(input_ids.device)

        x = self.embed(input_ids)

        all_logits = [] if return_all_loop_logits else None
        gate_means = [] if (return_gates and (self.gate is not None
                                               or self.step_scale is not None
                                               or self.loop_norm is not None)) else None

        for t in range(n_loops):
            proposal = self.block(x, cos, sin, attn_mask=None)
            if self.step_scale is not None:
                delta = proposal - x
                s = self.step_scale(t, x.device, x.dtype)   # скаляр, зависит только от t
                x = x + s * delta
                if gate_means is not None:
                    gate_means.append(s.item())
            elif self.gate is not None:
                delta = proposal - x
                g = self.gate(x, delta)
                x = x + g * delta
                if gate_means is not None:
                    gate_means.append(g.mean().item())
            elif self.loop_norm is not None:
                # структурная нормализация: не зависит от t вообще, поэтому
                # определена одинаково корректно для любого T, включая
                # T, никогда не виденные при обучении
                x = self.loop_norm(proposal)
                if gate_means is not None:
                    gate_means.append(x.norm(dim=-1).mean().item())
            else:
                x = proposal
            if return_all_loop_logits:
                all_logits.append(self._project(self.final_norm(x)))

        logits = self._project(self.final_norm(x))

        if return_all_loop_logits and return_gates:
            return logits, all_logits, gate_means
        if return_all_loop_logits:
            return logits, all_logits
        if return_gates:
            return logits, gate_means
        return logits

    def _project(self, x):
        if self.cfg.tie_embeddings:
            return x @ self.embed.weight.T
        return self.lm_head(x)


if __name__ == "__main__":
    cfg = LoopedConfig()
    model = LoopedTransformer(cfg)
    print(f"Параметров (baseline): {model.num_params():,}")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits = model(x, n_loops=8)
    print("logits shape:", logits.shape)

    step_cfg = LoopedConfig(use_step_scale=True)
    step_model = LoopedTransformer(step_cfg)
    print(f"\nПараметров (step_scale, без истории): {step_model.num_params():,}  "
          f"(+{step_model.num_params() - model.num_params():,})")
    logits, scales = step_model(x, n_loops=16, return_gates=True)
    print("logits shape:", logits.shape)
    print("scale(t) по лупам:", [round(s, 3) for s in scales])
    # backward pass — проверяем, что градиенты доходят до step_scale
    loss = logits.float().pow(2).mean()
    loss.backward()
    print("grad на step_scale.net[0].weight:",
          step_model.step_scale.net[0].weight.grad is not None)

    gated_cfg = LoopedConfig(use_gating=True)
    gated_model = LoopedTransformer(gated_cfg)
    print(f"\nПараметров (gating, с состоянием): {gated_model.num_params():,}  "
          f"(+{gated_model.num_params() - model.num_params():,})")
    logits, gate_means = gated_model(x, n_loops=16, return_gates=True)
    print("logits shape:", logits.shape)
    print("gate means по лупам:", [round(g, 3) for g in gate_means])

    loopnorm_cfg = LoopedConfig(use_loop_norm=True)
    loopnorm_model = LoopedTransformer(loopnorm_cfg)
    print(f"\nПараметров (loop_norm, структурная): {loopnorm_model.num_params():,}  "
          f"(+{loopnorm_model.num_params() - model.num_params():,})")
    logits, norms = loopnorm_model(x, n_loops=64, return_gates=True)
    print("logits shape:", logits.shape)
    print("||x|| после loop_norm, T=1..64 (первые 10 и последние 5):",
          [round(n, 2) for n in norms[:10]], "...", [round(n, 2) for n in norms[-5:]])
    loss = logits.float().pow(2).mean()
    loss.backward()
    print("grad на loop_norm.weight:", loopnorm_model.loop_norm.weight.grad is not None)
