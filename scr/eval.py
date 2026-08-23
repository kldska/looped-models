"""
Строит кривую perplexity(T) на валидации — это и есть "увидеть сатурацию
своими глазами". Модель обучена с ОДНИМ фиксированным T (например 8), а
здесь мы прогоняем инференс с T = 1, 2, 4, 8, 16, 32, 64 и смотрим, где
качество перестаёт улучшаться (или начинает деградировать / NaN-ить).

Дополнительно логируем норму ||h_t|| и косинусное сходство h_t vs h_{t+1}
по итерациям — прямой диагностический сигнал сходимости к fixed point
(если cos_sim -> 1, состояние перестало меняться).

Запуск:
    python src/eval.py --checkpoint checkpoints/baseline_T8/final.pt \
        --tokenizer_dir tokenizer --loop_counts 1,2,4,8,16,32,64
"""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from model import LoopedConfig, LoopedTransformer
from data import get_dataloaders


@torch.no_grad()
def eval_at_T(model, loader, cfg, n_loops, n_batches, device):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        losses.append(loss.item())
    mean_loss = sum(losses) / max(1, len(losses))
    return mean_loss, math.exp(min(mean_loss, 20))


@torch.no_grad()
def get_learned_step_scales(model, max_loops, device):
    """Если модель обучена с use_step_scale=True — читает, какую кривую
    scale(t) она выучила, для каждого t от 1 до max_loops. Не требует
    данных: scale(t) зависит только от t, не от входа."""
    if model.step_scale is None:
        return None
    scales = []
    for t in range(max_loops):
        s = model.step_scale(t, device, torch.float32)
        scales.append(s.item())
    return scales


@torch.no_grad()
def get_loop_norm_trace(model, x, max_loops, device):
    """Если модель обучена с use_loop_norm=True — прогоняет один батч и
    записывает ||x|| после loop_norm на каждой итерации. В отличие от
    step_scale, здесь норма — не выучиваемая функция t, а следствие
    нормализации; ожидается, что она будет примерно константой на любом
    T по построению (это и есть проверка гипотезы)."""
    if model.loop_norm is None:
        return None
    x = x.to(device)
    cos_ = model.rope_cos[: x.shape[1]].to(device)
    sin_ = model.rope_sin[: x.shape[1]].to(device)
    h = model.embed(x)
    norms = []
    for t in range(max_loops):
        proposal = model.block(h, cos_, sin_, attn_mask=None)
        h = model.loop_norm(proposal)
        norms.append(h.norm(dim=-1).mean().item())
    return norms


@torch.no_grad()
def diagnose_fixed_point(model, x, cfg, max_loops, device):
    """Прогоняет один батч через до max_loops итераций и на каждом шаге
    считает норму состояния и cos-similarity с предыдущим шагом —
    диагностика сходимости к неподвижной точке (см. базу по looped-моделям)."""
    x = x.to(device)
    cfg_ = cfg
    cos_ = model.rope_cos[: x.shape[1]].to(device)
    sin_ = model.rope_sin[: x.shape[1]].to(device)

    h = model.embed(x)
    prev = None
    stats = []
    for t in range(max_loops):
        h = model.block(h, cos_, sin_, attn_mask=None)
        norm = h.norm(dim=-1).mean().item()
        if prev is not None:
            cos_sim = F.cosine_similarity(h.flatten(1), prev.flatten(1), dim=-1).mean().item()
        else:
            cos_sim = float("nan")
        stats.append({"loop": t + 1, "state_norm": norm, "cos_sim_prev": cos_sim})
        prev = h
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    ap.add_argument("--loop_counts", type=str, default="1,2,4,8,16,32,64")
    ap.add_argument("--n_batches", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_json", type=str, default="results/eval_curve.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # weights_only=False: чекпоинт хранит LoopedConfig (обычный python-объект),
    # а PyTorch>=2.6 по умолчанию блокирует загрузку не-тензорных объектов
    # из соображений безопасности. Это безопасно, т.к. чекпоинт — наш собственный.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg: LoopedConfig = ckpt["cfg"]
    model = LoopedTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model"])

    _, val_loader = get_dataloaders(args.tokenizer_dir, cfg.max_seq_len, args.batch_size)

    loop_counts = [int(t) for t in args.loop_counts.split(",")]
    curve = []
    print(f"Модель обучена с T={cfg.n_loops_train} (checkpoint step={ckpt.get('step')})")
    print(f"{'T':>4} | {'loss':>8} | {'ppl':>10}")
    for T in loop_counts:
        loss, ppl = eval_at_T(model, val_loader, cfg, T, args.n_batches, device)
        curve.append({"n_loops": T, "loss": loss, "ppl": ppl})
        print(f"{T:>4} | {loss:8.4f} | {ppl:10.2f}")

    # диагностика fixed-point на одном батче с максимальным T
    val_loader2 = get_dataloaders(args.tokenizer_dir, cfg.max_seq_len, args.batch_size)[1]
    x, _ = next(iter(val_loader2))
    diag = diagnose_fixed_point(model, x, cfg, max(loop_counts), device)

    learned_scales = get_learned_step_scales(model, max(loop_counts), device)
    if learned_scales is not None:
        print(f"\nВыученный scale(t), t=1..{max(loop_counts)}:")
        print([round(s, 3) for s in learned_scales])

    loop_norm_trace = get_loop_norm_trace(model, x, max(loop_counts), device)
    if loop_norm_trace is not None:
        print(f"\n||x|| после loop_norm, t=1..{max(loop_counts)} "
              f"(должно быть примерно константой):")
        print([round(n, 2) for n in loop_norm_trace])

    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"trained_T": cfg.n_loops_train, "curve": curve,
                   "fixed_point_diagnostics": diag,
                   "learned_step_scales": learned_scales,
                   "loop_norm_trace": loop_norm_trace}, f, indent=2)
    print(f"\nСохранено: {args.out_json}")


if __name__ == "__main__":
    main()
