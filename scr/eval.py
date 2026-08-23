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
    ckpt = torch.load(args.checkpoint, map_location=device)
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

    import os
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"trained_T": cfg.n_loops_train, "curve": curve,
                   "fixed_point_diagnostics": diag}, f, indent=2)
    print(f"\nСохранено: {args.out_json}")


if __name__ == "__main__":
    main()
