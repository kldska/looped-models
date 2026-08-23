"""
Обучение baseline looped-трансформера на FineWeb с ФИКСИРОВАННЫМ T.

Это Шаг 1 плана: самая тупая версия, без input injection / loop-index
conditioning / random T — чтобы увидеть сатурацию своими глазами через
eval.py (эвал на T = 1,2,4,8,16,32,64 после обучения).

Запуск:
    python src/train.py --tokenizer_dir tokenizer --max_tokens 20_000_000 \
        --n_loops_train 8 --out_dir checkpoints/baseline_T8
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F

from model import LoopedConfig, LoopedTransformer
from data import get_dataloaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    ap.add_argument("--out_dir", type=str, default="checkpoints/baseline_T8")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=20_000_000,
                     help="суммарный бюджет токенов обучения (<=100M по заданию)")
    ap.add_argument("--n_loops_train", type=int, default=8)
    ap.add_argument("--use_step_scale", action="store_true",
                     help="включить обучаемый scale(t), зависящий от номера лупа "
                          "(вариант A — без истории, но НЕ обобщается за пределы "
                          "виденных при обучении T)")
    ap.add_argument("--use_loop_norm", action="store_true",
                     help="включить структурную RMSNorm residual stream после "
                          "каждого лупа (вариант C — не зависит от t вообще, "
                          "корректно определена для любого T)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--val_batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = LoopedConfig(max_seq_len=args.seq_len, n_loops_train=args.n_loops_train,
                       use_step_scale=args.use_step_scale,
                       use_loop_norm=args.use_loop_norm)
    model = LoopedTransformer(cfg).to(device)
    print(f"[model] параметров: {model.num_params():,}  |  device={device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay, betas=(0.9, 0.95))

    total_steps = args.max_tokens // (args.batch_size * args.seq_len)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_loader, val_loader = get_dataloaders(
        args.tokenizer_dir, args.seq_len, args.batch_size
    )

    model.train()
    tokens_seen = 0
    step = 0
    t0 = time.time()

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, n_loops=args.n_loops_train)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        (loss / args.grad_accum_steps).backward()

        if (step + 1) % args.grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad()

        tokens_seen += x.numel()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step={step:6d}  tokens={tokens_seen:>10,}  "
                  f"loss={loss.item():.4f}  ppl={math.exp(min(loss.item(), 20)):.2f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  ({dt:.0f}s)")

        if step % args.val_every == 0:
            val_loss = evaluate(model, val_loader, cfg, args.n_loops_train,
                                 args.val_batches, device)
            print(f"  [val] T={args.n_loops_train}  loss={val_loss:.4f}  "
                  f"ppl={math.exp(min(val_loss, 20)):.2f}")
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
                       os.path.join(args.out_dir, "last.pt"))
            model.train()

        if tokens_seen >= args.max_tokens:
            break

    torch.save({"model": model.state_dict(), "cfg": cfg, "step": step},
               os.path.join(args.out_dir, "final.pt"))
    print(f"Готово. Обучено на {tokens_seen:,} токенов, чекпоинт: {args.out_dir}/final.pt")


@torch.no_grad()
def evaluate(model, loader, cfg, n_loops, n_batches, device):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x, n_loops=n_loops)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    main()
