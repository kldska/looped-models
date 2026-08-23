"""
Обучает маленький byte-level BPE токенизатор на сэмпле FineWeb.

Зачем свой токенизатор, а не готовый GPT-2 (vocab=50257): при бюджете
<10M параметров embedding-матрица (vocab_size x d_model) съедает
непропорционально много бюджета. Токенизатор с vocab~8000 держит
embedding маленьким и оставляет параметры собственно looped-блоку.

Запуск:
    python scripts/train_tokenizer.py --vocab_size 8000 --n_docs 20000
"""
import argparse

from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--n_docs", type=int, default=20000,
                     help="сколько документов FineWeb использовать для обучения токенизатора")
    ap.add_argument("--out_dir", type=str, default="tokenizer")
    args = ap.parse_args()

    ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                       split="train", streaming=True)

    def text_iterator():
        for i, ex in enumerate(ds):
            if i >= args.n_docs:
                break
            yield ex["text"]

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        text_iterator(),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    tok.save_model(args.out_dir)
    print(f"Токенизатор сохранён в {args.out_dir}/ (vocab_size={args.vocab_size})")


if __name__ == "__main__":
    main()
