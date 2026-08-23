"""
Стриминговая загрузка FineWeb + упаковка в фиксированные по длине блоки
токенов (стандартный подход для претрена: документы конкатенируются через
<eos> и режутся на чанки seq_len, без паддинга — экономит компьют).
"""
from __future__ import annotations

import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import IterableDataset


class PackedFineWeb(IterableDataset):
    def __init__(self, tokenizer_dir: str, seq_len: int = 512,
                 split_seed: int = 0, is_val: bool = False,
                 val_holdout_docs: int = 2000):
        self.tok = ByteLevelBPETokenizer(
            f"{tokenizer_dir}/vocab.json", f"{tokenizer_dir}/merges.txt"
        )
        self.eos_id = self.tok.token_to_id("<eos>")
        self.seq_len = seq_len
        self.is_val = is_val
        self.val_holdout_docs = val_holdout_docs
        self.split_seed = split_seed

    def _doc_stream(self):
        ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                           split="train", streaming=True)
        ds = ds.shuffle(seed=self.split_seed, buffer_size=10_000)
        for i, ex in enumerate(ds):
            # первые val_holdout_docs документов идут в val, остальные — в train
            in_val_region = i < self.val_holdout_docs
            if in_val_region != self.is_val:
                continue
            yield ex["text"]

    def __iter__(self):
        buffer: list[int] = []
        for text in self._doc_stream():
            ids = self.tok.encode(text).ids
            buffer.extend(ids)
            buffer.append(self.eos_id)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y


def get_dataloaders(tokenizer_dir: str, seq_len: int, batch_size: int,
                     val_holdout_docs: int = 2000, num_workers: int = 2):
    from torch.utils.data import DataLoader

    train_ds = PackedFineWeb(tokenizer_dir, seq_len, is_val=False,
                              val_holdout_docs=val_holdout_docs)
    val_ds = PackedFineWeb(tokenizer_dir, seq_len, is_val=True,
                            val_holdout_docs=val_holdout_docs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)
    return train_loader, val_loader
