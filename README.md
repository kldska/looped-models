# Looped Transformer — борьба с сатурацией по глубине

Претрен looped-трансформера (weight-tied блок, применяемый T раз) на
FineWeb, задача — максимизировать пользу от больших T, не давая качеству
выходить на плато.

## Статус

**Шаг 1 (baseline)** — реализован и готов к запуску: чистый looped-блок
без модификаций, обучение с фиксированным T=8, эвал на T=1..64 для
наблюдения сатурации своими глазами.

Модификации, атакующие сатурацию (input injection, loop-index
conditioning, random-T во время обучения) — следующий шаг, не в этой
версии.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Запуск

### 1. Обучить токенизатор (один раз)

```bash
python scripts/train_tokenizer.py --vocab_size 8000 --n_docs 20000
```

Сохранит `tokenizer/vocab.json` и `tokenizer/merges.txt`.

### 2. Обучить baseline (T=8, фиксированный)

```bash
python src/train.py \
    --tokenizer_dir tokenizer \
    --max_tokens 20000000 \
    --n_loops_train 8 \
    --out_dir checkpoints/baseline_T8
```

`--max_tokens` — бюджет токенов обучения (задание допускает до 100M,
для быстрой первой проверки достаточно 10-20M).

### 3. Построить кривую perplexity(T) на инференсе

```bash
python src/eval.py \
    --checkpoint checkpoints/baseline_T8/final.pt \
    --tokenizer_dir tokenizer \
    --loop_counts 1,2,4,8,16,32,64 \
    --out_json results/baseline_T8_curve.json
```

Результат — JSON с perplexity на каждом T плюс диагностика fixed-point
(норма состояния и cos-similarity между соседними итерациями — признак
сходимости к неподвижной точке, если cos_sim → 1).

## Архитектура (`src/model.py`)

Qwen3-style блок: RMSNorm, RoPE, GQA-attention с QK-norm, SwiGLU MLP.
Embedding весит с unembedding (tie_embeddings=True). При
`d_model=256, n_layers_per_block=2, vocab_size=8000` — ~3.5M параметров
(проверьте `python src/model.py`), укладывается в бюджет <10M с запасом
под увеличение блока при необходимости.

## Структура проекта

```
src/
  model.py   — архитектура (LoopedTransformer)
  data.py    — стриминг + упаковка FineWeb в фикс.-длины последовательности
  train.py   — обучение с фиксированным T
  eval.py    — эвал: кривая ppl(T) + диагностика fixed-point
scripts/
  train_tokenizer.py — обучение BPE-токенизатора на сэмпле FineWeb
configs/     — (заполняется по мере добавления вариантов архитектуры)
checkpoints/ — (в .gitignore, кроме финального — на HuggingFace)
results/     — json/графики экспериментов, коммитятся вместе с кодом
```

## Известные ограничения текущей версии

- Данные читаются стримингом напрямую с HuggingFace на каждом запуске
  (нет локального кеша) — для повторных быстрых экспериментов может
  быть разумно один раз сохранить упакованные тензоры на диск.
- `data.py` делит train/val по первым `val_holdout_docs` документам
  потока — простое, но не идеальное разбиение (зависит от shuffle-buffer).
