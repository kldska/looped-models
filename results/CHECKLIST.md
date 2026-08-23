# Чек-лист файлов для финальной сдачи

По заданию (looped_models) нужно прислать:
- GitHub-репозиторий с кодом (чистый, воспроизводимый) и отчётом
- Лучший чекпоинт, сохранённый в открытый репозиторий на HuggingFace

## Что уже есть и должно лежать в репозитории

```
looped-models/
├── README.md                    # инструкция запуска (есть)
├── requirements.txt              # зависимости (есть)
├── .gitignore                    # venv, checkpoints, tokenizer (есть)
├── src/
│   ├── model.py                  # архитектура + все 4 механизма (есть)
│   ├── data.py                   # загрузка FineWeb (есть)
│   ├── train.py                  # обучение, с флагами вариантов (есть)
│   └── eval.py                   # эвал + диагностика (есть)
├── scripts/
│   ├── train_tokenizer.py        # обучение BPE-токенизатора (есть)
│   └── make_plots.py / make_step2_plots.py   # построение графиков
├── results/
│   ├── baseline_curve.json
│   ├── step_scale_curve.json
│   ├── loop_norm_curve.json
│   ├── trajectory_curve.json     # ⚠ ЕЩЁ НЕ ГОТОВ — обучается
│   ├── ppl_vs_loops.png / state_norm_growth.png / cos_sim_growth.png
│   │   / combined_diagnostics.png            # графики Шага 1
│   └── step2_comparison_full.png / step2_comparison_zoom.png  # Шага 2
└── REPORT.md                     # главный документ — Шаг 1 + Шаг 2
```

## Чего не хватает прямо сейчас

1. **`results/trajectory_curve.json`** — дождаться окончания обучения
   на Kaggle, прогнать `eval.py`, прислать.
2. **Объединить `REPORT.md`** — сейчас есть отдельно отчёт по Шагу 1
   (baseline, диагностика) и черновик по Шагу 2 (сравнение трёх
   механизмов). Их нужно свести в один файл — предложу цельную версию
   ниже, как только придут результаты `trajectory`.
3. **Несколько сидов** — все текущие результаты получены с одним сидом
   обучения (`--seed 0` по умолчанию). Для честного сравнения (особенно
   учитывая, что разница между вариантами не всегда велика) стоит
   прогнать хотя бы 2-3 сида на каждый вариант и показывать
   среднее±разброс, а не одно число.
4. **Лучший чекпоинт на HuggingFace** — судя по текущим цифрам, лучший
   кандидат по T=8..16 — `loop_norm`, но с явной оговоркой о его
   провале на больших T. Нужно решить: заливать именно его, или
   `trajectory`, если он окажется более сбалансированным по всему
   диапазону T. Инструкция по загрузке — см. ниже.
5. **`docs/architecture.md` или раздел в REPORT.md** с итоговой
   архитектурой и обоснованием "почему именно такой вид даёт лучшие
   результаты" — формальное требование задания, сейчас это разбросано
   по объяснениям в чате, нужно собрать в одном месте отчёта.

## Как залить чекпоинт на HuggingFace (когда определитесь с финальным вариантом)

```bash
pip install huggingface_hub
huggingface-cli login   # понадобится токен с правами write с huggingface.co/settings/tokens

python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('ваш_юзер/looped-transformer-antifreeze', repo_type='model', exist_ok=True)
api.upload_file(
    path_or_fileobj='checkpoints/loop_norm/final.pt',  # или trajectory, когда решите
    path_in_repo='final.pt',
    repo_id='ваш_юзер/looped-transformer-antifreeze',
    repo_type='model',
)
"
```

Плюс небольшой `README.md` в самом HF-репозитории с парой строк: что за
модель, конфигурация (`d_model`, `n_loops_train`, какой механизм
включён), и на каком T она лучше всего работает — по нашим данным это
пока узкое окно T=4..16, а не "любое T", это важно указать честно.
