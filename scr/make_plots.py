"""
Строит графики (perplexity vs T, рост нормы состояния, cos-similarity)
из results/baseline_T8_curve.json — файла, который производит eval.py.

Запуск из корня репозитория:
    python scripts/make_plots.py
"""
import json
import matplotlib.pyplot as plt

INPUT = 'results/baseline_T8_curve.json'
OUTDIR = 'results'

with open(INPUT) as f:
    data = json.load(f)

curve = data['curve']
diag = data['fixed_point_diagnostics']
trained_T = data['trained_T']

loops = [p['n_loops'] for p in curve]
ppl = [p['ppl'] for p in curve]

diag_loops = [d['loop'] for d in diag]
norms = [d['state_norm'] for d in diag]
cos_sims = [d['cos_sim_prev'] for d in diag]

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(loops, ppl, marker='o', color='#2a78d6', linewidth=2)
ax.axvline(trained_T, color='#999', linestyle='--', linewidth=1, label=f'обучено при T={trained_T}')
ax.set_xscale('log', base=2)
ax.set_xticks(loops)
ax.set_xticklabels(loops)
ax.set_xlabel('Число лупов на инференсе')
ax.set_ylabel('Perplexity')
ax.set_title('Baseline: perplexity vs число лупов (log2 шкала X)')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/ppl_vs_loops.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(diag_loops, norms, color='#eb6834', linewidth=2)
ax.set_xlabel('Номер лупа')
ax.set_ylabel('||h_t|| (норма скрытого состояния)')
ax.set_title('Рост нормы состояния по итерациям (линейный, без насыщения)')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/state_norm_growth.png', dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.2))
valid_loops = [l for l, c in zip(diag_loops, cos_sims) if c == c]
valid_cos = [c for c in cos_sims if c == c]
ax.plot(valid_loops, valid_cos, color='#1baf7a', linewidth=2)
ax.set_xlabel('Номер лупа')
ax.set_ylabel('cos(h_t, h_{t-1})')
ax.set_ylim(0.9, 1.001)
ax.set_title('Косинусное сходство соседних состояний — направление фиксируется')
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUTDIR}/cos_sim_growth.png', dpi=150)
plt.close(fig)

fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
ax1.plot(diag_loops, norms, color='#eb6834', linewidth=2)
ax1.set_xlabel('Номер лупа')
ax1.set_ylabel('||h_t||', color='#eb6834')
ax1.tick_params(axis='y', labelcolor='#eb6834')
ax2 = ax1.twinx()
ax2.plot(valid_loops, valid_cos, color='#1baf7a', linewidth=2)
ax2.set_ylabel('cos(h_t, h_{t-1})', color='#1baf7a')
ax2.tick_params(axis='y', labelcolor='#1baf7a')
ax2.set_ylim(0.9, 1.001)
ax1.set_title('Directional lock-in: норма растёт, направление фиксируется')
fig.tight_layout()
fig.savefig(f'{OUTDIR}/combined_diagnostics.png', dpi=150)
plt.close(fig)

print("Готово. min ppl:", min(ppl), "at T=", loops[ppl.index(min(ppl))])
