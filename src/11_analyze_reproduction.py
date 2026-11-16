"""Summarise 10-seed results vs paper Table 4; plot beta sensitivities (Figs 6-7 style)."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

OUT = 'results/reproduction'
PAPER = {  # Table 4: CAE MSE, ADAPT MSE, CAE MAE, ADAPT MAE
 'JPN_M': (0.065292, 0.071878, 0.175547, 0.189706), 'JPN_F': (0.069508, 0.085754, 0.207027, 0.358504),
 'GBR_M': (0.053807, 0.070150, 0.198507, 0.375135), 'GBR_F': (0.058592, 0.087923, 0.218596, 0.332840),
 'DEU_M': (0.075179, 0.077532, 0.188122, 0.295723)}
LABEL = {'JPN_M': 'Japan male', 'JPN_F': 'Japan female', 'GBR_M': 'UK male', 'GBR_F': 'UK female', 'DEU_M': 'Germany male'}
INK, INK2, SURF, BLUE, ORANGE = '#0b0b0b', '#52514e', '#fcfcfb', '#2a78d6', '#eb6834'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'grid.color': '#e4e3df'})

r = pd.read_json(f'{OUT}/results.jsonl', lines=True).drop_duplicates(['name', 'seed'], keep='last')
r['mse'] = r.rw_drift.map(lambda d: d['mse']); r['mae'] = r.rw_drift.map(lambda d: d['mae'])
r['mse_nd'] = r.rw_nodrift.map(lambda d: d['mse']); r['mae_nd'] = r.rw_nodrift.map(lambda d: d['mae'])
rows = []
for n in PAPER:
    s = r[r.name == n]
    p = PAPER[n]
    rows.append(dict(population=LABEL[n], seeds=len(s),
                     our_MSE=s.mse.mean(), our_MSE_sd=s.mse.std(), paper_CAE_MSE=p[0], paper_ADAPT_MSE=p[1],
                     our_MAE=s.mae.mean(), paper_CAE_MAE=p[2], paper_ADAPT_MAE=p[3],
                     our_MSE_nodrift=s.mse_nd.mean(), our_MAE_nodrift=s.mae_nd.mean(),
                     beats_ADAPT_MSE=s.mse.mean() < p[1], beats_ADAPT_MAE=s.mae.mean() < p[3],
                     seeds_beating_ADAPT_MSE=int((s.mse < p[1]).sum())))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/table4_comparison.csv', index=False)
pd.set_option('display.width', 250); print(T.round(4).to_string(index=False))

# ---- Figure: MSE per seed vs paper (Fig. 2 analogue)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for i, n in enumerate(PAPER):
    s = r[r.name == n].mse.values
    ax.scatter(np.full(len(s), i) + np.linspace(-.12, .12, len(s)), s, s=22, color=BLUE, alpha=.8,
               label='Ours, each of 10 seeds' if i == 0 else None, zorder=3)
    ax.hlines(PAPER[n][0], i - .3, i + .3, color=INK, lw=2, label='Paper CAE (10-seed mean)' if i == 0 else None)
    ax.hlines(PAPER[n][1], i - .3, i + .3, color=ORANGE, lw=2, ls=(0, (4, 2)), label='Paper ADAPT' if i == 0 else None)
ax.set_xticks(range(5), [LABEL[n] for n in PAPER]); ax.set_ylabel('Test MSE (log death rate)')
ax.set_title('Out-of-sample MSE, 2015–2019 test years', loc='left', color=INK, fontsize=10)
ax.grid(axis='y', lw=.6); ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=3)
ax.set_ylim(0, max(r.mse.max(), 0.09) * 1.25)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_mse_vs_paper.png', dpi=180)

# ---- Beta plots (Figs 6-7 analogue) + seed stability
AGES = ['0', '1-4'] + [f'{a}' for a in range(5, 85, 5)] + ['85+']
cmap = LinearSegmentedColormap.from_list('b', ['#86b6ef', '#0d366b'])
stab = []
for c, cname in [(2, 'C3 Nervous system'), (3, 'C4 Circulatory system')]:
    fig, axes = plt.subplots(5, 2, figsize=(7.5, 11))
    for i, n in enumerate(PAPER):
        z = np.load(f'data/processed/data_{n}.npz'); yrs = z['years'][:-5]; logm = z['logm'][:-5]
        betas = []
        for sd in range(10):
            o = np.load(f'{OUT}/{n}_s{sd}.npz'); mu, b = o['mu'][:, 0], o['beta']
            sign = 1 if mu[-1] < mu[0] else -1   # orient so that mu decreases over time, as in the paper
            betas.append(sign * b[:, c])         # (T, age)
        betas = np.array(betas)
        ref = betas.mean(0).ravel()
        corr = [np.corrcoef(b.ravel(), ref)[0, 1] for b in betas]
        stab.append(dict(population=LABEL[n], cause=cname, min_corr_with_mean=min(corr),
                         sign_agree_old_ages=float((np.sign(betas[:, -1, -4:]) == np.sign(ref.reshape(len(yrs), 19)[-1, -4:])).mean())))
        b0 = betas[0]
        for t in range(len(yrs)):
            col = cmap(t / (len(yrs) - 1))
            axes[i, 0].plot(range(19), b0[t], color=col, lw=1)
            axes[i, 1].plot(range(19), logm[t, c], color=col, lw=1)
        axes[i, 0].axhline(0, color=INK2, lw=.6)
        axes[i, 0].set_ylabel(LABEL[n], color=INK)
        for a in axes[i]:
            a.set_xticks(range(0, 19, 3), [AGES[k] for k in range(0, 19, 3)]); a.grid(lw=.4)
    axes[0, 0].set_title('Age sensitivity β (seed 0)', loc='left', color=INK, fontsize=10)
    axes[0, 1].set_title('Observed log death rate', loc='left', color=INK, fontsize=10)
    axes[-1, 0].set_xlabel('Age'); axes[-1, 1].set_xlabel('Age')
    fig.suptitle(f'{cname}: lines darken toward 2014 (last training year)', x=0.02, ha='left', color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_beta_C{c+1}.png', dpi=160)
S = pd.DataFrame(stab); S.to_csv(f'{OUT}/beta_seed_stability.csv', index=False); print(S.round(3).to_string(index=False))
