"""All manuscript figures as vector PDFs (ms/figs/)."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
B = '.'; F = 'figures'; os.makedirs(F, exist_ok=True)
INK, INK2, GRID, BLUE, ORANGE, GREEN, GREY = '#111111', '#555555', '#e3e3e3', '#2a78d6', '#eb6834', '#1baf7a', '#9a9a95'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'grid.color': GRID, 'font.family': 'DejaVu Sans',
                     'pdf.fonttype': 42})
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']
LAB = {'JPN_M': 'Japan M', 'JPN_F': 'Japan F', 'GBR_M': 'UK M', 'GBR_F': 'UK F', 'DEU_M': 'Germany M'}
PAPER = {'JPN_M': (0.065292, 0.071878), 'JPN_F': (0.069508, 0.085754), 'GBR_M': (0.053807, 0.070150),
         'GBR_F': (0.058592, 0.087923), 'DEU_M': (0.075179, 0.077532)}
W = 6.3

# Fig 1 reproduction
r = pd.read_json(f'{B}/results/reproduction/results.jsonl', lines=True).drop_duplicates(['name', 'seed'], keep='last')
r['mse'] = r.rw_drift.map(lambda d: d['mse'])
fig, ax = plt.subplots(figsize=(W, 2.6))
for i, n in enumerate(POPS):
    s = r[r.name == n].mse.values
    ax.scatter(i + np.linspace(-.13, .13, len(s)), s, s=14, color=BLUE, alpha=.85, label='This study (each of 10 seeds)' if i == 0 else None, zorder=3)
    ax.hlines(PAPER[n][0], i - .3, i + .3, color=INK, lw=1.6, label='Tanaka & Matsuyama CAE (10-seed mean)' if i == 0 else None)
    ax.hlines(PAPER[n][1], i - .3, i + .3, color=ORANGE, lw=1.6, ls=(0, (4, 2)), label='ADAPT (published)' if i == 0 else None)
ax.set_xticks(range(5), [LAB[n] for n in POPS]); ax.set_ylabel('Test MSE, 2015–2019'); ax.set_ylim(0, 0.11)
ax.grid(axis='y', lw=.5); ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(f'{F}/fig1_reproduction.pdf')

# Fig 2 beta for C3 and C4 (mean over 10 seeds, last training year) + observed change
fig, axs = plt.subplots(1, 2, figsize=(W, 2.8), sharey=False)
AG = ['0', '1–4'] + [f'{a}' for a in range(5, 85, 5)] + ['85+']
cols = {'JPN_M': BLUE, 'JPN_F': BLUE, 'GBR_M': ORANGE, 'GBR_F': ORANGE, 'DEU_M': GREEN}
ls = {'JPN_M': '-', 'JPN_F': '--', 'GBR_M': '-', 'GBR_F': '--', 'DEU_M': '-'}
for ax, c, nm in zip(axs, (2, 3), ('C3 nervous system', 'C4 circulatory system')):
    for n in POPS:
        bs = []
        for s in range(10):
            o = np.load(f'{B}/results/reproduction/{n}_s{s}.npz'); mu = o['mu'][:, 0]; sg = 1 if mu[-1] < mu[0] else -1
            bs.append(sg * o['beta'][-1, c])
        ax.plot(range(19), np.mean(bs, 0), color=cols[n], ls=ls[n], lw=1.3, label=LAB[n])
    ax.axhline(0, color=INK2, lw=.6); ax.set_xticks(range(0, 19, 3), [AG[k] for k in range(0, 19, 3)])
    ax.set_title(f'{nm}', loc='left'); ax.set_xlabel('Age'); ax.grid(lw=.4)
axs[0].set_ylabel(r'Age sensitivity $\beta_{x,c}$ (2014)'); h_, l_ = axs[1].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=5, loc='lower center')
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f'{F}/fig2_beta.pdf')

# Fig 3 rolling: MSE by horizon per latent model; coverage by layer
P = pd.read_csv(f'{B}/results/rolling/point_metrics.csv'); I = pd.read_csv(f'{B}/results/rolling/interval_metrics.csv')
NAME = {'RWD': 'RW + drift', 'RW': 'RW, no drift', 'ARIMA110': 'ARIMA(1,1,0) + drift', 'DAMPED': 'Damped trend (ETS)'}
COL = {'RWD': BLUE, 'RW': ORANGE, 'ARIMA110': GREEN, 'DAMPED': '#c98500'}
fig, axs = plt.subplots(1, 2, figsize=(W, 2.6))
t = P[P.kind == 'ensemble'].pivot_table(index='h', columns='model', values='mse')
for k in NAME: axs[0].plot(t.index, t[k], marker='o', ms=3, lw=1.4, color=COL[k], label=NAME[k])
axs[0].set_xlabel('Horizon (years)'); axs[0].set_ylabel('MSE (5-seed ensemble)'); axs[0].set_xticks(range(1, 6))
axs[0].grid(axis='y', lw=.4); axs[0].legend(frameon=False); axs[0].set_title('(a) Point accuracy', loc='left')
cc = I[(I.model == 'RWD') & (I.level == 0.95) & (I.origin >= 2012)].groupby(['layer', 'h']).coverage.mean().unstack(0)
for l, lab, col in (('A_latent', 'A: latent only', GREY), ('B_latent+seeds', 'B: + seed spread', '#c98500'), ('C_latent+seeds+cell', 'C: + cell noise', BLUE)):
    axs[1].plot(cc.index, cc[l], marker='o', ms=3, lw=1.4, color=col, label=lab)
axs[1].axhline(.95, color=INK2, lw=.8, ls=':', label='Nominal 95%'); axs[1].set_ylim(0, 1.02)
axs[1].set_xlabel('Horizon (years)'); axs[1].set_ylabel('Empirical coverage'); axs[1].set_xticks(range(1, 6))
axs[1].grid(axis='y', lw=.4); axs[1].legend(frameon=False, loc='lower right'); axs[1].set_title('(b) Uncalibrated 95% intervals', loc='left')
fig.tight_layout(); fig.savefig(f'{F}/fig3_rolling.pdf')

# Fig 4 calibration
O = pd.read_csv(f'{B}/results/rolling/calib_h_results.csv')
SC = {'RT': ('Real-time, constant $k$', ORANGE, '-'), 'RT_gfit': (r'Real-time, $k_0h^{\hat\gamma}$', BLUE, '-'),
      'LOO_h': ('Horizon-specific, not real-time', GREY, (0, (4, 2)))}
fig, axs = plt.subplots(1, 2, figsize=(W, 2.6))
for ax, a in zip(axs, (0.80, 0.95)):
    g = O[O.level == a].groupby(['scheme', 'h']).coverage.mean().unstack(0)
    unc = pd.read_csv(f'{B}/results/rolling/calib_results.csv'); unc = unc[(unc.level == a) & (unc.scheme == 'Uncalibrated')].groupby('h').coverage.mean()
    ax.plot(unc.index, unc.values, marker='o', ms=3, lw=1.4, color='#bbbbb5', label='Uncalibrated (layer C)')
    for k, (lab, col, st) in SC.items(): ax.plot(g.index, g[k], marker='o', ms=3, lw=1.4, color=col, ls=st, label=lab)
    ax.axhline(a, color=INK2, lw=.8, ls=':'); ax.set_ylim(a - 0.25, min(1, a + 0.07) + 0.01); ax.set_xticks(range(1, 6))
    ax.set_xlabel('Horizon (years)'); ax.grid(axis='y', lw=.4); ax.set_title(f'({"a" if a < .9 else "b"}) Nominal {int(a*100)}%', loc='left')
axs[0].set_ylabel('Empirical coverage'); axs[1].legend(frameon=False, loc='lower left')
fig.tight_layout(); fig.savefig(f'{F}/fig4_calibration.pdf')

# Fig 5 latent dimension
PT = pd.read_csv(f'{B}/results/latent/point.csv')
fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
g = PT[PT.K < 5].pivot_table(index='h', columns='K', values='mse_ens')
for K, col in ((1, BLUE), (2, ORANGE), (3, GREEN)): axs[0].plot(g.index, g[K], marker='o', ms=3, lw=1.4, color=col, label=f'K = {K}')
axs[0].set_xlabel('Horizon (years)'); axs[0].set_ylabel('MSE (3-seed ensemble)'); axs[0].set_xticks(range(1, 6))
axs[0].grid(axis='y', lw=.4); axs[0].legend(frameon=False); axs[0].set_title('(a) Forecast error by horizon', loc='left')
rr = PT[PT.K < 5].groupby('K')[['rec_mse', 'mse_ens']].mean(); x = np.arange(3)
axs[1].bar(x - .19, rr.rec_mse, .36, color='#9ec5f4', label='In-sample (training)')
axs[1].bar(x + .19, rr.mse_ens, .36, color='#184f95', label='Out-of-sample (h = 1–5)')
for i, (a_, b_) in enumerate(zip(rr.rec_mse, rr.mse_ens)):
    axs[1].text(i - .19, a_ + .001, f'{a_:.3f}', ha='center', fontsize=6.5, color=INK2); axs[1].text(i + .19, b_ + .001, f'{b_:.3f}', ha='center', fontsize=6.5, color=INK2)
axs[1].set_xticks(x, ['K = 1', 'K = 2', 'K = 3']); axs[1].set_ylabel('MSE'); axs[1].set_ylim(0, .075)
axs[1].legend(frameon=False, loc='upper left'); axs[1].grid(axis='y', lw=.4); axs[1].set_title('(b) Fit versus forecast', loc='left')
fig.tight_layout(); fig.savefig(f'{F}/fig5_latent.pdf')

# Fig 6 COVID: forecast bias by cause, 2020
BC = pd.read_csv(f'{B}/results/rolling/covid_bycause.csv')
CA = ['Infectious', 'Neoplasms', 'Nervous', 'Circulatory', 'Respiratory', 'Congenital', 'External', 'Others']
M = BC[BC.year == 2020].pivot_table(index='cause', columns='pop', values='bias').loc[CA, POPS]
fig, ax = plt.subplots(figsize=(W * .62, 2.7))
cm = LinearSegmentedColormap.from_list('d', ['#1c5cab', '#f0efec', '#c0461c'])
lim = np.abs(M.values).max()
im = ax.imshow(M.values, cmap=cm, norm=TwoSlopeNorm(0, -lim, lim), aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M.values[i, j]; ax.text(j, i, f'{v:+.2f}', ha='center', va='center', fontsize=6.5, color='white' if abs(v) > .6 * lim else INK)
ax.set_xticks(range(5), [LAB[p] for p in POPS]); ax.set_yticks(range(8), [f'C{k+1} {c}' for k, c in enumerate(CA)])
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0); cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.03); cb.ax.tick_params(labelsize=6.5)
cb.set_label('Observed − forecast, mean log rate', fontsize=7)
fig.tight_layout(); fig.savefig(f'{F}/fig6_covid_bias.pdf')
print('ok')
