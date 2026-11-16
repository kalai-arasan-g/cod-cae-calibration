import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
O = 'results/rolling'
P = pd.read_csv(f'{O}/point_metrics.csv'); I = pd.read_csv(f'{O}/interval_metrics.csv')
INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
COL = {'RWD': '#2a78d6', 'RW': '#eb6834', 'ARIMA110': '#1baf7a', 'DAMPED': '#eda100'}
NAME = {'RWD': 'RW + drift (paper)', 'RW': 'RW, no drift', 'ARIMA110': 'ARIMA(1,1,0) + drift', 'DAMPED': 'Damped trend (ETS)'}
LNAME = {'A_latent': 'Latent μ only', 'B_latent+seeds': '+ seed ensemble', 'C_latent+seeds+cell': '+ cell noise'}
LCOL = {'A_latent': '#eb6834', 'B_latent+seeds': '#eda100', 'C_latent+seeds+cell': '#2a78d6'}
POPL = {'JPN_M': 'Japan M', 'JPN_F': 'Japan F', 'GBR_M': 'UK M', 'GBR_F': 'UK F', 'DEU_M': 'Germany M'}
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'grid.color': '#e4e3df'})
pd.set_option('display.width', 250)

# 1. point accuracy: single seed vs 5-seed ensemble, by model (avg over pops, origins, h)
t1 = P.groupby(['model', 'kind']).mse.mean().unstack().loc[list(NAME)]
print('Mean MSE over pops x origins x h=1..5\n', t1.round(4))
t2 = P[P.kind == 'ensemble'].pivot_table(index='pop', columns='model', values='mse').loc[list(POPL), list(NAME)]
print('\nEnsemble MSE by population\n', t2.round(4))
t3 = P[P.kind == 'ensemble'].pivot_table(index='h', columns='model', values='mse')[list(NAME)]
print('\nEnsemble MSE by horizon\n', t3.round(4))
# how often each model wins (pop x origin), ensemble, averaged over h
w = P[P.kind == 'ensemble'].groupby(['pop', 'origin', 'model']).mse.mean().unstack()
print('\nWins (pop x origin, 25 cases):\n', w.idxmin(1).value_counts())
# seed spread: single-seed MSE sd vs ensemble
sp = P[(P.kind == 'single') & (P.model == 'RWD')].groupby(['pop', 'origin', 'seed']).mse.mean().groupby('pop').agg(['mean', 'std', 'max'])
sp['ensemble'] = P[(P.kind == 'ensemble') & (P.model == 'RWD')].groupby('pop').mse.mean()
print('\nRWD single-seed vs ensemble MSE\n', sp.round(4))
t1.to_csv(f'{O}/t_point_by_model.csv'); t2.to_csv(f'{O}/t_point_by_pop.csv'); t3.to_csv(f'{O}/t_point_by_h.csv'); sp.to_csv(f'{O}/t_seed_spread.csv')

# 2. intervals
c = I.groupby(['model', 'layer', 'level']).agg(coverage=('coverage', 'mean'), width=('width', 'mean'),
                                                score=('interval_score', 'mean')).reset_index()
print('\nInterval coverage (nominal 0.80 / 0.95)\n', c.round(3).to_string(index=False))
c.to_csv(f'{O}/t_intervals.csv', index=False)
cp = I[(I.model == 'RWD') & (I.level == 0.95)].pivot_table(index='pop', columns='layer', values='coverage')
print('\nRWD 95% coverage by population\n', cp.round(3)); cp.to_csv(f'{O}/t_cov_by_pop.csv')

# Figure 1: ensemble MSE by horizon
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for k in NAME:
    axs[0].plot(t3.index, t3[k], marker='o', ms=4, lw=2, color=COL[k], label=NAME[k])
axs[0].set_xlabel('Forecast horizon (years)'); axs[0].set_ylabel('MSE (log death rate)')
axs[0].set_title('Point accuracy by horizon\n(5-seed ensemble, 5 pops × 5 origins)', loc='left', fontsize=9, color=INK)
axs[0].grid(axis='y', lw=.6); axs[0].set_xticks(range(1, 6)); axs[0].legend(frameon=False, fontsize=7.5)
cc = I[(I.model == 'RWD') & (I.level == 0.95)].groupby(['layer', 'h']).coverage.mean().unstack(0)
for l in LNAME:
    axs[1].plot(cc.index, cc[l], marker='o', ms=4, lw=2, color=LCOL[l], label=LNAME[l])
axs[1].axhline(0.95, color=INK2, lw=1, ls=(0, (4, 2))); axs[1].text(5, 0.955, 'nominal 95%', ha='right', va='bottom', fontsize=7.5, color=INK2)
axs[1].set_ylim(0, 1.05); axs[1].set_xticks(range(1, 6)); axs[1].set_xlabel('Forecast horizon (years)')
axs[1].set_ylabel('Share of cells inside interval')
axs[1].set_title('95% interval coverage (RW + drift)', loc='left', fontsize=9, color=INK)
axs[1].grid(axis='y', lw=.6); axs[1].legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
fig.tight_layout(); fig.savefig(f'{O}/fig_rolling_summary.png', dpi=180)
