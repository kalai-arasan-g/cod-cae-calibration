"""Calibrating CAE prediction intervals with normalized (locally weighted) split-conformal scaling.

Base predictive distribution: layer C from eval_rolling.py (RW+drift latent paths with drift uncertainty,
pooled over 5 seeds, plus per-cell training-residual noise).  For each cell we keep
  centre = median, half = (q_hi - q_lo)/2   for nominal level a in {0.80, 0.95}
Score  s = |truth - centre| / half.  A calibrated interval is centre +/- k * half, where k is the
a-quantile of calibration scores (k = 1 means no change).

Calibration schemes (evaluated on test origins 2012, 2013, 2014; h = 1..5):
  RT     real-time: for test origin o, use scores from earlier origins o' < o whose target year <= o
         (i.e. only errors observable at time o); pooled over h and over all 5 populations.
  RT_pop real-time, per population.
  LOO_h  leave-one-origin-out, horizon-specific k_h, per population (uses future windows: NOT real-time;
         a benchmark of what perfect horizon-aware calibration could achieve).
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, torch, warnings
from cae import CAE
warnings.filterwarnings('ignore'); torch.set_num_threads(2)
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']; ORIGINS = list(range(2010, 2015)); SEEDS = range(5)
H, N = 5, 1000; rng = np.random.default_rng(2024); R = 'models'

def rwd_paths(mu):
    d = np.diff(mu); n = len(d); h = np.arange(1, H + 1)
    dbar, sig = d.mean(), d.std(ddof=1)
    dr = rng.normal(dbar, sig / np.sqrt(n), (N, 1))
    return mu[-1] + h * dr + np.cumsum(rng.normal(0, sig, (N, H)), 1)

import os
SCORES='results/rolling/calib_scores.csv.gz'
rec = []   # one row per (pop, origin, h, cause, age)
for p in ([] if os.path.exists(SCORES) else POPS):
    z = np.load(f'data/processed/data_{p}.npz'); years = list(z['years']); logm = z['logm']
    for o in ORIGINS:
        n_tr = years.index(o) + 1; truth = logm[n_tr:n_tr + H]
        sims = []
        for s in SEEDS:
            m = CAE(); m.load_state_dict(torch.load(f'{R}/{p}_o{o}_s{s}.pt')); m.eval()
            f = np.load(f'{R}/{p}_o{o}_s{s}.npz'); sd = (logm[:n_tr] - f['rec']).std(0)
            with torch.no_grad():
                pa = m.decode(torch.tensor(rwd_paths(f['mu']).reshape(-1, 1), dtype=torch.float32))[:, 0].numpy()
            sims.append(pa.reshape(N, H, 8, 19) + rng.normal(size=(N, H, 8, 19)) * sd)
        sim = np.concatenate(sims)
        c = np.median(sim, 0)
        for a in (0.80, 0.95):
            half = (np.quantile(sim, 1 - (1 - a) / 2, 0) - np.quantile(sim, (1 - a) / 2, 0)) / 2
            score = np.abs(truth - c) / half
            for hh in range(H):
                rec.append(pd.DataFrame(dict(pop=p, origin=o, h=hh + 1, level=a, target=o + hh + 1,
                    centre=c[hh].ravel(), half=half[hh].ravel(), truth=truth[hh].ravel(), score=score[hh].ravel())))
        print(p, o, flush=True)
if rec: pd.concat(rec, ignore_index=True).to_csv(SCORES, index=False)
D = pd.read_csv(SCORES).rename(columns={'pop': 'popn'})

TEST = [2012, 2013, 2014]
out = []
for a in (0.80, 0.95):
    A = D[D.level == a]
    for o in TEST:
        for p in POPS:
            T = A[(A.origin == o) & (A.popn == p)]
            cal_rt = A[(A.origin < o) & (A.target <= o)]
            ks = {'Uncalibrated': pd.Series(1.0, index=range(1, H + 1)),
                  'RT': pd.Series(np.quantile(cal_rt.score, a), index=range(1, H + 1)),
                  'RT_pop': pd.Series(np.quantile(cal_rt[cal_rt.popn == p].score, a), index=range(1, H + 1)),
                  'LOO_h': A[(A.origin != o) & (A.popn == p)].groupby('h').score.quantile(a)}
            for name, k in ks.items():
                kk = T.h.map(k).values; lo, hi = T.centre - kk * T.half, T.centre + kk * T.half
                inside = (T.truth >= lo) & (T.truth <= hi); al = 1 - a
                isc = (hi - lo) + 2 / al * ((lo - T.truth) * (T.truth < lo) + (T.truth - hi) * (T.truth > hi))
                g = pd.DataFrame(dict(h=T.h, inside=inside, width=hi - lo, isc=isc)).groupby('h').mean()
                for hh, r in g.iterrows():
                    out.append(dict(level=a, origin=o, pop=p, scheme=name, h=hh, k=float(k[hh]),
                                    coverage=r.inside, width=r.width, interval_score=r.isc))
O = pd.DataFrame(out); O.to_csv('results/rolling/calib_results.csv', index=False)
pd.set_option('display.width', 250)
S = O.groupby(['level', 'scheme']).agg(k=('k', 'mean'), coverage=('coverage', 'mean'), width=('width', 'mean'),
                                       interval_score=('interval_score', 'mean')).reset_index()
print(S.round(3).to_string(index=False))
print(O[O.level == 0.95].pivot_table(index='h', columns='scheme', values='coverage').round(3))
print(O[O.level == 0.95].pivot_table(index='pop', columns='scheme', values='coverage').round(3))
print(O[O.level == 0.95].pivot_table(index='origin', columns='scheme', values='coverage').round(3))
