"""Reviewer point 1: calibration uncertainty when cell-level scores are dependent.
Unit of resampling = window (population x origin), never a cell.
- design effect / effective sample size of the coverage indicator via one-way ANOVA ICC with windows as clusters
- per-window coverage spread
- cluster bootstrap (B draws): resample calibration windows (to re-estimate k) and test windows (to evaluate)
Schemes: RT (constant k) and RT_gfit (k0 h^gamma). Test origins 2012-2014, h = 1..5."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, json
rng = np.random.default_rng(11)
D = pd.read_csv('results/rolling/calib_scores.csv.gz').rename(columns={'pop': 'popn'})
D['win'] = D.popn + '_' + D.origin.astype(str)
B = 2000; H = np.arange(1, 6); out = {}

def kfun(cal, a, scheme):
    if scheme == 'RT':
        return np.full(5, np.quantile(cal.score, a))
    qh = cal.groupby('h').score.quantile(a)
    g = float(np.polyfit(np.log(qh.index), np.log(qh.values), 1)[0]) if len(qh) > 1 else 0.0
    return np.quantile(cal.score / cal.h ** g, a) * H ** g

def icc(df, col):
    """ANOVA estimator of intra-cluster correlation; clusters = windows."""
    g = df.groupby('win')[col]; k = g.size(); n = len(df); m0 = (n - (k ** 2).sum() / n) / (len(k) - 1)
    msb = (k * (g.mean() - df[col].mean()) ** 2).sum() / (len(k) - 1)
    msw = ((df[col] - g.transform('mean')) ** 2).sum() / (n - len(k))
    return float((msb - msw) / (msb + (m0 - 1) * msw)), float(m0)

for a in (0.80, 0.95):
    A = D[D.level == a].copy()
    res = {}
    for scheme in ('RT', 'RT_gfit'):
        # point estimate + per-window coverage
        rows, ins_all = [], []
        for o in (2012, 2013, 2014):
            cal = A[(A.origin < o) & (A.target <= o)]; T = A[A.origin == o].copy()
            k = kfun(cal, a, scheme); T['ins'] = (np.abs(T.truth - T.centre) <= T.h.map(dict(zip(H, k))) * T.half).astype(float)
            ins_all.append(T)
        T = pd.concat(ins_all)
        perwin = T.groupby('win').ins.mean()
        rho, m0 = icc(T, 'ins'); n = len(T); deff = 1 + (m0 - 1) * rho
        # cluster bootstrap
        cal_w = {o: A[(A.origin < o) & (A.target <= o)].win.unique() for o in (2012, 2013, 2014)}
        test_w = T.win.unique()
        byw = {w: g for w, g in A.groupby('win')}
        covs, ks = [], []
        for b in range(B):
            parts = []
            for o in (2012, 2013, 2014):
                cw = rng.choice(cal_w[o], len(cal_w[o]), replace=True)
                cal = pd.concat([byw[w] for w in cw]); cal = cal[cal.target <= o]
                k = kfun(cal, a, scheme); ks.append(k.mean())
                tw = rng.choice([w for w in test_w if w.endswith(str(o))], 5, replace=True)
                for w in tw:
                    t = byw[w]; parts.append((np.abs(t.truth - t.centre) <= t.h.map(dict(zip(H, k))) * t.half).mean())
            covs.append(np.mean(parts))
        res[scheme] = dict(coverage=float(T.ins.mean()), boot_90=[float(np.quantile(covs, .05)), float(np.quantile(covs, .95))],
                           k_mean_boot_90=[float(np.quantile(ks, .05)), float(np.quantile(ks, .95))],
                           per_window_min=float(perwin.min()), per_window_q25=float(perwin.quantile(.25)),
                           per_window_q75=float(perwin.quantile(.75)), per_window_max=float(perwin.max()),
                           n_cells=int(n), n_test_windows=int(len(perwin)), icc=rho, cells_per_window=m0,
                           design_effect=deff, effective_n=float(n / deff))
    out[str(a)] = res
    # effective calibration sample size for k (score exceedance indicator, origins 2010-2013 used for calibration)
for a in (0.80, 0.95):
    A = D[(D.level == a) & (D.origin <= 2013)].copy()
    A['exc'] = (A.score > np.quantile(A.score, a)).astype(float)
    rho, m0 = icc(A, 'exc'); out[str(a)]['calibration_exceedance'] = dict(icc=rho, cells_per_window=m0,
        design_effect=1 + (m0 - 1) * rho, n_cells=len(A), effective_n=len(A) / (1 + (m0 - 1) * rho), n_windows=int(A.win.nunique()))
print(json.dumps(out, indent=1))
json.dump(out, open('results/rolling/window_boot.json', 'w'), indent=1)
