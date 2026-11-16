"""Horizon-dependent real-time calibration of layer-C intervals (RW + drift, 5 seeds).
k_h = k0 * h^gamma.  Scores are rescaled s' = s / h^gamma; k0 = a-quantile of s' over real-time calibration
scores (earlier origins, targets <= o, pooled over populations and horizons).
  RT          gamma = 0 (constant k; the scheme in the draft)
  RT_g0.25    gamma = 0.25 fixed
  RT_g0.5     gamma = 0.5 fixed
  RT_gfit     gamma estimated in real time: OLS of log(per-h a-quantile of s) on log(h), over calibration h
  LOO_h       benchmark (not real-time)
Test origins 2012-2014."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd
D = pd.read_csv('results/rolling/calib_scores.csv.gz').rename(columns={'pop': 'popn'})
POPS = sorted(D.popn.unique()); H = 5; out = []
for a in (0.80, 0.95):
    A = D[D.level == a]
    for o in (2012, 2013, 2014):
        cal = A[(A.origin < o) & (A.target <= o)]
        qh = cal.groupby('h').score.quantile(a)
        gfit = float(np.polyfit(np.log(qh.index), np.log(qh.values), 1)[0]) if len(qh) > 1 else 0.0
        for p in POPS:
            T = A[(A.origin == o) & (A.popn == p)]
            ks = {}
            for name, g in (('RT', 0.0), ('RT_g0.25', 0.25), ('RT_g0.5', 0.5), ('RT_gfit', gfit)):
                k0 = np.quantile(cal.score / cal.h ** g, a)
                ks[name] = (pd.Series(k0 * np.arange(1, H + 1) ** g, index=range(1, H + 1)), g)
            ks['LOO_h'] = (A[(A.origin != o) & (A.popn == p)].groupby('h').score.quantile(a), np.nan)
            for name, (k, g) in ks.items():
                kk = T.h.map(k).values; lo, hi = T.centre - kk * T.half, T.centre + kk * T.half; al = 1 - a
                isc = (hi - lo) + 2 / al * ((lo - T.truth) * (T.truth < lo) + (T.truth - hi) * (T.truth > hi))
                gg = pd.DataFrame(dict(h=T.h, ins=(T.truth >= lo) & (T.truth <= hi), w=hi - lo, isc=isc)).groupby('h').mean()
                for hh, r in gg.iterrows():
                    out.append(dict(level=a, origin=o, pop=p, scheme=name, gamma=g, h=hh, k=float(k[hh]),
                                    coverage=r.ins, width=r.w, iscore=r.isc))
O = pd.DataFrame(out); O.to_csv('results/rolling/calib_h_results.csv', index=False)
pd.set_option('display.width', 220)
print(O.groupby(['level', 'scheme']).agg(gamma=('gamma', 'mean'), cov=('coverage', 'mean'), width=('width', 'mean'), iscore=('iscore', 'mean')).round(3))
for a in (0.8, 0.95):
    print(a); print(O[O.level == a].pivot_table(index='h', columns='scheme', values='coverage').round(3))
print(O[O.scheme == 'RT_gfit'].groupby(['level', 'origin']).gamma.first())
