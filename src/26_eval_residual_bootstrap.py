"""Reviewer point 3: independent Gaussian cell noise vs whole-year residual-matrix bootstrap.
For each window (pop x origin 2010-2014), 5 seeds, RW+drift latent paths with drift uncertainty, then noise:
  GAUSS : independent N(0, s_cx^2) per cell (as in the manuscript, distribution C)
  RBOOT : add the full 8x19 training residual matrix of a randomly drawn training year (keeps cross-cause and
          cross-age dependence and the non-Gaussian shape)
Targets: (i) each cause x age cell (152), (ii) all-cause log rate per age group (19), log(sum_c exp(logm_cx)).
Scores |y - median| / half-width are saved for real-time calibration (constant k and k0 h^gamma), test origins 2012-2014.
Also: residual diagnostics (cross-cause correlation, excess kurtosis) for training and out-of-sample errors."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, torch, json
from scipy import stats
from cae import CAE
torch.set_num_threads(2); rng = np.random.default_rng(5)
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']; R = 'models'; N = 600; H = 5
rows, diag = [], []

def allcause(x):  # x (..., 8, 19) log rates -> (..., 19)
    return np.log(np.exp(x).sum(-2))

for p in POPS:
    z = np.load(f'data/processed/data_{p}.npz'); years = list(z['years']); logm = z['logm']
    for o in range(2010, 2015):
        n_tr = years.index(o) + 1; truth = logm[n_tr:n_tr + H]; truth_ac = allcause(truth)
        sims = {'GAUSS': [], 'RBOOT': []}; ens = []
        for s in range(5):
            m = CAE(); m.load_state_dict(torch.load(f'{R}/{p}_o{o}_s{s}.pt')); m.eval()
            f = np.load(f'{R}/{p}_o{o}_s{s}.npz'); mu = f['mu'].reshape(-1); res = logm[:n_tr] - f['rec']   # (n_tr,8,19)
            sd = res.std(0); d = np.diff(mu); n = len(d); dbar, sig = d.mean(), d.std(ddof=1); hh = np.arange(1, H + 1)
            paths = mu[-1] + hh * rng.normal(dbar, sig / np.sqrt(n), (N, 1)) + np.cumsum(rng.normal(0, sig, (N, H)), 1)
            with torch.no_grad():
                base = m.decode(torch.tensor(paths.reshape(-1, 1), dtype=torch.float32))[:, 0].numpy().reshape(N, H, 8, 19)
                ens.append(m.decode(torch.tensor((mu[-1] + hh * dbar)[:, None], dtype=torch.float32))[:, 0].numpy())
            sims['GAUSS'].append(base + rng.normal(size=base.shape) * sd)
            sims['RBOOT'].append(base + res[rng.integers(0, n_tr, (N, H))])
            if s == 0:  # diagnostics on training residuals: cross-cause correlation within age, excess kurtosis
                cc = [np.corrcoef(res[:, :, x].T) for x in range(19)]
                off = np.concatenate([c[np.triu_indices(8, 1)] for c in cc]); off = off[np.isfinite(off)]
                ca = [np.corrcoef(res[:, c, :-1].ravel(), res[:, c, 1:].ravel())[0, 1] for c in range(8)]
                diag.append(dict(pop=p, origin=o, kind='train_residual', mean_abs_cross_cause_corr=float(np.mean(np.abs(off))),
                                 adjacent_age_corr=float(np.nanmean(ca)), excess_kurtosis=float(stats.kurtosis(res.ravel())),
                                 shapiro_p=float(stats.shapiro((res / res.std(0)).ravel()[:5000]).pvalue)))
        err = truth - np.mean(ens, 0)                                       # out-of-sample errors (H,8,19)
        cc = [np.corrcoef(err[:, :, x].T) for x in range(19)]
        off = np.concatenate([c[np.triu_indices(8, 1)] for c in cc]); off = off[np.isfinite(off)]
        diag.append(dict(pop=p, origin=o, kind='forecast_error', mean_abs_cross_cause_corr=float(np.mean(np.abs(off))),
                         mean_cross_cause_corr=float(np.mean(off)), excess_kurtosis=float(stats.kurtosis(err.ravel()))))
        for noise, lst in sims.items():
            sim = np.concatenate(lst)                                          # (5N,H,8,19)
            for target, S, Y in (('cell', sim, truth), ('allcause', allcause(sim), truth_ac)):
                c = np.median(S, 0)
                for a in (0.80, 0.95):
                    half = (np.quantile(S, 1 - (1 - a) / 2, 0) - np.quantile(S, (1 - a) / 2, 0)) / 2
                    sc = np.abs(Y - c) / half
                    for h in range(H):
                        rows.append(pd.DataFrame(dict(noise=noise, target=target, popn=p, origin=o, h=h + 1, level=a,
                                                      tyear=o + h + 1, centre=c[h].ravel(), half=half[h].ravel(),
                                                      truth=Y[h].ravel(), score=sc[h].ravel())))
        print(p, o, flush=True)
S = pd.concat(rows, ignore_index=True); S.to_csv('results/rolling/resboot_scores.csv.gz', index=False)
DG = pd.DataFrame(diag); DG.to_csv('results/rolling/residual_diagnostics.csv', index=False)

Hh = np.arange(1, 6); out = []
for (noise, target, a), A in S.groupby(['noise', 'target', 'level']):
    for o in (2012, 2013, 2014):
        cal = A[(A.origin < o) & (A.tyear <= o)]; T = A[A.origin == o]
        qh = cal.groupby('h').score.quantile(a); g = float(np.polyfit(np.log(qh.index), np.log(qh.values), 1)[0])
        ks = {'uncal': np.ones(5), 'RT': np.full(5, np.quantile(cal.score, a)),
              'RT_gfit': np.quantile(cal.score / cal.h ** g, a) * Hh ** g}
        for nm, k in ks.items():
            kk = T.h.map(dict(zip(Hh, k))); lo, hi = T.centre - kk * T.half, T.centre + kk * T.half; al = 1 - a
            isc = (hi - lo) + 2 / al * ((lo - T.truth) * (T.truth < lo) + (T.truth - hi) * (T.truth > hi))
            out.append(dict(noise=noise, target=target, level=a, origin=o, scheme=nm,
                            coverage=float(((T.truth >= lo) & (T.truth <= hi)).mean()), width=float((hi - lo).mean()),
                            iscore=float(isc.mean()), k=float(k.mean())))
O = pd.DataFrame(out); O.to_csv('results/rolling/resboot_results.csv', index=False)
pd.set_option('display.width', 220)
print(O.groupby(['target', 'level', 'scheme', 'noise'])[['coverage', 'width', 'iscore', 'k']].mean().round(3).unstack('noise'))
print(DG.groupby('kind').mean(numeric_only=True).round(3))
