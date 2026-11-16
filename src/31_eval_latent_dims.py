"""Issue 1: does a 1-D latent suffice?  Compare K = 1, 2, 3 (rolling origins 2010-2014, seeds 0-2)
and K = 5 (origin 2014 only).  Latent forecasting: multivariate RW + drift (drift + innovation covariance,
drift parameter uncertainty).  Intervals: layer C (latent paths + 3-seed pooling + cell residual noise),
then real-time normalized-conformal calibration as in eval_calib.py."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, torch, warnings, os
from cae import CAE
warnings.filterwarnings('ignore'); torch.set_num_threads(2)
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']; SEEDS = range(3); H, N = 5, 600
rng = np.random.default_rng(7); R = 'models'; OUT = 'results/latent'
os.makedirs(OUT, exist_ok=True)

def tag(K): return '' if K == 1 else f'_K{K}'

def mrwd(mu):
    d = np.diff(mu, axis=0); n, K = d.shape; dbar = d.mean(0)
    S = np.cov(d.T).reshape(K, K) + 1e-10 * np.eye(K); L = np.linalg.cholesky(S)
    h = np.arange(1, H + 1)[:, None]
    point = mu[-1] + h * dbar
    dr = dbar + (rng.standard_normal((N, K)) @ L.T) / np.sqrt(n)
    inn = np.cumsum(rng.standard_normal((N, H, K)) @ L.T, 1)
    return point, mu[-1] + h[None] * dr[:, None] + inn

def dec(m, z):
    with torch.no_grad():
        return m.decode(torch.tensor(z.reshape(-1, z.shape[-1]), dtype=torch.float32))[:, 0].numpy()

pts, scores, interp = [], [], []
configs = [(K, o) for K in (1, 2, 3) for o in range(2010, 2015)] + [(5, 2014)]
for K, o in configs:
    for p in POPS:
        z = np.load(f'data/processed/data_{p}.npz'); years = list(z['years']); logm = z['logm']
        n_tr = years.index(o) + 1; truth = logm[n_tr:n_tr + H]
        P, sims, recm = [], [], []
        for s in SEEDS:
            m = CAE(latent=K); m.load_state_dict(torch.load(f'{R}/{p}_o{o}_s{s}{tag(K)}.pt')); m.eval()
            f = np.load(f'{R}/{p}_o{o}_s{s}{tag(K)}.npz'); mu = f['mu'].reshape(n_tr, K); rec = f['rec']
            recm.append(((logm[:n_tr] - rec) ** 2).mean()); sd = (logm[:n_tr] - rec).std(0)
            pt, pa = mrwd(mu)
            P.append(dec(m, pt)); sims.append(dec(m, pa).reshape(N, H, 8, 19) + rng.normal(size=(N, H, 8, 19)) * sd)
            if o == 2014 and s == 0:   # interpretability: beta per latent dim at last training year
                eps = 1e-3; base = mu[-1]
                B = np.stack([(dec(m, base + eps * np.eye(K)[k]) - dec(m, base - eps * np.eye(K)[k]))[0] / (2 * eps)
                              for k in range(K)])            # (K, 8, 19)
                energy = (B ** 2).sum(2); share = energy / energy.sum(0, keepdims=True)   # (K, 8)
                corr = np.corrcoef(np.diff(mu, axis=0).T) if K > 1 else np.ones((1, 1))
                interp.append(dict(K=K, pop=p, max_abs_latent_diff_corr=float(np.abs(corr[np.triu_indices(K, 1)]).max()) if K > 1 else np.nan,
                                   **{f'share_dim1_C{c+1}': float(share[0, c]) for c in range(8)}))
                np.save(f'{OUT}/beta_{p}_K{K}.npy', B)
        P = np.array(P); ens = P.mean(0)
        for hh in range(H):
            pts.append(dict(K=K, pop=p, origin=o, h=hh + 1, rec_mse=float(np.mean(recm)),
                            mse_ens=float(((ens[hh] - truth[hh]) ** 2).mean()),
                            mse_single=float(((P[:, hh] - truth[hh]) ** 2).mean()),
                            mae_ens=float(np.abs(ens[hh] - truth[hh]).mean())))
        sim = np.concatenate(sims); c = np.median(sim, 0)
        for a in (0.80, 0.95):
            half = (np.quantile(sim, 1 - (1 - a) / 2, 0) - np.quantile(sim, (1 - a) / 2, 0)) / 2
            sc = np.abs(truth - c) / half
            for hh in range(H):
                scores.append(pd.DataFrame(dict(K=K, popn=p, origin=o, h=hh + 1, level=a, target=o + hh + 1,
                    centre=c[hh].ravel(), half=half[hh].ravel(), truth=truth[hh].ravel(), score=sc[hh].ravel())))
        print(K, o, p, flush=True)
PT = pd.DataFrame(pts); PT.to_csv(f'{OUT}/point.csv', index=False)
SC = pd.concat(scores, ignore_index=True); SC.to_csv(f'{OUT}/scores.csv.gz', index=False)
pd.DataFrame(interp).to_csv(f'{OUT}/interp.csv', index=False)

# calibration (real-time, pooled pops) per K, test origins 2012-2014
res = []
for K in (1, 2, 3):
    for a in (0.80, 0.95):
        A = SC[(SC.K == K) & (SC.level == a)]
        for o in (2012, 2013, 2014):
            T = A[A.origin == o]; cal = A[(A.origin < o) & (A.target <= o)]
            for name, k in (('uncal', 1.0), ('RT', np.quantile(cal.score, a))):
                lo, hi = T.centre - k * T.half, T.centre + k * T.half; al = 1 - a
                isc = (hi - lo) + 2 / al * ((lo - T.truth) * (T.truth < lo) + (T.truth - hi) * (T.truth > hi))
                g = pd.DataFrame(dict(h=T.h, ins=(T.truth >= lo) & (T.truth <= hi), w=hi - lo, isc=isc)).groupby('h').mean()
                for hh, r in g.iterrows():
                    res.append(dict(K=K, level=a, origin=o, scheme=name, h=hh, k=k, coverage=r.ins, width=r.w, iscore=r.isc))
CR = pd.DataFrame(res); CR.to_csv(f'{OUT}/calib.csv', index=False)
pd.set_option('display.width', 250)
roll = PT[PT.K < 5]
print('\nRolling (origins 2010-14): mean over pops, origins, h\n',
      roll.groupby('K')[['rec_mse', 'mse_single', 'mse_ens', 'mae_ens']].mean().round(4))
print('\nEnsemble MSE by horizon\n', roll.pivot_table(index='h', columns='K', values='mse_ens').round(4))
print('\nEnsemble MSE by population\n', roll.pivot_table(index='pop', columns='K', values='mse_ens').round(4))
w = roll.groupby(['pop', 'origin', 'K']).mse_ens.mean().unstack(); print('\nWins over 25 pop x origin:\n', w.idxmin(1).value_counts())
print('\nPaper split (origin 2014), K=1,2,3,5\n', PT[PT.origin == 2014].groupby('K')[['rec_mse', 'mse_ens', 'mse_single']].mean().round(4))
print('\nCoverage/interval score (test origins 2012-14)\n',
      CR.groupby(['level', 'K', 'scheme']).agg(k=('k', 'mean'), cov=('coverage', 'mean'), width=('width', 'mean'), iscore=('iscore', 'mean')).round(3))
print('\nInterpretability (origin 2014, seed 0)\n', pd.DataFrame(interp).round(2).to_string(index=False))
