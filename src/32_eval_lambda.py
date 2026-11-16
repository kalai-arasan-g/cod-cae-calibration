"""Regularisation check: K=2,3 with lambda in {0.0005, 0.005, 0.05} vs K=1 (lambda=0.0005).
Origins 2010, 2012, 2014; seeds 0,1; 2-seed ensemble point forecasts with multivariate RW+drift; h=1..5."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, torch
from cae import CAE
torch.set_num_threads(2)
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']; R = 'models'; H = 5

def tag(K, lam):
    return ('' if K == 1 else f'_K{K}') + ('' if lam == 5e-4 else f'_L{lam:g}')

rows = []
for K, lams in [(1, [5e-4]), (2, [5e-4, 5e-3, 5e-2]), (3, [5e-4, 5e-3, 5e-2])]:
    for lam in lams:
        for p in POPS:
            z = np.load(f'data/processed/data_{p}.npz'); years = list(z['years']); logm = z['logm']
            for o in (2010, 2012, 2014):
                n_tr = years.index(o) + 1; truth = logm[n_tr:n_tr + H]; preds, recs = [], []
                for s in (0, 1):
                    m = CAE(latent=K); m.load_state_dict(torch.load(f'{R}/{p}_o{o}_s{s}{tag(K, lam)}.pt')); m.eval()
                    f = np.load(f'{R}/{p}_o{o}_s{s}{tag(K, lam)}.npz'); mu = f['mu'].reshape(n_tr, K)
                    recs.append(((logm[:n_tr] - f['rec']) ** 2).mean())
                    d = np.diff(mu, axis=0).mean(0); fut = mu[-1] + np.arange(1, H + 1)[:, None] * d
                    with torch.no_grad():
                        preds.append(m.decode(torch.tensor(fut, dtype=torch.float32))[:, 0].numpy())
                ens = np.mean(preds, 0)
                for hh in range(H):
                    rows.append(dict(K=K, lam=lam, pop=p, origin=o, h=hh + 1, rec_mse=float(np.mean(recs)),
                                     mse=float(((ens[hh] - truth[hh]) ** 2).mean())))
D = pd.DataFrame(rows); D.to_csv('results/latent/lambda_check.csv', index=False)
pd.set_option('display.width', 200)
S = D.groupby(['K', 'lam']).agg(rec_mse=('rec_mse', 'mean'), mse=('mse', 'mean')).reset_index()
S['mse_h5'] = D[D.h == 5].groupby(['K', 'lam']).mse.mean().values
print(S.round(4).to_string(index=False))
w = D.groupby(['pop', 'origin', 'K', 'lam']).mse.mean().reset_index()
base = w[w.K == 1].set_index(['pop', 'origin']).mse
for (K, lam), g in w[w.K > 1].groupby(['K', 'lam']):
    beat = (g.set_index(['pop', 'origin']).mse < base).sum()
    print(f'K={K} lam={lam:g}: beats K=1 in {beat}/15 windows')
best = w[w.K > 1].groupby(['pop', 'origin']).mse.min()
print('oracle best-of-6 (K>1, any lambda) mean MSE', round(best.mean(), 4), 'vs K=1', round(base.mean(), 4),
      '; oracle beats K=1 in', int((best < base).sum()), '/15')
