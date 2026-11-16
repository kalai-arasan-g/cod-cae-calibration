"""Rolling-origin evaluation of latent time-series models and prediction intervals.

For each population p, origin o (last training year 2010..2014), seed s:
  - latent series mu_1..mu_n from the trained CAE
  - 4 latent forecasting models: RW, RWD (paper), ARIMA(1,1,0)+drift, damped drift (ETS AAdN)
  - N Monte Carlo latent paths per model -> decoder -> log death rates for h=1..5
Interval layers:
  A  latent-only (single seed)
  B  latent + seed ensemble (pool paths from all 5 seeds)
  C  B + cell-level residual noise (per cause x age, from training reconstruction residuals)
Point forecasts: single-seed deterministic decode; and 5-seed ensemble mean of log rates.
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import json, warnings, numpy as np, pandas as pd, torch
from cae import CAE
warnings.filterwarnings('ignore')
torch.set_num_threads(2)
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']; ORIGINS = range(2010, 2015); SEEDS = range(5)
H, N = 5, 1000
rng = np.random.default_rng(12345)
R = 'models'

def latent_paths(mu, model):
    """Returns (point[H], paths[N,H]) for a latent series mu (length n)."""
    d = np.diff(mu); n = len(d); h = np.arange(1, H + 1)
    if model == 'RW':
        sig = np.sqrt(np.mean(d ** 2))
        point = np.full(H, mu[-1])
        paths = mu[-1] + np.cumsum(rng.normal(0, sig, (N, H)), 1)
    elif model == 'RWD':
        dbar, sig = d.mean(), d.std(ddof=1)
        point = mu[-1] + h * dbar
        dr = rng.normal(dbar, sig / np.sqrt(n), (N, 1))           # drift (parameter) uncertainty
        paths = mu[-1] + h * dr + np.cumsum(rng.normal(0, sig, (N, H)), 1)
    elif model == 'ARIMA110':
        x, y = d[:-1], d[1:]
        X = np.c_[np.ones_like(x), x]; b = np.linalg.lstsq(X, y, rcond=None)[0]
        b[1] = np.clip(b[1], -0.95, 0.95)
        res = y - X @ b; sig = res.std(ddof=2)
        def roll(eps):
            out, last, cur = [], d[-1], mu[-1]
            for k in range(H):
                last = b[0] + b[1] * last + eps[..., k]; cur = cur + last; out.append(cur)
            return np.stack(out, -1)
        point = roll(np.zeros(H)); paths = roll(rng.normal(0, sig, (N, H)))
    elif model == 'DAMPED':
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel
        f = ETSModel(mu, error='add', trend='add', damped_trend=True).fit(disp=False)
        point = np.asarray(f.forecast(H))
        paths = np.asarray(f.simulate(H, repetitions=N, anchor='end')).T
    return point, paths

def decode(m, z):
    with torch.no_grad():
        return m.decode(torch.tensor(z.reshape(-1, 1), dtype=torch.float32))[:, 0].numpy()

MODELS = ['RW', 'RWD', 'ARIMA110', 'DAMPED']
point_rows, int_rows = [], []
for p in POPS:
    z = np.load(f'data/processed/data_{p}.npz'); years = list(z['years']); logm = z['logm']
    for o in ORIGINS:
        n_tr = years.index(o) + 1
        truth = logm[n_tr:n_tr + H]                                      # (H, 8, 19)
        per_model = {k: dict(points=[], paths=[], noise=[]) for k in MODELS}
        for s in SEEDS:
            m = CAE(); m.load_state_dict(torch.load(f'{R}/{p}_o{o}_s{s}.pt')); m.eval()
            o_ = np.load(f'{R}/{p}_o{o}_s{s}.npz'); mu, rec = o_['mu'], o_['rec']
            cell_sd = (logm[:n_tr] - rec).std(0)                           # (8,19) residual sd
            for k in MODELS:
                pt, pa = latent_paths(mu, k)
                P = decode(m, pt)                                         # (H,8,19)
                PA = decode(m, pa).reshape(N, H, 8, 19)
                per_model[k]['points'].append(P); per_model[k]['paths'].append(PA)
                per_model[k]['noise'].append(cell_sd)
                err = P - truth
                for hh in range(H):
                    point_rows.append(dict(pop=p, origin=o, seed=s, model=k, h=hh + 1, kind='single',
                                           mse=float((err[hh] ** 2).mean()), mae=float(np.abs(err[hh]).mean())))
        for k in MODELS:
            pts = np.array(per_model[k]['points']); ens = pts.mean(0); err = ens - truth
            for hh in range(H):
                point_rows.append(dict(pop=p, origin=o, seed=-1, model=k, h=hh + 1, kind='ensemble',
                                       mse=float((err[hh] ** 2).mean()), mae=float(np.abs(err[hh]).mean())))
            paths = np.array(per_model[k]['paths'])                       # (S,N,H,8,19)
            sds = np.array(per_model[k]['noise'])                         # (S,8,19)
            layers = {
                'A_latent': paths[0],
                'B_latent+seeds': paths.reshape(-1, H, 8, 19),
                'C_latent+seeds+cell': (paths + rng.normal(size=paths.shape) * sds[:, None, None]).reshape(-1, H, 8, 19),
            }
            for lname, sim in layers.items():
                for a in (0.80, 0.95):
                    lo = np.quantile(sim, (1 - a) / 2, 0); hi = np.quantile(sim, 1 - (1 - a) / 2, 0)
                    inside = (truth >= lo) & (truth <= hi)
                    alpha = 1 - a
                    score = (hi - lo) + 2 / alpha * ((lo - truth) * (truth < lo) + (truth - hi) * (truth > hi))
                    for hh in range(H):
                        int_rows.append(dict(pop=p, origin=o, model=k, layer=lname, level=a, h=hh + 1,
                                             coverage=float(inside[hh].mean()), width=float((hi - lo)[hh].mean()),
                                             interval_score=float(score[hh].mean())))
        print(p, o, flush=True)
pd.DataFrame(point_rows).to_csv('results/rolling/point_metrics.csv', index=False)
pd.DataFrame(int_rows).to_csv('results/rolling/interval_metrics.csv', index=False)
print('done')
