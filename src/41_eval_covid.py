"""COVID-period test: models trained on data through 2019 (5 seeds), forecasts for 2020-2024 where WHO
data exist. Intervals: layer C (RW+drift latent paths with drift uncertainty, 5 seeds pooled, cell residual
noise), rescaled by k estimated ONLY from forecast errors observable by 2019 (origins 2010-2014, h=1..5):
  uncal, RT (constant k), RT_gfit (k0*h^gamma, gamma fitted), RT_h (horizon-specific k_h).
A cell with zero deaths (Japan female, infectious, age 10-14, 2020) is excluded."""
import sys
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)
INTERP = '--interp' in sys.argv  # add UK 2021-22 with interpolated population
SUF, TAG = ('_postI', '_interp') if INTERP else ('_post', '')

import numpy as np, pandas as pd, torch
from cae import CAE
torch.set_num_threads(2); rng = np.random.default_rng(99)
POPS = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']; N = 1000; H = 5; R = 'models'
CAUSES = ['Infectious', 'Neoplasms', 'Nervous', 'Circulatory', 'Respiratory', 'Congenital', 'External', 'Others']
S = pd.read_csv('results/rolling/calib_scores.csv.gz')
K = {}
for a in (0.80, 0.95):
    A = S[S.level == a]
    qh = A.groupby('h').score.quantile(a); g = float(np.polyfit(np.log(qh.index), np.log(qh.values), 1)[0])
    k0 = np.quantile(A.score / A.h ** g, a); h = np.arange(1, H + 1)
    K[a] = {'uncal': np.ones(H), 'RT': np.full(H, np.quantile(A.score, a)), 'RT_gfit': k0 * h ** g, 'RT_h': qh.values}
    print(a, {k: np.round(v, 3).tolist() for k, v in K[a].items()}, 'gamma', round(g, 3))

pts, cov, bycause = [], [], []
for p in POPS:
    z = np.load(f'data/processed/data_{p}.npz'); logm = z['logm']
    post = np.load(f'data/processed/data_{p}{SUF}.npz'); yrs = list(post['years']); truth = post['logm']
    hs = [y - 2019 for y in yrs]
    sims, P = [], []
    for s in range(5):
        m = CAE(); m.load_state_dict(torch.load(f'{R}/{p}_o2019_s{s}.pt')); m.eval()
        f = np.load(f'{R}/{p}_o2019_s{s}.npz'); mu = f['mu'].reshape(-1); sd = (logm - f['rec']).std(0)
        d = np.diff(mu); n = len(d); dbar, sig = d.mean(), d.std(ddof=1); hh = np.arange(1, H + 1)
        paths = mu[-1] + hh * rng.normal(dbar, sig / np.sqrt(n), (N, 1)) + np.cumsum(rng.normal(0, sig, (N, H)), 1)
        with torch.no_grad():
            P.append(m.decode(torch.tensor((mu[-1] + hh * dbar)[:, None], dtype=torch.float32))[:, 0].numpy())
            pa = m.decode(torch.tensor(paths.reshape(-1, 1), dtype=torch.float32))[:, 0].numpy().reshape(N, H, 8, 19)
        sims.append(pa + rng.normal(size=pa.shape) * sd)
    ens = np.mean(P, 0); sim = np.concatenate(sims); c = np.median(sim, 0)
    for i, (y, hv) in enumerate(zip(yrs, hs)):
        tr = truth[i]; ok = np.isfinite(tr); e = (ens[hv - 1] - tr)[ok]
        pts.append(dict(pop=p, year=y, h=hv, mse=float((e ** 2).mean()), mae=float(np.abs(e).mean()),
                        bias=float((tr - ens[hv - 1])[ok].mean()), cells=int(ok.sum())))
        for ci in range(8):
            okc = ok[ci]; ec = (tr[ci] - ens[hv - 1, ci])[okc]
            bycause.append(dict(pop=p, year=y, cause=CAUSES[ci], mse=float((ec ** 2).mean()), bias=float(ec.mean())))
        for a in (0.80, 0.95):
            half = (np.quantile(sim[:, hv - 1], 1 - (1 - a) / 2, 0) - np.quantile(sim[:, hv - 1], (1 - a) / 2, 0)) / 2
            for name, kv in K[a].items():
                k = kv[hv - 1]; lo, hi = c[hv - 1] - k * half, c[hv - 1] + k * half
                ins = ((tr >= lo) & (tr <= hi))[ok]; al = 1 - a
                isc = ((hi - lo) + 2 / al * ((lo - tr) * (tr < lo) + (tr - hi) * (tr > hi)))[ok]
                cov.append(dict(pop=p, year=y, h=hv, level=a, scheme=name, coverage=float(ins.mean()),
                                width=float((2 * k * half)[ok].mean()), iscore=float(isc.mean())))
PT, CV, BC = pd.DataFrame(pts), pd.DataFrame(cov), pd.DataFrame(bycause)
for nm, df in (('covid_point', PT), ('covid_coverage', CV), ('covid_bycause', BC)):
    df.to_csv(f'results/rolling/{nm}{TAG}.csv', index=False)
pd.set_option('display.width', 220)
print(PT.round(4).to_string(index=False))
print(CV[CV.level == 0.95].pivot_table(index=['pop', 'year'], columns='scheme', values='coverage').round(3))
print(CV.groupby(['level', 'scheme'])[['coverage', 'width', 'iscore']].mean().round(3))
print(BC.groupby('cause')[['mse', 'bias']].mean().round(3))
print(BC[BC.year <= 2021].pivot_table(index='cause', columns=['pop', 'year'], values='bias').round(2))
