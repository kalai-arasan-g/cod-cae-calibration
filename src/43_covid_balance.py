import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)
import numpy as np, pandas as pd, json
rng = np.random.default_rng(3)
C = pd.read_csv('results/rolling/covid_coverage.csv'); CI = pd.read_csv('results/rolling/covid_coverage_interp.csv')
P = pd.read_csv('results/rolling/covid_point_interp.csv')
out = {}
for lvl in (0.80, 0.95):
    c = C[C.level == lvl]; ci = CI[CI.level == lvl]
    piv = c.pivot_table(index=['pop', 'year'], columns='scheme', values='coverage')
    out[f'{lvl}_per_pop'] = c.groupby(['pop', 'scheme']).coverage.mean().unstack().round(3).to_dict('index')
    out[f'{lvl}_year2020_all5'] = c[c.year == 2020].groupby('scheme').coverage.mean().round(3).to_dict()
    out[f'{lvl}_year2020_min'] = c[c.year == 2020].groupby('scheme').coverage.min().round(3).to_dict()
    out[f'{lvl}_after2020'] = c[c.year > 2020].groupby('scheme').coverage.mean().round(3).to_dict()
    out[f'{lvl}_cell_weighted_mean'] = c.groupby('scheme').coverage.mean().round(3).to_dict()
    out[f'{lvl}_pop_equal_weight'] = c.groupby(['scheme', 'pop']).coverage.mean().groupby('scheme').mean().round(3).to_dict()
    # bootstrap over population-years (15 clusters)
    boot = {s: [] for s in piv.columns}
    for b in range(5000):
        idx = rng.integers(0, len(piv), len(piv))
        for s in piv.columns: boot[s].append(piv[s].values[idx].mean())
    out[f'{lvl}_popyear_boot90'] = {s: [round(float(np.quantile(v, .05)), 3), round(float(np.quantile(v, .95)), 3)] for s, v in boot.items()}
    # bootstrap over populations (5 clusters), equal weight to each population's mean
    pm = c.groupby(['pop', 'scheme']).coverage.mean().unstack()
    bp = {s: [] for s in pm.columns}
    for b in range(5000):
        idx = rng.integers(0, 5, 5)
        for s in pm.columns: bp[s].append(pm[s].values[idx].mean())
    out[f'{lvl}_pop_boot90'] = {s: [round(float(np.quantile(v, .05)), 3), round(float(np.quantile(v, .95)), 3)] for s, v in bp.items()}
    uk = ci[ci['pop'].str.startswith('GBR') & ci.year.isin([2021, 2022])]
    out[f'{lvl}_UK_2021_22_interp'] = uk.pivot_table(index=['pop', 'year'], columns='scheme', values='coverage').round(3).reset_index().to_dict('records')
    out[f'{lvl}_all_with_interp_mean'] = ci.groupby('scheme').coverage.mean().round(3).to_dict()
out['UK_interp_point'] = P[P['pop'].str.startswith('GBR')][['pop', 'year', 'h', 'mse']].round(4).to_dict('records')
json.dump(out, open('results/rolling/covid_balance.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
