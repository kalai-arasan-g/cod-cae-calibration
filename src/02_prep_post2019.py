"""Post-2019 truth (log death rates) for the COVID-period test. Only years with both WHO deaths and WHO
population are kept: Japan 2020-2023, UK 2020, 2023, 2024 (no WHO population for 2021-2022), Germany 2020
(2021-2022 deaths exist in the WHO add-on file but WHO population does not)."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, json
exec(open('src/01_prep_data.py').read().split("d=pd.read_csv(RAW")[0])
d = pd.read_csv('data/processed/who_selected.csv', dtype=str); p = pd.read_csv('data/raw/pop', dtype=str)
rep = {}
for name, (cty, sex, y0, y1) in POPS.items():
    dd = d[(d.Country == cty) & (d.Sex == sex)]
    tot = dd[dd.Cause == 'AAA']; det = dd[dd.Cause != 'AAA'].copy(); det['g'] = det.Cause.map(cause_group)
    yrs, L, Dd = [], [], []
    for y in range(2020, 2025):
        s = det[det.Year == str(y)]; pp = p[(p.Country == cty) & (p.Sex == sex) & (p.Year == str(y))]
        if len(s) == 0 or len(pp) != 1: continue
        D = np.stack([to19(s[s.g == g], 'Deaths').sum(0) for g in range(8)])
        t = to19(tot[tot.Year == str(y)], 'Deaths').sum(0); assert np.allclose(D.sum(0), t), (name, y)
        E = to19(pp, 'Pop')[0]; yrs.append(y); L.append(np.log(D / E)); Dd.append(D)
    Dd = np.array(Dd); rep[name] = dict(years=yrs, zero_cells=int((Dd == 0).sum()),
        covid_U07_share_C8=None)
    np.savez(f'data/processed/data_{name}_post.npz', years=yrs, logm=np.array(L), deaths=Dd)
print(json.dumps(rep))
