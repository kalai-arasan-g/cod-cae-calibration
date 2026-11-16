"""Sensitivity data for the COVID-19 test: UK 2021-2022 with WHO deaths and population interpolated
log-linearly by age group between WHO population for 2020 and 2023 (WHO has no UK population for 2021-2022).
Other populations are unchanged."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import numpy as np, pandas as pd, json
exec(open('src/01_prep_data.py').read().split("d=pd.read_csv(RAW")[0])
d = pd.read_csv('data/processed/who_selected.csv', dtype=str); p = pd.read_csv('data/raw/pop', dtype=str)
rep = {}
for name, (cty, sex, y0, y1) in POPS.items():
    base = np.load(f'data/processed/data_{name}_post.npz')
    yrs, L, Dd, flag = list(base['years']), list(base['logm']), list(base['deaths']), [0] * len(base['years'])
    if cty == '4308':
        dd = d[(d.Country == cty) & (d.Sex == sex)]; det = dd[dd.Cause != 'AAA'].copy(); det['g'] = det.Cause.map(cause_group)
        tot = dd[dd.Cause == 'AAA']
        E20 = to19(p[(p.Country == cty) & (p.Sex == sex) & (p.Year == '2020')], 'Pop')[0]
        E23 = to19(p[(p.Country == cty) & (p.Sex == sex) & (p.Year == '2023')], 'Pop')[0]
        for y in (2021, 2022):
            w = (y - 2020) / 3; E = np.exp((1 - w) * np.log(E20) + w * np.log(E23))
            s = det[det.Year == str(y)]; D = np.stack([to19(s[s.g == g], 'Deaths').sum(0) for g in range(8)])
            assert np.allclose(D.sum(0), to19(tot[tot.Year == str(y)], 'Deaths').sum(0))
            yrs.append(y); L.append(np.log(D / E)); Dd.append(D); flag.append(1)
        o = np.argsort(yrs); yrs = [yrs[i] for i in o]; L = [L[i] for i in o]; Dd = [Dd[i] for i in o]; flag = [flag[i] for i in o]
    rep[name] = dict(years=yrs, interpolated=flag, zero_cells=int((np.array(Dd) == 0).sum()))
    np.savez(f'data/processed/data_{name}_postI.npz', years=yrs, logm=np.array(L), deaths=np.array(Dd), interp=flag)
print(json.dumps(rep, default=int))
