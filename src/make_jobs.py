"""Writes the training job lists used in the paper to jobs/*.txt (format POP:origin:seed[:K[:lambda]]).
Each list is split in two halves (_a, _b) so that two processes can run in parallel."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1]); _os.makedirs('jobs', exist_ok=True)
P = ['JPN_M', 'JPN_F', 'GBR_M', 'GBR_F', 'DEU_M']
sets = {
    'rolling_K1': [f'{p}:{o}:{s}' for s in range(5) for o in range(2010, 2015) for p in P],          # 125 models
    'latent_K2_K3_K5': [f'{p}:2014:{s}:5' for s in range(3) for p in P] +
                       [f'{p}:{o}:{s}:{K}' for K in (2, 3) for s in range(3) for o in range(2014, 2009, -1) for p in P],  # 165
    'lambda_check': [f'{p}:{o}:{s}:{K}:{l}' for l in ('0.005', '0.05') for K in (2, 3) for s in (0, 1)
                     for o in (2014, 2012, 2010) for p in P],                                         # 120
    'covid_origin2019': [f'{p}:2019:{s}' for s in range(5) for p in P],                               # 25
}
for name, jobs in sets.items():
    for half, js in (('a', jobs[0::2]), ('b', jobs[1::2])):
        open(f'jobs/{name}_{half}.txt', 'w').write(';'.join(js))
    print(name, len(jobs), 'models')
