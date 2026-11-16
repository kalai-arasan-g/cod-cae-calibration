"""Rolling-origin training: for each population, train the paper's CAE on data up to
each origin year (2010..2014) and save weights + latent series for offline forecasting."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import sys, time, json, numpy as np, torch
from cae import CAE

def train(logm_tr, seed, K=1, lam=5e-4, epochs=15000, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    tr = torch.tensor(logm_tr, dtype=torch.float32)[:, None]
    m = CAE(latent=K); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = (m(tr) - tr).flatten(1).norm(dim=1).sum() + lam * m.penalty()
        loss.backward(); opt.step()
    return m

if __name__ == '__main__':
    torch.set_num_threads(1)
    arg = sys.argv[1]
    if arg.startswith('@'): arg = open(arg[1:]).read().strip()  # @file with jobs
    jobs = [j.split(':') for j in arg.split(';') if j]  # "POP:origin:seed;..."
    for job in jobs:
        name, origin, seed = job[:3]; K = int(job[3]) if len(job) > 3 else 1
        lam = float(job[4]) if len(job) > 4 else 5e-4
        origin, seed = int(origin), int(seed)
        tag = ('' if K == 1 else f'_K{K}') + ('' if lam == 5e-4 else f'_L{lam:g}')
        z = np.load(f'data/processed/data_{name}.npz'); years = list(z['years'])
        n_tr = years.index(origin) + 1
        t0 = time.time()
        m = train(z['logm'][:n_tr], seed, K, lam)
        with torch.no_grad():
            tr = torch.tensor(z['logm'][:n_tr], dtype=torch.float32)[:, None]
            mu = m.encode(tr).numpy(); rec = m(tr)[:, 0].numpy()
        torch.save(m.state_dict(), f'models/{name}_o{origin}_s{seed}{tag}.pt')
        np.savez(f'models/{name}_o{origin}_s{seed}{tag}.npz', mu=mu, rec=rec)
        print(json.dumps(dict(name=name, origin=origin, seed=seed, K=K, lam=lam, secs=round(time.time() - t0, 1))), flush=True)
