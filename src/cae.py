"""Reproduction of Tanaka & Matsuyama (2025) CAE.
Architecture follows Appendix B, with two fixes to match Table 1:
  * conv / deconv layers have intercepts (bias=True)
  * log-sigmoid output activation
Loss follows Eq. (9) literally: sum_t ||err_t||_F + lambda * (sum of L2 norms of each filter / affine weight).
"""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import sys, json, time
import numpy as np, torch
from torch import nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self, num_cause=8, num_age=19, num_conv=2, bias=True, out_channels=(1, 4, 8), latent=1):
        super().__init__()
        self.nc, self.na, self.ch, self.nconv = num_cause, num_age, list(out_channels), num_conv
        self.conv, self.deconv = nn.ModuleList(), nn.ModuleList()
        for i in range(num_conv):
            self.conv += [nn.Conv2d(self.ch[i], self.ch[i+1], (1, 3), padding=(0, 1), bias=bias), nn.Tanh(),
                          nn.AvgPool2d((1, 2))]
            self.deconv += [nn.ConvTranspose2d(self.ch[i+1], self.ch[i], (1, 3), padding=(0, 1), bias=bias), nn.Tanh(),
                            nn.Upsample(size=(num_cause, int(num_age * 0.5 ** i)), mode='bilinear')]
        self.deconv = nn.ModuleList(list(self.deconv)[::-1])
        self.na_low = int(num_age * 0.5 ** num_conv)
        self.num = self.na_low * num_cause * self.ch[num_conv]
        self.enc = nn.Linear(self.num, latent)
        self.dec = nn.Linear(latent, self.num)

    def encode(self, x):
        for l in self.conv: x = l(x)
        return torch.tanh(self.enc(x.flatten(1)))

    def decode(self, mu):
        x = torch.tanh(self.dec(mu)).reshape(-1, self.ch[self.nconv], self.nc, self.na_low)
        for l in self.deconv: x = l(x)
        # deconv list ends with Tanh after the last ConvTranspose? No: reversed order is
        # [Upsample, Tanh, ConvT(8->4), Upsample, Tanh, ConvT(4->1)] -> matches Table 1.
        return F.logsigmoid(x)

    def forward(self, x): return self.decode(self.encode(x))

    def penalty(self):
        p = 0.
        for l in self.conv:
            if isinstance(l, nn.Conv2d):  # weight (out, in, kh, kw): one filter per output channel
                p = p + l.weight.flatten(1).norm(dim=1).sum()
        for l in self.deconv:
            if isinstance(l, nn.ConvTranspose2d):  # weight (in, out, kh, kw)
                p = p + l.weight.transpose(0, 1).flatten(1).norm(dim=1).sum()
        return p + self.enc.weight.norm() + self.dec.weight.norm()


def run(name, seed, lam=5e-4, epochs=15000, lr=1e-3, n_test=5, latent=1):
    torch.manual_seed(seed); np.random.seed(seed)
    z = np.load(f'data/processed/data_{name}.npz')
    logm = torch.tensor(z['logm'], dtype=torch.float32)[:, None]  # (T, 1, cause, age)
    tr, te = logm[:-n_test], logm[-n_test:]
    m = CAE(latent=latent)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        err = (m(tr) - tr).flatten(1).norm(dim=1).sum()
        loss = err + lam * m.penalty()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        mu = m.encode(tr).numpy()  # (T_train, latent)
        rec_mse = float(((m(tr) - tr) ** 2).mean())
        res = {}
        dmu = np.diff(mu, axis=0)
        for kind, drift in [('rw_drift', dmu.mean(0)), ('rw_nodrift', 0 * dmu.mean(0))]:
            h = np.arange(1, n_test + 1)[:, None]
            mu_f = mu[-1] + h * drift
            pred = m.decode(torch.tensor(mu_f, dtype=torch.float32))
            res[kind] = dict(mse=float(((pred - te) ** 2).mean()), mae=float((pred - te).abs().mean()))
        # age sensitivities beta_{x,c}^{(t)} via numerical differentiation of decoder (Eq. 10), training years
        eps = 1e-3
        mu_t = torch.tensor(mu, dtype=torch.float32)
        beta = ((m.decode(mu_t + eps) - m.decode(mu_t - eps)) / (2 * eps))[:, 0].numpy()
    return dict(name=name, seed=seed, rec_mse=rec_mse, final_loss=float(loss), **res), mu, beta


if __name__ == '__main__':
    torch.set_num_threads(1)
    name, seeds = sys.argv[1], [int(s) for s in sys.argv[2].split(',')]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 15000
    for s in seeds:
        t0 = time.time()
        r, mu, beta = run(name, s, epochs=epochs)
        r['secs'] = round(time.time() - t0, 1)
        np.savez(f'results/reproduction/{name}_s{s}.npz', mu=mu, beta=beta)
        with open('results/reproduction/results.jsonl', 'a') as f: f.write(json.dumps(r) + '\n')
        print(json.dumps(r), flush=True)
