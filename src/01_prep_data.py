"""Build log death-rate arrays (year x cause x age) for the 5 populations."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])  # run from the repository root
for _d in ('data/processed', 'models', 'results/reproduction', 'results/rolling', 'results/latent', 'figures'):
    _os.makedirs(_d, exist_ok=True)

import pandas as pd, numpy as np, re, json
RAW='data/processed/who_selected.csv'; POP='data/raw/pop'
POPS={'JPN_M':('3160','1',1995,2019),'JPN_F':('3160','2',1995,2019),
      'GBR_M':('4308','1',2001,2019),'GBR_F':('4308','2',2001,2019),
      'DEU_M':('4085','1',1998,2019)}
CAUSES=['Infectious','Neoplasms','Nervous','Circulatory','Respiratory','Congenital','External','Others']
AGES=['0','1-4']+[f'{a}-{a+4}' for a in range(5,85,5)]+['85+']
def cause_group(code):
    l=code[0]; n=int(code[1:3]) if code[1:3].isdigit() else -1
    if l in 'AB': return 0
    if l=='C' or (l=='D' and n<=48): return 1
    if l=='G': return 2
    if l=='I': return 3
    if l=='J': return 4
    if l=='Q': return 5
    if l in 'VWXY': return 6
    return 7
def to19(df,pfx):
    c=lambda k: df[f'{pfx}{k}'].astype(float).fillna(0)
    cols=[c(2), c(3)+c(4)+c(5)+c(6)]+[c(k) for k in range(7,23)]+[c(23)+c(24)+c(25)]
    return np.stack([x.values for x in cols],1)
d=pd.read_csv(RAW,dtype=str); p=pd.read_csv(POP,dtype=str)
out={}; report={}
for name,(cty,sex,y0,y1) in POPS.items():
    dd=d[(d.Country==cty)&(d.Sex==sex)]
    tot=dd[dd.Cause=='AAA']; det=dd[dd.Cause!='AAA'].copy()
    det['g']=det.Cause.map(cause_group)
    years=list(range(y0,y1+1)); D=np.zeros((len(years),8,19)); E=np.zeros((len(years),19)); chk=[]
    for i,y in enumerate(years):
        s=det[det.Year==str(y)]
        for g in range(8): D[i,g]=to19(s[s.g==g],'Deaths').sum(0)
        t=to19(tot[tot.Year==str(y)],'Deaths').sum(0)
        chk.append(abs(D[i].sum(0)-t).max()/t.max())
        pp=p[(p.Country==cty)&(p.Sex==sex)&(p.Year==str(y))]
        assert len(pp)==1,(name,y,len(pp))
        E[i]=to19(pp,'Pop')[0]
    zeros=int((D==0).sum())
    report[name]=dict(years=f'{y0}-{y1}',T=len(years),max_rel_diff_vs_AAA=float(max(chk)),zero_cells=zeros,
        pop_frmats=sorted(p[(p.Country==cty)&(p.Sex==sex)&p.Year.astype(int).between(y0,y1)].Frmat.unique().tolist()))
    np.savez(f'data/processed/data_{name}.npz',deaths=D,expo=E[:,None,:].repeat(8,1),years=years,
             logm=np.log(D/E[:,None,:]))
print(json.dumps(report,indent=1))
