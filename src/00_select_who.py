"""Step 0: read the WHO Mortality Database raw ICD-10 files (Morticd10_part1..part6) from data/raw/ and keep
Japan (3160), United Kingdom (4308) and Germany (4085). Output: data/processed/who_selected.csv."""
import os as _os, pathlib as _pl
_os.chdir(_pl.Path(__file__).resolve().parents[1])
import glob, pandas as pd
files = sorted(glob.glob('data/raw/Morticd10_part*'))
assert files, 'Put the unzipped WHO files Morticd10_part1 ... part6 and pop in data/raw/ (see data/raw/README.md)'
keep = []
for f in files:
    for ch in pd.read_csv(f, dtype=str, chunksize=500_000):
        keep.append(ch[ch.Country.isin(['3160', '4308', '4085'])])
d = pd.concat(keep); _os.makedirs('data/processed', exist_ok=True)
d.to_csv('data/processed/who_selected.csv', index=False)
print(len(d), 'rows;', d.groupby('Country').Year.agg(['min', 'max']).to_dict('index'))
