# Calibrated uncertainty for interpretable neural cause-of-death mortality forecasts

Code, processed data and results for the paper

> Kalaiarasan G. *Calibrated prediction intervals for an interpretable neural network approach to cause-of-death mortality forecasting* Submitted to *Annals of Actuarial Science*.

**Author:** Kalaiarasan G, Amrita School of Artificial Intelligence, Coimbatore, Amrita Vishwa Vidyapeetham, India  
**Contact:** kalai.forward@gmail.com

The paper reproduces the convolutional autoencoder (CAE) of Tanaka & Matsuyama (2025, *Annals of Actuarial Science* 19, 442–461) with WHO data for five populations (Japan males and females, UK males and females, German males). It then:

- evaluates the model at 25 rolling forecast origins (training data ending 2010–2014);
- compares four time-series models for the latent variable and the effect of seed averaging;
- builds prediction intervals and calibrates them with a real-time normalised conformal method;
- tests models with 2, 3 and 5 latent variables, with a regularisation check; and
- tests the calibrated model on 2020–2024 data, which include the COVID-19 pandemic.

## Repository structure

```
src/                 all scripts, numbered in the order they are run
  cae.py             the CAE model (Appendix B of Tanaka & Matsuyama, with Table 1 corrections);
                     run directly for the reproduction study
  train_rolling.py   trains models for a list of jobs (population:origin:seed[:K[:lambda]])
  make_jobs.py       writes the job lists used in the paper to jobs/
data/raw/            WHO raw files go here (not included, see data/raw/README.md)
data/processed/      log death rates by year x 8 causes x 19 ages for each population (.npz)
models/              trained models (not in git; see "Trained models" below)
results/             all result tables (.csv, .json) used in the paper
figures/             Figures 1-6 of the paper (PDF)
run_all.sh           runs the full pipeline
```

## Setup

Python 3.11 or later.

```bash
pip install -r requirements.txt
```

A CPU is enough; a GPU is not needed.

## Data

The WHO raw files are not included. Download `Morticd10_part1` to `part6` and `pop` from the
[WHO Mortality Database](https://www.who.int/data/data-collection-tools/who-mortality-database)
and unzip them into `data/raw/` (details in `data/raw/README.md`).

The processed arrays in `data/processed/` are included, so steps 2 onwards can be run without
downloading the raw files. ICD-10 codes are grouped by chapter into eight causes:
C1 infectious (A00–B99), C2 neoplasms (C00–D48), C3 nervous system (G00–G99),
C4 circulatory (I00–I99), C5 respiratory (J00–J99), C6 congenital (Q00–Q99),
C7 external causes (V01–Y98), C8 all others (including U07, COVID-19). The ages are
0, 1–4, 5–9, …, 80–84, 85+.

| File | Content |
| --- | --- |
| `data_<POP>.npz` | years up to 2019 (`logm`, `deaths`, `expo`, `years`) |
| `data_<POP>_post.npz` | years after 2019 with WHO deaths and population |
| `data_<POP>_postI.npz` | as above, plus UK 2021–2022 with interpolated population (sensitivity check) |

`<POP>` is one of `JPN_M`, `JPN_F`, `GBR_M`, `GBR_F`, `DEU_M`.

## Running

The whole pipeline:

```bash
./run_all.sh
```

This trains 485 models (50 for the reproduction and 435 for the extensions) and takes roughly 8–10 hours on two CPU cores (50–70 seconds per
model). The steps can also be run one at a time; each script reads and writes paths relative to
the repository root.

| Step | Script | Paper |
| --- | --- | --- |
| 0–3 | `00_select_who.py`, `01_prep_data.py`, `02_prep_post2019.py`, `03_prep_post2019_uk_interp.py` | Section 2, Table 1 |
| Reproduction | `cae.py <POP> <seeds>`, `11_analyze_reproduction.py` | Section 4.1, Table 2, Figures 1–2 |
| Training | `make_jobs.py`, `train_rolling.py @jobs/<list>.txt` | Sections 3.2, 3.6 |
| Rolling accuracy | `21_eval_rolling.py`, `22_summarize_rolling.py` | Section 4.2, Table 3, Figure 3 |
| Calibration | `23_eval_calibration.py`, `24_eval_calibration_horizon.py` | Section 4.3, Table 4, Figure 4 |
| Dependence checks | `25_eval_window_bootstrap.py`, `26_eval_residual_bootstrap.py` | Section 4.3 |
| Latent dimension | `31_eval_latent_dims.py`, `32_eval_lambda.py` | Section 4.4, Tables 5–6, Figure 5 |
| COVID-19 test | `41_eval_covid.py`, `41_eval_covid.py --interp`, `43_covid_balance.py` | Section 4.5, Table 7, Figure 6 |
| Figures | `51_make_figures.py` | Figures 1–6 |

## Reproducibility

- Training is deterministic on CPU for a given seed: retraining a model in a fresh copy of this
  repository gave identical weights.
- The settings follow Tanaka & Matsuyama (2025): λ = 0.0005, Adam with learning rate 0.001,
  15,000 full-batch epochs, last 5 years as test data.
- Two corrections to the published code (Appendix B) follow the paper's Table 1: the
  convolution layers have intercepts, and the output passes through a log-sigmoid.
- WHO revises its database. Results from a later download may differ slightly from the paper.

## Trained models

The 435 trained models used for Sections 4.2–4.5 (about 11 MB) are not stored in git. They are attached to the GitHub
release [v1.0](https://github.com/kalai-arasan-g/cod-cae-calibration/releases/tag/v1.0) as `models.zip`; unzip into `models/` to run steps 21 onwards without retraining.

## Licence

Code: MIT licence (see `LICENSE`). The processed data are derived from the WHO Mortality
Database and remain subject to WHO's terms of use; please cite WHO as the data source.

## Citation

If you use this code, please cite the paper above and the original model:

Tanaka, S. & Matsuyama, N. (2025). An interpretable neural network approach to cause-of-death
mortality forecasting. *Annals of Actuarial Science*, 19, 442–461.
https://doi.org/10.1017/S1748499524000319
