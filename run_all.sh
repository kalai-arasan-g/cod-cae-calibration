#!/usr/bin/env bash
# Full pipeline of the paper. Total CPU time is roughly 8-10 hours on 2 cores
# (about 50-70 s per model; 435 models + 50 reproduction runs). Steps can be run one by one.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

# 1. Data (needs WHO raw files in data/raw/, see data/raw/README.md)
$PY src/00_select_who.py
$PY src/01_prep_data.py
$PY src/02_prep_post2019.py
$PY src/03_prep_post2019_uk_interp.py

# 2. Reproduction of Tanaka & Matsuyama (2025): 10 seeds per population, test 2015-2019
for P in JPN_M JPN_F GBR_M GBR_F DEU_M; do $PY src/cae.py $P 0,1,2,3,4,5,6,7,8,9; done
$PY src/11_analyze_reproduction.py

# 3. Training for the extensions (two parallel streams per job list)
$PY src/make_jobs.py
for J in rolling_K1 latent_K2_K3_K5 lambda_check covid_origin2019; do
  $PY src/train_rolling.py @jobs/${J}_a.txt & $PY src/train_rolling.py @jobs/${J}_b.txt; wait
done

# 4. Rolling-origin accuracy, intervals and calibration
$PY src/21_eval_rolling.py
$PY src/22_summarize_rolling.py
$PY src/23_eval_calibration.py
$PY src/24_eval_calibration_horizon.py
$PY src/25_eval_window_bootstrap.py
$PY src/26_eval_residual_bootstrap.py

# 5. Number of latent dimensions and regularisation check
$PY src/31_eval_latent_dims.py
$PY src/32_eval_lambda.py

# 6. COVID-19 period test (main, and with UK 2021-22 interpolated population)
$PY src/41_eval_covid.py
$PY src/41_eval_covid.py --interp
$PY src/43_covid_balance.py

# 7. Figures of the manuscript
$PY src/51_make_figures.py
