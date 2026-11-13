# GB electricity demand forecasting

Half-hourly probabilistic forecasts of Great Britain's national electricity demand,
compared with the day-ahead forecast published by the National Energy System Operator
(NESO).

Work in progress. The notebooks in `notebooks/` are numbered in working order, and each
step explains what is being done and why.

## Results so far

Final test: 1 June 2024 to 9 November 2026, 42,818 half-hours, forecasts made at 08:30 UTC
the day before using day-ahead weather forecasts (notebook 08).

| | NESO day-ahead | This model |
|---|---|---|
| MAE | 689 MW | 880 MW |
| MAPE | 2.8% | 3.6% |
| TRIAD periods (MAE) | 806 MW | 799 MW |
| Bank holidays (MAE) | 695 MW | 1,439 MW |
| 90% interval coverage | no interval published | 88.5% |

NESO's point forecast is clearly more accurate. This model's intervals are well calibrated
on unseen data (50%, 80% and 90% intervals cover 49%, 79% and 88.5% of outcomes).

## Problems caught and how they were fixed

A running log of issues found while building the forecast, how each was caught, and what the
fix was worth where it could be measured. The notebook number is in brackets.

### Data and time alignment

| Problem | How it was caught | Fix | Effect |
|---|---|---|---|
| NESO's yearly demand files use three different date formats (`01-JAN-2018`, `01-Jan-23`, `2025-01-01`) | compared the yearly files before combining them (01) | parse each file with an explicit list of formats and stop if none fits | no silent date errors |
| NESO's benchmark `Datetime` ends in `Z` but is local clock time at the end of the period, and its convention changed on the last three clock-change days | timestamps jumped from 00:30 to 02:00 in spring and repeated 01:00 in autumn (02) | never join on it; build UTC from local midnight plus `(period − 1) × 30` minutes | tested on all 2,047 days, including 46 and 50 period days |
| Eight rows on two autumn clock-change days were mislabelled (periods 2 and 3 missing, 4 and 5 duplicated) | duplicate-key check (02) | dropped them rather than guess | 8 of about 98,000 rows |
| NESO's outturn is not exactly `ND`: about 100 MW lower in winter (GMT), equal in summer | joined the benchmark to the historic data and compared (02) | train on `ND`, score on NESO's outturn, and check the choice | changes NESO's MAE by under 1 MW |
| NESO's own `Absolute_Error` column disagreed with forecast minus outturn on 313 rows | recomputed the errors (02) | found they are all TRIAD periods, scored against the TRIAD-corrected outturn | TRIAD periods reported both ways |
| From mid-June 2026 NESO's `TSD` no longer matched its documented definition | definition check on the data (01) | confirmed against the benchmark that `ND`, the target, is unaffected | no change needed |
| "Same time last week" in UTC is an hour out in the week after each clock change | noticed when building lag features (03, 05) | look up lags by local wall-clock time | lags line up with daily routines |

### Leakage and the information cut-off

| Problem | How it was caught | Fix | Effect |
|---|---|---|---|
| Exploring the whole dataset would let the test period shape the model | planned up front (03) | all exploration stops at 31 May 2024; the test window was run once, at the end | test score is a fair estimate |
| The planned weather source (`_previous_day1`, predicted 24 h before valid time) comes from model runs that start **after** the forecast is made, for most target hours | checked the API definition against the 08:30 UTC origin (04) | use `_previous_day2` (48 h before valid time), always available in time | removed a leak that would have flattered the model |
| Using actual weather in testing would be a leak | planned up front (04) | train on ERA5 reanalysis, test only on archived day-ahead forecasts, and measure the cost | forecast weather costs a simple model 14% and the final model 6% in MAE |
| The plan only allowed demand from two days back, assuming no live data | checked when Elexon's live demand (`INDO`) is published (04) | use `INDO` up to 08:00 UTC on the day before, with a 30-minute margin | rescaling by the latest day cut baseline MAE by 6% |
| Training on revised `ND` lags but feeding early `INDO` values in use would mismatch the inputs | compared `INDO` with `ND` (04) | lag features come from `INDO` in training and testing | same inputs in both |
| ERA5 is published about five days late | checked the archive's end date (04) | in the test run, training data stops seven days before each month | retraining as it could happen live |

### Weather handling

| Problem | How it was caught | Fix | Effect |
|---|---|---|---|
| Open-Meteo radiation is the mean over the hour **before** the timestamp; temperature is the value at it | read the variable definitions (04) | each half-hour takes radiation from the hour that contains it | sunlight lands in the right periods |
| Population weights from GeoNames are for the city proper, which gave London 67% of the weight | looked at the weights (04) | equal-weight mean of ten locations | no single city dominates |
| Open-Meteo refused requests (HTTP 429, per-minute limit) | first full download failed (04) | pause and retry, and cache files locally | the download runs once |

### Modelling

| Problem | How it was caught | Fix | Effect |
|---|---|---|---|
| Trees cannot predict outside their training range, and demand has fallen about 4 GW since 2018 | the direct model scored worse than expected (06) | predict demand minus the latest 24 hours of live demand | validation MAE 851 to 786 MW |
| Correcting the similar-day baseline inherited last week's weather | tried as an alternative framing (06) | kept the latest-level framing | avoided a worse model (967 MW) |
| A lockdown flag added nothing | ablation (06) | dropped it | 7 MW better, one feature fewer |
| Training only on recent years, to handle drift, was worse | ablation (06) | kept all years from 2019 | 786 against 849 MW |
| 71% of rows had quantile forecasts out of order | crossing check (07) | sort each row's quantiles | valid distributions; sorting can only lower pinball loss |
| Quantile models were overconfident: the 90% interval held the outcome only 74% of the time | coverage check on validation (07) | compared conformal calibration with the point forecast plus recent out-of-sample errors, and chose by interval score on validation | 88.5% coverage for the 90% interval on the test window |
| Calibrating on actual-weather errors would understate the errors with forecast weather | followed from notebook 04's results (08) | calibrate each test month only on forecast-weather errors known before it | coverage held on the test window |

### Engineering

| Problem | How it was caught | Fix | Effect |
|---|---|---|---|
| Notebooks ran in another project's environment because a virtual environment was active in the shell | a `pyarrow` import failed (03) | run notebooks with an explicit interpreter | the right environment every run |
| Training was slow: about 25 seconds per LightGBM fit | one validation run took 47 minutes (06) | moved the project to uv with a lock file and set the thread count | about 2 seconds per fit, identical results |
| numpy 2.5 raised deprecation warnings inside pandas 2.3 | warnings in re-run notebooks (06) | pinned numpy below 2.4 in `pyproject.toml` | clean runs |

### Still open

- **Bank holidays and Christmas** are the largest weakness (1,439 MW MAE on bank holidays,
  against NESO's 695 MW). Ideas: features for the days around each holiday, and a holiday
  adjustment learnt from all years.
- **Weather inputs** come from ten equal-weight points and one model; NESO uses more. Ideas:
  locations weighted by installed solar, and more than one weather model.

## Data

- [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data)
- [NESO Day Ahead Half Hourly Demand Forecast Performance](https://www.neso.energy/data-portal/day-ahead-half-hourly-demand-forecast-performance)
- [Elexon Insights API](https://developer.data.elexon.co.uk): initial national demand outturn (`INDO`)
- [Open-Meteo](https://open-meteo.com): ERA5 reanalysis and archived forecasts (Previous Runs API), CC BY 4.0

NESO data is used under the [NESO Open Data Licence](https://www.neso.energy/data-portal/ngeso-open-licence).
Data files are not stored in this repo; the notebooks download them.

## Running

```
uv sync
uv run jupyter lab
```
