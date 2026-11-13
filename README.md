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

The ten most important issues found while building the forecast. The notebook number is in brackets.

| # | Problem | How it was caught | Fix | Effect |
|---|---|---|---|---|
| 1 | The planned weather source (`_previous_day1`, predicted 24 h before valid time) comes from model runs that start **after** the forecast is made, for most target hours | checked the API definition against the 08:30 UTC forecast time (04) | use `_previous_day2` (48 h before valid time), always available in time | removed a leak that would have flattered the model |
| 2 | Using actual weather when testing would be a leak | planned up front (04) | train on ERA5 reanalysis, test only on archived day-ahead forecasts, and measure the cost | forecast weather costs the final model 6% in MAE (14% for a simple model) |
| 3 | NESO's benchmark `Datetime` ends in `Z` but is local clock time at the end of the period, and its convention changed on recent clock-change days | timestamps jumped from 00:30 to 02:00 in spring and repeated 01:00 in autumn (02) | never join on it; build UTC from local midnight plus `(period − 1) × 30` minutes | tested on all 2,047 days, including 46 and 50 period days |
| 4 | Exploring the whole dataset would let the test period shape the model | planned up front (03) | exploration stops at 31 May 2024; the test window was run once, at the end | validation gap to NESO (174 MW) matched the test gap (191 MW) |
| 5 | The plan only allowed demand from two days back, assuming no live data | checked when Elexon's live demand (`INDO`) is published (04) | use `INDO` up to 08:00 UTC the day before, from the same source in training and testing | the latest day of demand cut baseline MAE by 6% |
| 6 | Trees cannot predict outside their training range, and demand has fallen about 4 GW since 2018 | the direct model scored worse than expected (06) | predict demand minus the latest 24 hours of live demand | validation MAE 851 to 786 MW |
| 7 | Quantile models were overconfident: the 90% interval held the outcome only 74% of the time, and 71% of rows had quantiles out of order | coverage and crossing checks on validation (07) | compared conformal calibration with the point forecast plus recent out-of-sample errors, chose by interval score on validation | 88.5% coverage for the 90% interval on the test window |
| 8 | Calibrating on actual-weather errors would understate the errors with forecast weather | followed from notebook 04's results (08) | calibrate each test month only on forecast-weather errors known before it | coverage held on unseen data |
| 9 | Open-Meteo radiation is the mean over the hour **before** the timestamp, while temperature is the value at it | read the variable definitions (04) | each half-hour takes radiation from the hour that contains it | sunlight lands in the right half-hours |
| 10 | Training was slow: about 25 seconds per LightGBM fit | one validation run took 47 minutes (06) | moved the project to uv with a lock file and set the thread count | about 2 seconds per fit, identical results |

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
