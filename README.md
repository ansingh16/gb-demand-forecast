# GB electricity demand forecasting

Half-hourly probabilistic forecasts of Great Britain's national electricity demand,
compared with the day-ahead forecast published by the National Energy System Operator
(NESO).

The forecast for each day is made at 08:30 UTC the day before, using only data public at that
moment, and comes with 50%, 80% and 90% intervals. It was tested once, on June 2024 to November
2026, data it had never seen. NESO's forecast is more accurate; this one's intervals hold their
stated coverage. The notebooks in `notebooks/` show the work step by step, with notes on what
was done and why; the code they settled on lives in `src/gb_demand/`, with tests, a DVC
pipeline, a FastAPI service and monitoring.

## Approach

- **Target:** National Demand (`ND`), half-hourly, in MW. Settlement periods are turned into UTC
  from local midnight, so 46 and 50 period days need no special case.
- **What is known at 08:30 UTC the day before:** live demand from Elexon up to 08:00 UTC, the
  calendar, NESO's solar capacity with a four-week delay, and weather forecasts made 48 hours
  before each hour (Open-Meteo archive).
- **Model:** LightGBM with 17 features, predicting demand minus the latest 24 hours of live
  demand, so a falling level does not push the trees outside their training range.
- **Intervals:** the point forecast plus quantiles of its recent errors for each half-hour of the
  day, chosen over quantile regression and conformalized quantile regression on validation data.
- **Validation and test:** 24 rolling monthly folds (June 2022 to May 2024) for every choice;
  then one run on June 2024 to November 2026, refitted monthly, with forecast weather.

## Results

Final test: 1 June 2024 to 9 November 2026, 42,818 half-hours, forecasts made at 08:30 UTC
the day before using day-ahead weather forecasts (notebook 08).

| | NESO day-ahead | This model |
|---|---|---|
| MAE | 689 MW | 880 MW |
| MAPE | 2.8% | 3.6% |
| Months with the lower MAE | 30 of 30 | 0 of 30 |
| TRIAD periods (MAE) | 806 MW | 799 MW |
| Bank holidays (MAE) | 695 MW | 1,439 MW |
| 90% interval coverage | no interval published | 88.5% |

NESO's point forecast is clearly more accurate (this model's MAE is 28% higher), as expected: NESO has live
metering, its own solar and wind estimates and several weather providers. This model's intervals
are well calibrated on unseen data (50%, 80% and 90% intervals cover 49%, 79% and 88.5% of
outcomes), which NESO's published forecast does not offer. The gap to NESO on the test (191 MW)
matched the validation estimate (174 MW), so the validation setup did not flatter the model.

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
| 7 | Quantile models were overconfident: the 90% interval held the outcome only 74% of the time, and 71% of rows had quantiles out of order | coverage and crossing checks on validation (07) | compared conformal calibration with the point forecast plus recent out-of-sample errors, chose by interval score on validation, then calibrated each test month only on forecast-weather errors known before it | 88.5% coverage for the 90% interval on the test window |
| 8 | Filling in missed days with today's model would leak: it was trained on demand up to a recent date and could "forecast" days it had learnt from | designing catch-up for the Airflow DAG (Airflow) | every input is fixed at the day's 08:30 UTC cut-off (archived 48-hour weather, live demand to 08:00), and replays of days on or before the model's last training day are refused | missed days get the forecast that would have been made on time; a replay of 10 October was refused by a model trained to 19 October |
| 9 | A first drift check flagged almost every input in every month | ran it over the 30 test months (09) | compare each month with the same calendar month in past years; report but do not alert on inputs that follow known trends (demand level, solar capacity); rerun the weather check on ERA5 to separate the forecast feed from real weather | temperature flags fell from 30 months to 12, and 90 of 109 drift flags were trends; half the temperature drift turned out to be the forecast feed differing from the ERA5 training data |
| 10 | Code copied out of notebooks can quietly differ from what was tested, and a cut-off rule is easy to break later | ported everything to a package, rebuilt every table and reran the test window through it (Day 9) | compared each table and forecast with the notebook versions; added a test that changes every demand value published after the forecast time and checks that no feature moves | identical tables and test forecasts (largest difference 0 MW over 42,818 half-hours); the leak test runs in CI on every push |

## Limits

- **Bank holidays and Christmas** are the largest weakness (1,439 MW MAE on bank holidays,
  against NESO's 695 MW); most of the worst days are holidays. Ideas: features for the days
  around each holiday, and a holiday adjustment learnt from all years.
- **Weather inputs** come from ten equal-weight points and one weather model, and the forecasts
  are 48 hours old, older than NESO could use. Ideas: locations weighted by installed solar, more
  than one weather model.
- **Trained on ERA5, fed forecasts:** archived forecasts only start in 2024, so the model learns
  from reanalysis. Forecast weather costs it 6% in MAE, and monitoring found the two differ in
  distribution. Training on archived forecasts becomes possible as more years build up.
- **Intervals react slowly:** right on average, but monthly coverage fell to 78% in the worst
  months, because they are set from the previous six months.
- **Solar capacity keeps growing** past what each refit has seen (by about 1% on average), and
  trees cannot extrapolate.
- **Runs on a laptop:** the service (tested in Docker Desktop) and the Airflow scheduler run
  locally, not on a server, so forecasts are made on days the laptop is on and missed days are
  filled in later. The live record started in November 2026 and is still short.
- **Live scores use Elexon's early outturn** (`INDO`, about 36 MW below `ND`), and are compared
  with NESO only after the monthly refresh brings in NESO's benchmark file.

## Notebooks

| | Notebook | What it settles |
|---|---|---|
| 01 | `01_data_and_definitions` | NESO's historic data, what `ND` and `TSD` mean, clock-change days |
| 02 | `02_benchmark_alignment` | NESO's benchmark, its timestamps, how forecasts are scored |
| 03 | `03_eda` | trend, seasons, weekdays, holidays, solar, lags (data before June 2024 only) |
| 04 | `04_weather_and_cutoff` | the 08:30 UTC cut-off, live demand, weather sources, the leak in `_previous_day1` |
| 05 | `05_baselines` | rolling validation folds and simple baselines |
| 06 | `06_lightgbm_point` | the LightGBM point forecast and what each choice is worth |
| 07 | `07_quantiles` | quantile forecasts, pinball loss and coverage |
| 08 | `08_calibration_and_test` | calibration, then the one run on the test window |
| 09 | `09_monitoring` | drift, accuracy and coverage month by month |

## Live forecasts

The `train` stage fits the models on everything known and saves them with the interval table.
The service then builds the next day's features from public data: Elexon demand up to 08:00
UTC on the day before and Open-Meteo forecasts made 48 hours ahead, the same inputs and cut-off
as in testing. Because every input is fixed at that moment, a past day can be replayed and gives
the forecast that would have been made then; days the model was trained on are refused.

```
uv run dvc repro                     # or: uv run python -m gb_demand.pipeline train
docker build -t gb-demand .
docker run -p 8000:8000 gb-demand
curl "localhost:8000/forecast?date=2026-11-17"
```

A day can be forecast from 08:30 UTC on the day before until it ends; other dates get HTTP 409.
Example, made at 14:47 UTC on 16 November 2026 for the next day: evening peak 41,118 MW at 17:30,
90% interval 39,806 to 43,141 MW. The image was built and run with Docker Desktop.

## Scheduling with Airflow

Three DAGs in `dags/gb_demand.py` run the forecast as a daily service. Each task runs a command
from `gb_demand.ops` or `gb_demand.pipeline` in the project's own environment, while Airflow
has its own environment in `orchestration/`, so the two never share dependencies.

| DAG | When (UTC) | What it does |
|---|---|---|
| `daily_forecast` | 08:35 every day | forecasts the next day and saves it to `data/live/forecasts/` |
| `daily_score` | 02:00 every day | scores yesterday's forecast against Elexon outturn once all periods are published (retries hourly until then), then checks the last 30 days: the 90% interval must hold 85% to 95% of outcomes and MAE must stay under 1.5 times the backtest's |
| `monthly_retrain` | 03:00 on the 2nd | refreshes the data, reruns the backtest (`dvc repro`), and promotes a retrained model only if it passes a gate; the old model is kept for a rollback |

**Missed days are filled in honestly.** The forecast task only uses the run's date and data
fixed at that day's 08:30 UTC cut-off, including weather forecasts already archived 48 hours
ahead. So runs are safe to repeat, and when the scheduler starts after a gap (catchup) it makes
the forecasts that would have been made on time. Replays of days the model was trained on are
refused.

**The gate** fits a candidate the way the backtest did (data up to seven days before the latest
14 days) and scores it and the model in service on those 14 days, which neither has seen, with
forecast weather. Its job is to stop a broken refresh or code change from reaching the service,
so the candidate passes if it is no more than 5% worse. In a test on real data, with a
month-old model in service, the candidate scored 970 MW against 938 MW on 16 to 29 October 2026
(3.4% worse, within the noise of two weeks), passed, and was promoted. If the model in service
has already seen the latest 14 days, the gate is skipped and nothing is promoted.

**What the end-to-end runs caught:** a full monthly run on real data, rather than the unit
tests, found four faults, all fixed: the refresh died on an Open-Meteo timeout because only rate
limits were retried (it now retries timeouts, and fetches only new days instead of every year
since 2018); the DVC task failed because the PATH under WSL contains folders with spaces; the
gate held out Christmas 2025 instead of the latest days, because `christmas` is empty outside
the holidays by design and was treated as missing; and a test run dated before the DAG's start
date "succeeded" in 0.07 seconds without running a task.

**Scoring against NESO day to day is not possible yet.** Elexon's `NDF` dataset looks like a
daily copy of NESO's day-ahead forecast, but it is first published for a day at about 11:45 UTC
the day before, after NESO's benchmark forecast, and differs from it (MAE 267 MW against the
benchmark on 5 November 2026). So live scores compare with NESO only once the monthly refresh
brings in NESO's benchmark file. Live forecasts are scored on `INDO`, which runs about 36 MW
below `ND`.

```
cd orchestration && uv sync && cd ..
orchestration/airflow.sh db migrate
orchestration/airflow.sh standalone                            # scheduler and UI on localhost:8080
orchestration/airflow.sh dags test daily_forecast 2026-11-18   # one run, no scheduler
```

## Monitoring

`python -m gb_demand.monitor` writes [`reports/monitoring.md`](reports/monitoring.md):
population stability index (PSI) of each input against the same calendar month in 2019 to 2023,
monthly MAE against NESO, and 90% interval coverage. Over the test window:

- The intervals are right on average (88.5%) but slow to react: monthly coverage ranged from 78%
  to 97%, and fell in exactly the months where errors jumped (correlation −0.89).
- Half of the temperature drift came from the forecast feed differing from the ERA5 data the
  model was trained on, not from unusual weather.

## Data

- [NESO Historic Demand Data](https://www.neso.energy/data-portal/historic-demand-data)
- [NESO Day Ahead Half Hourly Demand Forecast Performance](https://www.neso.energy/data-portal/day-ahead-half-hourly-demand-forecast-performance)
- [Elexon Insights API](https://developer.data.elexon.co.uk): initial national demand outturn (`INDO`)
- [Open-Meteo](https://open-meteo.com): ERA5 reanalysis and archived forecasts (Previous Runs API), CC BY 4.0

NESO data is used under the [NESO Open Data Licence](https://www.neso.energy/data-portal/ngeso-open-licence).
Data files are not stored in this repo; the pipeline's `download` stage (or the notebooks)
fetches them. Open-Meteo's free API is for non-commercial use.

## Code and pipeline

| Path | What it does |
|---|---|
| `src/gb_demand/periods.py`, `cutoff.py` | settlement periods to UTC; what is known at the 08:30 UTC forecast time |
| `src/gb_demand/sources.py`, `prepare.py` | downloads from NESO, Elexon and Open-Meteo, and cleaning |
| `src/gb_demand/features.py`, `model.py`, `calibration.py` | features, LightGBM point and quantile models, calibrated intervals |
| `src/gb_demand/pipeline.py` | the stages below; the backtest is logged to MLflow |
| `src/gb_demand/live.py`, `api.py`, `Dockerfile` | forecast for any day once its cut-off has passed, served with FastAPI |
| `src/gb_demand/ops.py`, `dags/`, `orchestration/` | forecast, score, check, gate and promote tasks, and the Airflow DAGs that schedule them |
| `src/gb_demand/monitor.py`, `reports/monitoring.md` | input drift, accuracy and interval coverage by month (notebook 09) |
| `dvc.yaml`, `params.yaml` | stages `download → prepare → features → backtest → train` and their settings |
| `reports/metrics.json` | test-window scores from the latest backtest |
| `tests/` | period handling, cut-off rules (including a leak test), weather alignment, intervals, the API, replay, scoring, the gate, download retries, monitoring; `orchestration/tests/` checks the DAGs |

## Running

```
uv sync                     # Python 3.12 environment from uv.lock
uv run dvc repro            # download data, build features, backtest (about 25 minutes), train
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db   # backtest runs
uv run pytest               # tests
uv run jupyter lab          # notebooks
```
