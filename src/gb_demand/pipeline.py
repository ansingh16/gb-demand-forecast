"""Pipeline stages, run by DVC (`dvc repro`) or one at a time:

    python -m gb_demand.pipeline download [--refresh]
    python -m gb_demand.pipeline prepare
    python -m gb_demand.pipeline features
    python -m gb_demand.pipeline backtest
    python -m gb_demand.pipeline train
"""

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd

from . import calibration, metrics, model, prepare, sources
from .config import INTERVALS, MODELS, PROCESSED, QUANTILES, RAW, REPORTS, load_params
from .features import add_weather, build_base, complete_weather_days, solar_capacity_by_day, target_rows


def stage_download(refresh=False):
    sources.download_all(RAW, refresh=refresh)


def stage_prepare():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    prepare.demand(RAW).to_parquet(PROCESSED / "demand.parquet", index=False)
    prepare.benchmark(RAW).to_parquet(PROCESSED / "neso_benchmark.parquet", index=False)
    indo = prepare.indo(pd.read_parquet(RAW / "elexon_indo.parquet"))
    indo.to_parquet(PROCESSED / "indo.parquet", index=False)
    actual, forecast = prepare.weather(RAW)
    actual.to_parquet(PROCESSED / "weather_actual.parquet", index=False)
    forecast.to_parquet(PROCESSED / "weather_forecast.parquet", index=False)


def stage_features():
    demand = pd.read_parquet(PROCESSED / "demand.parquet")
    bench = pd.read_parquet(PROCESSED / "neso_benchmark.parquet")
    indo = pd.read_parquet(PROCESSED / "indo.parquet")
    base = build_base(target_rows(demand, bench), indo, solar_capacity_by_day(demand))
    for name in ["actual", "forecast"]:
        weather = pd.read_parquet(PROCESSED / f"weather_{name}.parquet")
        add_weather(base, weather).to_parquet(PROCESSED / f"features_{name}.parquet", index=False)


def monthly_folds(first, last):
    """(start, end) of each month from `first` to `last`; the first may start mid-month."""
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    starts = [first] + list(pd.date_range(first + pd.offsets.MonthBegin(1), last, freq="MS"))
    return [(s, min(s + pd.offsets.MonthEnd(0), last)) for s in starts]


def backtest(features_actual, features_forecast, params):
    """Refit monthly and forecast with day-ahead weather (notebook 08).

    Training uses ERA5 weather, which is published about five days late, so training stops
    `training_gap_days` before each month. Months before the test window are forecast too,
    only to calibrate the first test months on forecast-weather errors.
    """
    rules = params["backtest"]
    covered = complete_weather_days(features_forecast)
    has_outturn = features_forecast["Demand_Outturn"].notna()
    last_outturn = pd.to_datetime(features_forecast.loc[has_outturn, "Date"]).max()
    folds = monthly_folds(rules["calibration_start"], last_outturn)
    train_dates = pd.to_datetime(features_actual["Date"])
    test_dates = pd.to_datetime(features_forecast["Date"])

    parts = []
    for start, end in folds:
        train = features_actual[(train_dates <= start - pd.Timedelta(days=rules["training_gap_days"]))
                                & features_actual["ND"].notna()]
        rows = features_forecast[test_dates.between(start, end) & has_outturn & covered]
        models = model.fit(train, params["model"])
        out = rows[["start_utc", "Date", "slot", "Demand_Forecast", "Demand_Outturn"]].assign(
            fold=f"{start:%Y-%m}", spring=start < pd.Timestamp(rules["test_start"]))
        parts.append(pd.concat([out, model.predict(models, rows)], axis=1))
        print(f"{start:%Y-%m}: trained on {len(train):,} rows, forecast {len(out):,}", flush=True)
    runs = pd.concat(parts, ignore_index=True)

    run_dates = pd.to_datetime(runs["Date"])
    calibrated = []
    for start, _ in folds:
        if start < pd.Timestamp(rules["test_start"]):
            continue
        history = runs[(run_dates <= start - pd.Timedelta(days=rules["calibration_gap_days"]))
                       & (run_dates > start - pd.DateOffset(months=rules["calibration_months"]))]
        calibrated.append(calibration.reference(runs[runs["fold"] == f"{start:%Y-%m}"], history))
    return pd.concat(calibrated, ignore_index=True)


def summarise(test):
    """Headline test scores, with NESO scored on the same half-hours."""
    y = test["Demand_Outturn"]
    ours, neso = metrics.scores(test["point"], y), metrics.scores(test["Demand_Forecast"], y)
    monthly = test.assign(ours=(test["point"] - y).abs(), neso=(test["Demand_Forecast"] - y).abs())
    monthly = monthly.groupby("fold")[["ours", "neso"]].mean()
    summary = {
        "half_hours": len(test),
        "first_day": test["Date"].min(), "last_day": test["Date"].max(),
        "mae": ours["MAE"], "mape": ours["MAPE"], "rmse": ours["RMSE"], "bias": ours["bias"],
        "neso_mae": neso["MAE"], "neso_mape": neso["MAPE"],
        "months_better_than_neso": int((monthly["ours"] < monthly["neso"]).sum()),
        "months": len(monthly),
    }
    for low, high in INTERVALS:
        summary[f"coverage_{high - low:.0%}"] = metrics.coverage(y, test[f"c{low}"], test[f"c{high}"])
    summary["interval_score_90%"] = metrics.interval_score(y, test["c0.05"], test["c0.95"], 0.9)
    summary["pinball_q0.5"] = metrics.pinball(y, test["q0.5"], 0.5)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in summary.items()}


def log_to_mlflow(params, summary, artifact):
    """Record the run in a local MLflow store (skipped if MLflow is not installed)."""
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed; run not logged")
        return
    mlflow.set_tracking_uri(f"sqlite:///{REPORTS.parent / 'mlflow.db'}")
    mlflow.set_experiment("gb-demand-backtest")
    with mlflow.start_run(run_name=f"backtest {pd.Timestamp.now():%Y-%m-%d %H:%M}"):
        mlflow.log_params({f"model.{k}": v for k, v in params["model"].items()})
        mlflow.log_params({f"backtest.{k}": v for k, v in params["backtest"].items()})
        mlflow.log_metrics({k.replace("%", "pct"): v for k, v in summary.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mlflow.log_artifact(str(artifact))


def stage_backtest():
    params = load_params()
    started = time.time()
    test = backtest(pd.read_parquet(PROCESSED / "features_actual.parquet"),
                    pd.read_parquet(PROCESSED / "features_forecast.parquet"), params)
    test.to_parquet(PROCESSED / "test_forecasts.parquet", index=False)
    summary = summarise(test)
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "metrics.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"backtest took {time.time() - started:.0f} s")
    log_to_mlflow(params, summary, path)


def stage_train(folder=MODELS / "latest"):
    """Fit on everything known and save what the service needs in `folder`."""
    params = load_params()
    features = pd.read_parquet(PROCESSED / "features_actual.parquet")
    train = features[features["ND"].notna()]
    model.save(model.fit(train, params["model"], QUANTILES), folder)

    test = pd.read_parquet(PROCESSED / "test_forecasts.parquet")
    test_dates = pd.to_datetime(test["Date"])
    months = pd.DateOffset(months=params["backtest"]["calibration_months"])
    recent = test[test_dates > test_dates.max() - months]
    calibration.error_quantiles(recent).to_csv(folder / "error_quantiles.csv")

    shutil.copy(RAW / "weather_locations.csv", folder / "weather_locations.csv")
    demand = pd.read_parquet(PROCESSED / "demand.parquet")
    solar_capacity_by_day(demand).tail(90).rename("solar_capacity").to_csv(folder / "solar_capacity.csv")
    meta = {"trained_until": train["Date"].max(), "rows": len(train),
            "intervals_from": recent["Date"].min(), "intervals_to": recent["Date"].max(),
            "trained_at": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%MZ}"}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["download", "prepare", "features", "backtest", "train"])
    parser.add_argument("--refresh", action="store_true", help="download: fetch the latest data")
    parser.add_argument("--out", help="train: folder to save the models in (default models/latest)")
    args = parser.parse_args()
    if args.stage == "download":
        stage_download(args.refresh)
    elif args.stage == "train" and args.out:
        stage_train(MODELS.parent / args.out)
    else:
        globals()[f"stage_{args.stage}"]()


if __name__ == "__main__":
    main()
