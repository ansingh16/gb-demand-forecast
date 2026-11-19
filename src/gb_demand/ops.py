"""Daily and monthly operations, run by Airflow (`dags/gb_demand.py`) or by hand:

    python -m gb_demand.ops forecast --date 2026-11-19   make and save one day's forecast
    python -m gb_demand.ops score --date 2026-11-18      score it once outturn is published
    python -m gb_demand.ops check                        rolling coverage and accuracy check
    python -m gb_demand.ops gate                         is a retrained model at least as good?
    python -m gb_demand.ops promote                      put the retrained model into service

Forecasts and scores are written to `data/live/` (not in git). Running a task again for the
same day overwrites its result with the same values, so retries and replays are safe.

Exit codes, for Airflow: 0 done, 1 failed or not ready yet (retried), 99 nothing to do (skipped).
"""

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

from . import metrics, model, prepare, sources
from .config import COLUMNS, INTERVALS, MODELS, PROCESSED, REPORTS, ROOT, load_params
from .live import Forecaster, NotReady
from .monitor import COVERAGE_BAND

LIVE = ROOT / "data" / "live"
SKIP = 99


class Skip(Exception):
    """Nothing to do this time; not a failure."""


def forecast(date, folder=MODELS / "latest", live=LIVE, **fetchers):
    """Forecast one target day and save it with when and how it was made."""
    forecaster = Forecaster(folder)
    out = forecaster.forecast(date, **fetchers)
    out = out.assign(target_date=f"{pd.Timestamp(date):%Y-%m-%d}",
                     made_at_utc=pd.Timestamp.now(tz="UTC"),
                     model_trained_until=forecaster.meta["trained_until"])
    path = live / "forecasts" / f"{pd.Timestamp(date):%Y-%m-%d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    peak = out.loc[out["point"].idxmax()]
    print(f"{path.name}: {len(out)} periods, peak {peak['point']:,.0f} MW at {peak['local_time']}")
    return out


def neso_forecast(day, benchmark=PROCESSED / "neso_benchmark.parquet"):
    """NESO's day-ahead forecast for a day, if the benchmark file already has it.

    Elexon's `NDF` dataset looks like a daily copy but is published from about 11:45 UTC the
    day before, after NESO's benchmark forecast, and differs from it, so it is not used.
    """
    if not benchmark.exists():
        return None
    bench = pd.read_parquet(benchmark, columns=["start_utc", "Date", "Demand_Forecast"])
    rows = bench[bench["Date"] == day]
    return rows.set_index("start_utc")["Demand_Forecast"] if len(rows) else None


def score(date, live=LIVE, fetch_indo=None, benchmark=PROCESSED / "neso_benchmark.parquet"):
    """Score a stored forecast against Elexon's outturn (`INDO`) once the whole day is published.

    `INDO` runs about 36 MW below NESO's `ND` (notebook 04); it is used because `ND` is only
    published weeks later. NESO's forecast is scored on the same outturn when available.
    """
    day = f"{pd.Timestamp(date):%Y-%m-%d}"
    path = live / "forecasts" / f"{day}.parquet"
    if not path.exists():
        raise NotReady(f"no stored forecast for {day}")
    frame = pd.read_parquet(path)
    outturn = prepare.indo((fetch_indo or sources.fetch_indo)(day, day))
    outturn = outturn[outturn["Date"] == day]
    if len(outturn) < len(frame):
        raise NotReady(f"outturn for {day} is incomplete ({len(outturn)} of {len(frame)} periods)")
    frame = frame.merge(outturn[["start_utc", "INDO"]], on="start_utc")
    y = frame["INDO"]

    row = {"date": day, "periods": len(frame), "model_trained_until": frame["model_trained_until"].iloc[0],
           "mae": np.abs(frame["point"] - y).mean(), "bias": (frame["point"] - y).mean(),
           "interval_score_90": metrics.interval_score(y, frame["c0.05"], frame["c0.95"], 0.9)}
    for low, high in INTERVALS:
        row[f"inside_{high - low:.0%}"] = metrics.coverage(y, frame[f"c{low}"], frame[f"c{high}"])
    neso = neso_forecast(day, benchmark)
    row["neso_mae"] = np.abs(frame["start_utc"].map(neso) - y).mean() if neso is not None else np.nan

    scores_path = live / "scores.parquet"
    scores = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame()
    if len(scores):
        scores = scores[scores["date"] != day]
    scores = pd.concat([scores, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    scores.to_parquet(scores_path, index=False)
    print(f"{day}: MAE {row['mae']:,.0f} MW, bias {row['bias']:+,.0f} MW, "
          f"inside 90% interval {row['inside_90%']:.0%}")
    return row


def check(live=LIVE, window=30, min_days=20, reference_mae=None):
    """Alert (raise) if recent coverage or accuracy has moved away from the backtest.

    Coverage is pooled over the latest `window` scored days and must sit in the 85% to 95%
    band; MAE must be under 1.5 times the backtest's. With fewer than `min_days` days there is
    not enough to judge, so the check is skipped.
    """
    scores_path = live / "scores.parquet"
    scores = pd.read_parquet(scores_path).tail(window) if scores_path.exists() else pd.DataFrame()
    if len(scores) < min_days:
        raise Skip(f"{len(scores)} scored days, need {min_days} before judging")
    if reference_mae is None:
        reference_mae = json.loads((REPORTS / "metrics.json").read_text())["mae"]
    coverage = np.average(scores["inside_90%"], weights=scores["periods"])
    mae = np.average(scores["mae"], weights=scores["periods"])
    problems = []
    low, high = COVERAGE_BAND
    if not low <= coverage <= high:
        problems.append(f"90% interval held {coverage:.1%} of outcomes over {len(scores)} days")
    if mae > 1.5 * reference_mae:
        problems.append(f"MAE {mae:,.0f} MW is over 1.5 times the backtest's {reference_mae:,.0f} MW")
    print(f"last {len(scores)} days: MAE {mae:,.0f} MW, 90% coverage {coverage:.1%}")
    if problems:
        raise RuntimeError("; ".join(problems))


def gate(holdout_days=14, tolerance=0.05, folder=MODELS / "latest", report=MODELS / "gate.json",
         processed=PROCESSED):
    """Compare a retrained model with the one in service on recent days neither has seen.

    The candidate is fitted the way the backtest fitted each month (data up to seven days
    before the held-out days, ERA5 weather) and both are scored on the latest `holdout_days`
    with forecast weather. Its job is to stop a broken refresh or code change from reaching
    the service, not to prove an improvement: the candidate passes if its MAE is within
    `tolerance` of the current model's.
    """
    params = load_params()
    actual = pd.read_parquet(processed / "features_actual.parquet")
    forecast_rows = pd.read_parquet(processed / "features_forecast.parquet")
    known = forecast_rows[forecast_rows["ND"].notna()]
    # `christmas` is empty outside 20 December to 7 January by design, so it is not required
    needed = [c for c in COLUMNS if c != "christmas"]
    complete = known.groupby("Date")[needed].apply(lambda g: g.notna().all().all())
    days = sorted(complete[complete].index)[-holdout_days:]
    holdout = known[known["Date"].isin(days)]
    start = pd.Timestamp(days[0])

    current = Forecaster(folder)
    trained_until = pd.Timestamp(current.meta["trained_until"])
    if trained_until >= start:
        raise Skip(f"the model in service was trained up to {trained_until:%Y-%m-%d}, "
                   f"inside the held-out days from {days[0]}; nothing new to compare on")

    dates = pd.to_datetime(actual["Date"])
    train = actual[(dates <= start - pd.Timedelta(days=7)) & actual["ND"].notna()]
    candidate = model.fit(train, params["model"], quantiles=[])
    y = holdout["ND"]
    current_point = model.predict({"point": current.models["point"]}, holdout)["point"]
    result = {
        "held_out": f"{days[0]} to {days[-1]}",
        "current_trained_until": f"{trained_until:%Y-%m-%d}",
        "candidate_trained_until": train["Date"].max(),
        "current_mae": float(np.abs(current_point - y).mean()),
        "candidate_mae": float(np.abs(model.predict(candidate, holdout)["point"] - y).mean()),
    }
    result["passed"] = result["candidate_mae"] <= result["current_mae"] * (1 + tolerance)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise RuntimeError("candidate is worse than the model in service; not promoted")
    return result


def promote(models=MODELS):
    """Swap the staged model in, keeping the old one as `previous` for a rollback."""
    staging, latest, previous = models / "staging", models / "latest", models / "previous"
    if not (staging / "meta.json").exists():
        raise NotReady("no staged model to promote")
    if previous.exists():
        shutil.rmtree(previous)
    if latest.exists():
        latest.rename(previous)
    staging.rename(latest)
    print("promoted:", json.loads((latest / "meta.json").read_text()))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["forecast", "score", "check", "gate", "promote"])
    parser.add_argument("--date", help="target day, YYYY-MM-DD (forecast, score)")
    args = parser.parse_args()
    if args.task in ("forecast", "score") and not args.date:
        parser.error(f"{args.task} needs --date")
    try:
        if args.task == "forecast":
            forecast(args.date)
        elif args.task == "score":
            score(args.date)
        else:
            {"check": check, "gate": gate, "promote": promote}[args.task]()
    except Skip as reason:
        print("skipped:", reason)
        sys.exit(SKIP)
    except (NotReady, RuntimeError) as error:
        print("error:", error)
        sys.exit(1)


if __name__ == "__main__":
    main()
