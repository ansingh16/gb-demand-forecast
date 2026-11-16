"""Raw downloads to clean tables, each with a UTC period start (notebooks 01, 02 and 04)."""

import pandas as pd

from .config import VARIABLES
from .periods import parse_settlement_dates, period_start_utc
from .weather import to_half_hours

KEY = ["Date", "Settlement_Period"]
BENCHMARK_COLUMNS = ["start_utc", "Date", "Settlement_Period", "Demand_Forecast", "Demand_Outturn",
                     "TRIAD_Avoidance_Estimate", "TRIAD_Avoidance_Corrected_Demand_Outturn",
                     "Publish_Datetime"]


def demand(raw):
    """NESO historic demand. Each yearly file has its own date format (notebook 01)."""
    frames = []
    for path in sorted(raw.glob("demanddata_*.csv")):
        frame = pd.read_csv(path, low_memory=False)
        frame["SETTLEMENT_DATE"] = parse_settlement_dates(frame["SETTLEMENT_DATE"])
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["Date"] = out["SETTLEMENT_DATE"].dt.strftime("%Y-%m-%d")
    out = out.rename(columns={"SETTLEMENT_PERIOD": "Settlement_Period"})
    out["start_utc"] = period_start_utc(out["Date"], out["Settlement_Period"])
    return out.drop(columns=["SETTLEMENT_DATE"]).sort_values("start_utc").reset_index(drop=True)


def benchmark(raw):
    """NESO's day-ahead forecast and outturn (notebook 02).

    The `Datetime` column ends in `Z` but is local clock time at the end of each period, and
    its convention changed on recent clock-change days, so it is never used. Rows sharing a
    date and period are dropped, as are incomplete days.
    """
    bench = pd.read_csv(raw / "neso_day_ahead_forecast.csv", encoding="utf-8-sig")
    bench = bench[~bench.duplicated(KEY, keep=False)].copy()
    per_day = bench.groupby("Date")["Settlement_Period"]
    complete = (per_day.transform("count") == per_day.transform("max")) & (per_day.transform("max") >= 46)
    bench = bench[complete].reset_index(drop=True)
    bench["start_utc"] = period_start_utc(bench["Date"], bench["Settlement_Period"])
    return bench[BENCHMARK_COLUMNS]


def indo(raw_indo):
    """Elexon live demand: published at the end of each period (notebook 04)."""
    out = raw_indo.copy()
    out["start_utc"] = pd.to_datetime(out["startTime"], utc=True)
    out["publish_utc"] = pd.to_datetime(out["publishTime"], utc=True)
    out = out.rename(columns={"settlementDate": "Date", "settlementPeriod": "Settlement_Period",
                              "initialDemandOutturn": "INDO"})
    out = out.drop_duplicates("start_utc", keep="last")
    return out[["start_utc", "Date", "Settlement_Period", "INDO", "publish_utc"]].reset_index(drop=True)


def weather(raw):
    """Half-hourly actual (ERA5) and forecast (`_previous_day2`) weather."""
    era5 = pd.read_parquet(raw / "weather_era5.parquet").dropna(subset=VARIABLES)
    forecast = pd.read_parquet(raw / "weather_forecast_day2.parquet").dropna(subset=VARIABLES)
    return to_half_hours(era5), to_half_hours(forecast)
