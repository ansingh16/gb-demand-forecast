"""Daily operations, run by Airflow (`dags/gb_demand.py`) or by hand:

    python -m gb_demand.ops forecast --date 2026-11-19

Forecasts are written to `data/live/` (not in git), one file per target day. Running the same
day again overwrites its file with the same result, so a retried or replayed task is safe.
"""

import argparse

import pandas as pd

from .config import MODELS, ROOT
from .live import Forecaster

LIVE = ROOT / "data" / "live"


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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["forecast"])
    parser.add_argument("--date", required=True, help="target day, YYYY-MM-DD")
    args = parser.parse_args()
    if args.task == "forecast":
        forecast(args.date)


if __name__ == "__main__":
    main()
