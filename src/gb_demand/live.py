"""A forecast for one day, from the saved models and public data.

The same rules as the backtest apply: the forecast for day D is made at 08:30 UTC on D-1,
uses live demand (`INDO`) only up to the period ending 08:00 UTC that day, and uses weather
forecasts made 48 hours before each hour (`_previous_day2`). Because every input is fixed at
the origin, a past day can be replayed and gives the forecast that would have been made then.
This is what lets the scheduler fill in days it missed.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import calibration, model, prepare, sources
from .cutoff import forecast_origin, latest_demand_allowed
from .features import add_weather, bank_holidays, build_base, day_rows
from .weather import to_half_hours

DEMAND_HISTORY_DAYS = 16  # lag14, plus the similar day before a bank holiday


class NotReady(ValueError):
    """The requested day cannot be forecast honestly (too early, or seen in training)."""


def check_target(target_date, now, trained_until=None):
    """Allow a day once its origin has passed, unless the model was trained on it."""
    target = pd.Timestamp(target_date).normalize()
    origin = forecast_origin(target)
    if now < origin:
        raise NotReady(f"the forecast for {target:%Y-%m-%d} can be made from {origin:%Y-%m-%d %H:%M} UTC")
    if trained_until is not None and target <= pd.Timestamp(trained_until):
        raise NotReady(f"the model was trained on data up to {trained_until}, so {target:%Y-%m-%d} "
                       "cannot be replayed without a leak")
    return target


def live_features(target, locations, capacity, fetch_indo=None, fetch_weather=None):
    """Feature rows for one local day, using only data known at the origin."""
    fetch_indo = fetch_indo or sources.fetch_indo
    fetch_weather = fetch_weather or sources.fetch_day2_forecast
    raw = fetch_indo(target - pd.Timedelta(days=DEMAND_HISTORY_DAYS), target - pd.Timedelta(days=1))
    indo = prepare.indo(raw)
    indo = indo[indo["start_utc"] <= latest_demand_allowed(target)]
    base = build_base(day_rows(target), indo, capacity, bank_holidays(target.year + 1))
    weather = to_half_hours(fetch_weather(locations, target))
    return add_weather(base, weather)


@dataclass
class Forecaster:
    """Saved models, interval table and inputs, as written by the `train` stage."""

    folder: Path
    models: dict = field(init=False)
    intervals: pd.DataFrame = field(init=False)
    capacity: pd.Series = field(init=False)
    locations: pd.DataFrame = field(init=False)
    meta: dict = field(init=False)

    def __post_init__(self):
        self.models = model.load(self.folder)
        self.intervals = pd.read_csv(self.folder / "error_quantiles.csv", index_col=0)
        self.intervals.columns = [float(c) for c in self.intervals.columns]
        capacity = pd.read_csv(self.folder / "solar_capacity.csv", index_col=0, parse_dates=True)
        self.capacity = capacity["solar_capacity"]
        self.locations = pd.read_csv(self.folder / "weather_locations.csv")
        self.meta = json.loads((self.folder / "meta.json").read_text())

    def forecast(self, target_date, now=None, **fetchers):
        now = now or pd.Timestamp.now(tz="UTC")
        target = check_target(target_date, now, self.meta.get("trained_until"))
        rows = live_features(target, self.locations, self.capacity, **fetchers)
        if rows["shortwave_radiation"].isna().any() or rows["level24"].isna().any():
            raise NotReady(f"inputs for {target:%Y-%m-%d} are incomplete (weather or live demand missing)")
        out = rows[["start_utc", "slot"]].join(model.predict(self.models, rows))
        out = calibration.apply_intervals(out, self.intervals)
        out["local_time"] = out["start_utc"].dt.tz_convert("Europe/London").dt.strftime("%H:%M")
        return out
