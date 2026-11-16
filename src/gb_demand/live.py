"""A live forecast for one day, from the saved models and today's public data.

The same rules as the backtest apply: the forecast for day D can be made from 08:30 UTC on
D-1, and only uses live demand (`INDO`) up to the period ending 08:00 UTC that day. Weather
is the latest Open-Meteo forecast at the time of the request.
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
    """The requested day cannot be forecast now (too early, or already over)."""


def check_target(target_date, now):
    """Allow a day whose forecast origin has passed and which has not ended yet."""
    target = pd.Timestamp(target_date).normalize()
    origin = forecast_origin(target)
    if now < origin:
        raise NotReady(f"the forecast for {target:%Y-%m-%d} can be made from {origin:%Y-%m-%d %H:%M} UTC")
    day_end = (target + pd.Timedelta(days=1)).tz_localize("Europe/London")
    if now >= day_end:
        raise NotReady(f"{target:%Y-%m-%d} is over; this service only makes forecasts")
    return target


def live_features(target, locations, capacity, fetch_indo=None, fetch_weather=None):
    """Feature rows for one local day, using only demand known at the origin."""
    fetch_indo = fetch_indo or sources.fetch_indo
    fetch_weather = fetch_weather or sources.fetch_live_forecast
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
        target = check_target(target_date, now)
        rows = live_features(target, self.locations, self.capacity, **fetchers)
        out = rows[["start_utc", "slot"]].join(model.predict(self.models, rows))
        out = calibration.apply_intervals(out, self.intervals)
        out["local_time"] = out["start_utc"].dt.tz_convert("Europe/London").dt.strftime("%H:%M")
        return out
