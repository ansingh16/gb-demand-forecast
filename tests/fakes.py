"""Fake data sources and tiny models with the real file layout, shared by the tests."""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from gb_demand.config import COLUMNS, QUANTILES


def make_model_folder(path):
    """Tiny models trained on random rows, saved as the `train` stage saves them."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(500, len(COLUMNS))), columns=COLUMNS)
    y = rng.normal(0, 500, 500)
    params = dict(n_estimators=5, num_leaves=4, min_child_samples=5, verbose=-1)
    path.mkdir(parents=True, exist_ok=True)
    lgb.LGBMRegressor(**params).fit(x, y).booster_.save_model(path / "point.txt")
    for q in QUANTILES:
        lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(x, y).booster_.save_model(
            path / f"{q}.txt")
    errors = pd.DataFrame({q: np.full(48, 2000 * (q - 0.5)) for q in QUANTILES}, index=range(48))
    errors.to_csv(path / "error_quantiles.csv")
    pd.Series(15_000.0, index=pd.date_range("2024-12-01", "2025-01-31"), name="solar_capacity").to_csv(
        path / "solar_capacity.csv")
    pd.DataFrame({"city": ["A", "B"], "latitude": [51.5, 53.5], "longitude": [-0.1, -2.2]}).to_csv(
        path / "weather_locations.csv", index=False)
    (path / "meta.json").write_text(json.dumps({"trained_until": "2025-01-20"}))
    return path


def fake_indo(start, end):
    times = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1),
                          freq="30min", inclusive="left")
    return pd.DataFrame({"startTime": times.strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "publishTime": (times + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                         "settlementDate": times.strftime("%Y-%m-%d"), "settlementPeriod": 1,
                         "initialDemandOutturn": 25_000 + 3_000 * np.sin(times.hour / 24 * 2 * np.pi)})


def fake_weather(locations, target):
    times = pd.date_range(pd.Timestamp(target, tz="UTC") - pd.Timedelta(days=1), periods=72, freq="h")
    return pd.concat([pd.DataFrame({"time": times, "city": city, "temperature_2m": 5.0,
                                    "shortwave_radiation": 100.0, "wind_speed_10m": 15.0})
                      for city in locations["city"]], ignore_index=True)


