import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from gb_demand import api
from gb_demand.config import COLUMNS, QUANTILES
from gb_demand.cutoff import latest_demand_allowed
from gb_demand.live import Forecaster, NotReady, check_target

NOW = pd.Timestamp("2025-02-02 12:00", tz="UTC")


@pytest.fixture
def model_folder(tmp_path):
    """Tiny models with the real file layout, trained on random rows."""
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(size=(500, len(COLUMNS))), columns=COLUMNS)
    y = rng.normal(0, 500, 500)
    params = dict(n_estimators=5, num_leaves=4, min_child_samples=5, verbose=-1)
    lgb.LGBMRegressor(**params).fit(x, y).booster_.save_model(tmp_path / "point.txt")
    for q in QUANTILES:
        lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(x, y).booster_.save_model(
            tmp_path / f"{q}.txt")
    errors = pd.DataFrame({q: np.full(48, 2000 * (q - 0.5)) for q in QUANTILES}, index=range(48))
    errors.to_csv(tmp_path / "error_quantiles.csv")
    pd.Series(15_000.0, index=pd.date_range("2024-12-01", "2025-01-31"), name="solar_capacity").to_csv(
        tmp_path / "solar_capacity.csv")
    pd.DataFrame({"city": ["A", "B"], "latitude": [51.5, 53.5], "longitude": [-0.1, -2.2]}).to_csv(
        tmp_path / "weather_locations.csv", index=False)
    (tmp_path / "meta.json").write_text(json.dumps({"trained_until": "2025-01-20"}))
    return tmp_path


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


def test_targets_open_at_the_origin_and_close_at_the_end_of_the_day():
    assert check_target("2025-02-03", NOW) == pd.Timestamp("2025-02-03")
    with pytest.raises(NotReady):
        check_target("2025-02-04", NOW)             # origin is 08:30 on 3 February
    with pytest.raises(NotReady):
        check_target("2025-02-01", NOW)             # already over


def test_live_forecast_is_complete_and_ordered(model_folder):
    seen = {}

    def recording_indo(start, end):
        frame = fake_indo(start, end)
        seen["latest"] = pd.to_datetime(frame["startTime"], utc=True).max()
        return frame

    out = Forecaster(model_folder).forecast("2025-02-03", now=NOW, fetch_indo=recording_indo,
                                            fetch_weather=fake_weather)
    assert len(out) == 48
    assert seen["latest"] > latest_demand_allowed("2025-02-03")  # the fetch returns too much ...
    assert out[["point", "c0.05", "c0.95"]].notna().all().all()   # ... and the forecast still works
    assert (out["c0.05"] < out["c0.25"]).all() and (out["c0.25"] < out["c0.75"]).all()
    assert (out["c0.75"] < out["c0.95"]).all()


def test_api_returns_periods_and_refuses_early_requests(model_folder, monkeypatch):
    monkeypatch.setenv("GB_DEMAND_MODELS", str(model_folder))
    monkeypatch.setattr(api, "_forecaster", None)
    monkeypatch.setattr("gb_demand.sources.fetch_indo", fake_indo)
    monkeypatch.setattr("gb_demand.sources.fetch_live_forecast", fake_weather)
    client = TestClient(api.app)

    assert client.get("/health").json()["model"]["trained_until"] == "2025-01-20"
    today = pd.Timestamp.now(tz="Europe/London").normalize()
    ok = client.get("/forecast", params={"date": f"{today:%Y-%m-%d}"})
    assert ok.status_code == 200
    body = ok.json()
    assert len(body["periods"]) in (46, 48, 50)
    assert {"point", "p05", "p95"} <= set(body["periods"][0])

    too_early = client.get("/forecast", params={"date": f"{today + pd.Timedelta(days=5):%Y-%m-%d}"})
    assert too_early.status_code == 409
    assert client.get("/forecast", params={"date": "tomorrow"}).status_code == 422
