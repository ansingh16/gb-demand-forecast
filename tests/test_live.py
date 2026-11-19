import pandas as pd
import pytest
from fakes import fake_indo, fake_weather
from fastapi.testclient import TestClient

from gb_demand import api
from gb_demand.cutoff import latest_demand_allowed
from gb_demand.live import Forecaster, NotReady, check_target

NOW = pd.Timestamp("2025-02-02 12:00", tz="UTC")


def test_targets_open_at_the_origin_and_past_days_can_be_replayed():
    assert check_target("2025-02-03", NOW) == pd.Timestamp("2025-02-03")
    assert check_target("2025-01-25", NOW) == pd.Timestamp("2025-01-25")   # replay
    with pytest.raises(NotReady):
        check_target("2025-02-04", NOW)             # origin is 08:30 on 3 February


def test_replay_refuses_days_the_model_was_trained_on():
    with pytest.raises(NotReady):
        check_target("2025-01-20", NOW, trained_until="2025-01-20")
    assert check_target("2025-01-21", NOW, trained_until="2025-01-20") == pd.Timestamp("2025-01-21")


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
    monkeypatch.setattr("gb_demand.sources.fetch_day2_forecast", fake_weather)
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

