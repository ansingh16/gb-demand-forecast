import pandas as pd
import pytest
import requests

from gb_demand import sources
from gb_demand.config import FORECAST_SUFFIX, VARIABLES


class Response:
    def __init__(self, status=200, payload=None):
        self.status_code, self.payload = status, payload or {"ok": True}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


def test_get_json_retries_after_timeouts_and_rate_limits(monkeypatch):
    replies = [requests.exceptions.ReadTimeout("slow"), Response(429), requests.exceptions.ConnectionError(),
               Response(200, {"value": 1})]

    def fake_get(*args, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(sources.requests, "get", fake_get)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    assert sources.get_json("https://example.org", {}) == {"value": 1}


def test_get_json_gives_up_after_the_last_try(monkeypatch):
    def always_slow(*args, **kwargs):
        raise requests.exceptions.ReadTimeout("slow")

    monkeypatch.setattr(sources.requests, "get", always_slow)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    with pytest.raises(requests.exceptions.ReadTimeout):
        sources.get_json("https://example.org", {}, tries=3)


def test_merge_refresh_keeps_old_rows_and_prefers_new_ones(tmp_path):
    path = tmp_path / "weather.parquet"
    pd.DataFrame({"time": [1, 2, 3], "city": "A", "temperature_2m": [10.0, 11.0, 12.0]}).to_parquet(path)
    sources.merge_refresh(path, pd.DataFrame({"time": [3, 4], "city": "A", "temperature_2m": [12.5, 13.0]}))
    merged = pd.read_parquet(path)
    assert merged["time"].tolist() == [1, 2, 3, 4]
    assert merged["temperature_2m"].tolist() == [10.0, 11.0, 12.5, 13.0]


def test_forecast_refresh_only_asks_for_new_days(tmp_path, monkeypatch):
    locations = pd.DataFrame({"city": ["A"], "latitude": [51.5], "longitude": [-0.1]})
    stored = pd.DataFrame({"time": pd.date_range("2026-10-01", "2026-11-10 23:00", freq="h", tz="UTC"),
                           "city": "A", **{v: 1.0 for v in VARIABLES}})
    stored.to_parquet(tmp_path / "weather_forecast_day2.parquet", index=False)
    asked = []

    def fake_get_json(url, params):
        asked.append((params["start_date"], params["end_date"]))
        times = pd.date_range(params["start_date"], params["end_date"] + " 23:00", freq="h")
        return [{"hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                            **{v + FORECAST_SUFFIX: [2.0] * len(times) for v in VARIABLES}}}]

    monkeypatch.setattr(sources, "get_json", fake_get_json)
    monkeypatch.setattr(sources.time, "sleep", lambda seconds: None)
    today = pd.Timestamp("2026-11-19")
    monkeypatch.setattr(sources.pd.Timestamp, "now", classmethod(lambda cls, tz=None: today))
    sources.download_forecast_weather(tmp_path, locations, refresh=True)

    assert asked == [("2026-11-08", "2026-11-18")]
    merged = pd.read_parquet(tmp_path / "weather_forecast_day2.parquet")
    assert merged["time"].min() == stored["time"].min()
    assert merged["time"].max() == pd.Timestamp("2026-11-18 23:00", tz="UTC")
    assert merged["time"].is_unique
