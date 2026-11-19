import json

import numpy as np
import pandas as pd
import pytest
from fakes import fake_indo, fake_weather, make_model_folder

from gb_demand import ops
from gb_demand.config import COLUMNS
from gb_demand.live import NotReady

DAY = "2025-02-03"


@pytest.fixture
def stored(model_folder, tmp_path):
    """A forecast for DAY saved the way the daily task saves it."""
    live = tmp_path / "live"
    ops.forecast(DAY, folder=model_folder, live=live, fetch_indo=fake_indo, fetch_weather=fake_weather)
    return live


def test_forecast_writes_one_file_per_day_and_reruns_cleanly(model_folder, stored):
    again = ops.forecast(DAY, folder=model_folder, live=stored,
                         fetch_indo=fake_indo, fetch_weather=fake_weather)
    files = list((stored / "forecasts").iterdir())
    assert [f.name for f in files] == [f"{DAY}.parquet"]
    pd.testing.assert_series_equal(pd.read_parquet(files[0])["point"], again["point"])


def test_score_waits_for_the_whole_day(stored):
    def half_day(start, end):
        return fake_indo(start, end).iloc[:24]

    with pytest.raises(NotReady):
        ops.score(DAY, live=stored, fetch_indo=half_day)


def test_score_is_stored_once_per_day(stored, tmp_path):
    no_benchmark = tmp_path / "missing.parquet"
    first = ops.score(DAY, live=stored, fetch_indo=fake_indo, benchmark=no_benchmark)
    ops.score(DAY, live=stored, fetch_indo=fake_indo, benchmark=no_benchmark)
    scores = pd.read_parquet(stored / "scores.parquet")
    assert len(scores) == 1 and scores["date"].iloc[0] == DAY
    assert scores["mae"].iloc[0] == pytest.approx(first["mae"])
    assert 0 <= first["inside_90%"] <= 1 and np.isnan(first["neso_mae"])


def write_scores(live, days, inside):
    live.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": [f"2025-01-{d:02d}" for d in range(1, days + 1)], "periods": 48,
                  "mae": 800.0, "inside_90%": inside}).to_parquet(live / "scores.parquet", index=False)


def test_check_skips_until_there_are_enough_days(tmp_path):
    write_scores(tmp_path, 5, 0.9)
    with pytest.raises(ops.Skip):
        ops.check(live=tmp_path, reference_mae=880)


def test_check_passes_in_band_and_alerts_outside(tmp_path):
    write_scores(tmp_path, 25, 0.9)
    ops.check(live=tmp_path, reference_mae=880)
    write_scores(tmp_path, 25, 0.7)
    with pytest.raises(RuntimeError, match="90% interval"):
        ops.check(live=tmp_path, reference_mae=880)


@pytest.fixture
def processed(tmp_path):
    """Feature tables from December 2024 to mid-February 2025 with a simple daily pattern."""
    rng = np.random.default_rng(2)
    times = pd.date_range("2024-12-01", "2025-02-15 23:30", freq="30min", tz="UTC")
    frame = pd.DataFrame(rng.normal(size=(len(times), len(COLUMNS))), columns=COLUMNS)
    frame["level24"] = 27_000.0
    frame["christmas"] = np.nan                     # empty outside the Christmas period, as in real data
    christmas = (times >= "2024-12-20") & (times < "2025-01-08")
    frame.loc[christmas, "christmas"] = 0.0
    frame["Date"] = times.strftime("%Y-%m-%d")
    frame["ND"] = 27_000 + 400 * frame["temperature_2m"] + rng.normal(0, 50, len(times))
    folder = tmp_path / "processed"
    folder.mkdir()
    frame.to_parquet(folder / "features_actual.parquet")
    frame.to_parquet(folder / "features_forecast.parquet")
    return folder


def test_gate_compares_on_days_neither_model_has_seen(model_folder, processed, tmp_path):
    report = tmp_path / "gate.json"
    result = ops.gate(folder=model_folder, processed=processed, report=report, tolerance=10)
    assert result["held_out"] == "2025-02-02 to 2025-02-15"
    assert result["candidate_trained_until"] == "2025-01-26"   # seven days before the held-out days
    assert result["current_trained_until"] < "2025-02-02"
    assert json.loads(report.read_text())["passed"]
    with pytest.raises(RuntimeError):
        ops.gate(folder=model_folder, processed=processed, report=report, tolerance=-1)


def test_gate_skips_when_the_current_model_saw_the_held_out_days(model_folder, processed, tmp_path):
    meta = json.loads((model_folder / "meta.json").read_text())
    (model_folder / "meta.json").write_text(json.dumps({**meta, "trained_until": "2025-02-10"}))
    with pytest.raises(ops.Skip):
        ops.gate(folder=model_folder, processed=processed, report=tmp_path / "gate.json")


def test_promote_keeps_the_old_model_for_a_rollback(tmp_path):
    models = tmp_path / "models"
    make_model_folder(models / "latest")
    staged = make_model_folder(models / "staging")
    (staged / "meta.json").write_text(json.dumps({"trained_until": "2025-02-28"}))
    ops.promote(models)
    assert json.loads((models / "latest" / "meta.json").read_text())["trained_until"] == "2025-02-28"
    assert json.loads((models / "previous" / "meta.json").read_text())["trained_until"] == "2025-01-20"
    assert not (models / "staging").exists()
    with pytest.raises(NotReady):
        ops.promote(models)
