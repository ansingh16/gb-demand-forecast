import numpy as np
import pandas as pd
import pytest

from gb_demand.cutoff import forecast_origin, latest_demand_allowed
from gb_demand.features import DemandHistory, bank_holidays, build_base, day_rows, reference_days_back
from gb_demand.periods import local_day_utc


def test_origin_is_0830_utc_the_day_before():
    assert forecast_origin("2025-01-15") == pd.Timestamp("2025-01-14 08:30", tz="UTC")
    assert forecast_origin(pd.Timestamp("2025-07-15")) == pd.Timestamp("2025-07-14 08:30", tz="UTC")


@pytest.mark.parametrize("day, first, last", [("2025-01-15", 15.5, 39.5), ("2025-07-15", 14.5, 38.5)])
def test_lead_times(day, first, last):
    start, end = local_day_utc(day)
    origin = forecast_origin(day)
    assert (start - origin) / pd.Timedelta(hours=1) == first
    assert (end - origin) / pd.Timedelta(hours=1) == last


def test_latest_demand_period_ends_before_origin():
    latest = latest_demand_allowed("2025-01-15")
    assert latest == pd.Timestamp("2025-01-14 07:30", tz="UTC")
    assert latest + pd.Timedelta(minutes=30) < forecast_origin("2025-01-15")


def test_similar_day_is_never_closer_than_two_days():
    holidays = bank_holidays(2027)
    for day in pd.date_range("2018-01-01", "2026-12-31"):
        assert reference_days_back(day, holidays) >= 2


def test_similar_day_rules():
    holidays = bank_holidays(2027)
    assert reference_days_back(pd.Timestamp("2025-03-12"), holidays) == 7       # ordinary Wednesday
    assert reference_days_back(pd.Timestamp("2025-05-05"), holidays) == 8       # bank holiday: Sunday before
    assert reference_days_back(pd.Timestamp("2025-05-12"), holidays) == 14      # week after a bank holiday
    assert reference_days_back(pd.Timestamp("2025-12-25"), holidays) == 4       # Christmas, a Thursday


def test_latest_level_uses_the_48_periods_before_the_cutoff(indo):
    history = DemandHistory(indo)
    target = pd.Timestamp("2025-01-30")
    level, _ = history.latest_level(target)
    window = indo[(indo["start_utc"] <= latest_demand_allowed(target))].tail(48)
    assert level == pytest.approx(window["INDO"].mean())


def test_features_ignore_demand_published_after_the_origin(indo):
    """Changing every value after the cut-off must not change a single feature."""
    target = "2025-02-03"
    rows = day_rows(target)
    capacity = pd.Series(15_000.0, index=pd.date_range("2024-12-01", "2025-02-28"))
    before = build_base(rows, indo, capacity)

    tampered = indo.copy()
    late = tampered["start_utc"] > latest_demand_allowed(target)
    assert late.any()
    tampered.loc[late, "INDO"] = 1e9
    after = build_base(rows, tampered, capacity)
    pd.testing.assert_frame_equal(before, after)
    assert np.isfinite(before[["lag7", "lag14", "similar_day", "level24"]].to_numpy()).all()


@pytest.mark.parametrize("day, periods", [("2025-03-30", 46), ("2025-06-01", 48), ("2025-10-26", 50)])
def test_live_day_has_every_settlement_period(day, periods):
    rows = day_rows(day)
    assert len(rows) == periods
    assert list(rows["Settlement_Period"]) == list(range(1, periods + 1))
