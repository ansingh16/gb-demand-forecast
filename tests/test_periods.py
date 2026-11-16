import pandas as pd
import pytest

from gb_demand.periods import local_day_utc, parse_settlement_dates, period_start_utc


@pytest.mark.parametrize("day, periods", [("2024-01-15", 48), ("2024-07-15", 48),
                                          ("2024-03-31", 46), ("2024-10-27", 50)])
def test_periods_are_half_hours_and_fill_the_local_day(day, periods):
    starts = period_start_utc([day] * periods, list(range(1, periods + 1)))
    assert (starts.diff().dropna() == pd.Timedelta(minutes=30)).all()
    first, end = local_day_utc(day)
    assert starts.iloc[0] == first
    assert starts.iloc[-1] + pd.Timedelta(minutes=30) == end


def test_summer_period_one_starts_at_23_utc_the_day_before():
    start = period_start_utc(["2024-07-15"], [1]).iloc[0]
    assert start == pd.Timestamp("2024-07-14 23:00", tz="UTC")


def test_autumn_clock_change_repeats_local_one_am_but_not_utc():
    starts = period_start_utc(["2024-10-27"] * 50, list(range(1, 51)))
    local = starts.dt.tz_convert("Europe/London").dt.strftime("%H:%M")
    assert (local == "01:00").sum() == 2
    assert starts.is_unique


@pytest.mark.parametrize("values", [["01-Jan-2019", "02-Jan-2019"], ["01-Jan-19", "02-Jan-19"],
                                    ["2019-01-01", "2019-01-02"]])
def test_each_file_date_format_is_parsed(values):
    parsed = parse_settlement_dates(pd.Series(values))
    assert list(parsed) == [pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-02")]


def test_unknown_date_format_raises():
    with pytest.raises(ValueError):
        parse_settlement_dates(pd.Series(["2019/01/01"]))
