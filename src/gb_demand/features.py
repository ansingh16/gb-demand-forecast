"""The feature table (notebook 06).

Every feature passes one test: was it known at the forecast origin (08:30 UTC on the day
before the target day)? Demand features come from Elexon's live demand (`INDO`), in training
and in use, so the inputs do not change between the two.
"""

import holidays as holiday_calendars
import numpy as np
import pandas as pd

from .config import FEATURES_FROM, LOCKDOWNS
from .cutoff import latest_demand_allowed
from .periods import local_naive, period_start_utc

SOLAR_CAPACITY_DELAY = pd.Timedelta(days=28)


def bank_holidays(last_year=None):
    """Bank holidays in England and Wales from 2018, as midnight timestamps."""
    last_year = last_year or pd.Timestamp.now().year + 1
    days = holiday_calendars.UK(subdiv="ENG", years=range(2018, last_year + 1))
    return set(pd.to_datetime(list(days)))


def christmas_days(dates):
    """Days from Christmas Day for 20 December to 7 January, missing otherwise."""
    year = np.where(dates.dt.month == 1, dates.dt.year - 1, dates.dt.year)
    christmas = pd.to_datetime(pd.Series(year, index=dates.index).astype(str) + "-12-25")
    offset = (dates - christmas).dt.days
    return offset.where(offset.between(-5, 13))


def reference_days_back(target, bank_holiday_days):
    """How many days back the 'similar day' is (notebook 05).

    A bank holiday is compared with the latest Sunday at least two days before. A normal day
    is compared with the same weekday a week before, or two weeks before if that was a holiday.
    """
    if target in bank_holiday_days:
        back = 2
        while (target - pd.Timedelta(days=back)).dayofweek != 6:
            back += 1
        return back
    return 14 if (target - pd.Timedelta(days=7)) in bank_holiday_days else 7


class DemandHistory:
    """Live demand (`INDO`) looked up by local clock time or by UTC period start."""

    def __init__(self, indo):
        indo = indo.drop_duplicates("start_utc", keep="last")
        self.by_local = (indo.assign(local=local_naive(indo["start_utc"]))
                         .drop_duplicates("local").set_index("local")["INDO"].sort_index())
        self.by_utc = indo.set_index("start_utc")["INDO"].sort_index()

    def days_before(self, rows, days):
        """INDO at the same local clock time `days` before each row (gaps filled from neighbours)."""
        keys = rows["local"] - pd.to_timedelta(np.broadcast_to(days, len(rows)), unit="D")
        values = self.by_local.reindex(keys.to_numpy())
        return pd.Series(values.to_numpy(), index=rows.index).ffill().bfill().to_numpy()

    def latest_level(self, target):
        """Mean INDO over the latest 48 periods known at the origin, and over the week before."""
        window = pd.date_range(end=latest_demand_allowed(target), periods=48, freq="30min")
        week_before = window - pd.Timedelta(days=7)
        return self.by_utc.reindex(window).mean(), self.by_utc.reindex(week_before).mean()


def target_rows(demand, bench, start=FEATURES_FROM):
    """One row per half-hour: NESO's historic demand joined to NESO's day-ahead benchmark."""
    key = ["start_utc", "Date", "Settlement_Period"]
    rows = demand[key + ["ND"]].merge(
        bench[key + ["Demand_Forecast", "Demand_Outturn"]], on=key, how="outer")
    return rows[rows["Date"] >= start].sort_values("start_utc").reset_index(drop=True)


def day_rows(target_date):
    """Rows for one local day not yet in any file (live use): 46, 48 or 50 periods."""
    day = pd.Timestamp(target_date).normalize()
    start = day.tz_localize("Europe/London")
    end = (day + pd.Timedelta(days=1)).tz_localize("Europe/London")
    periods = np.arange(1, int((end - start) / pd.Timedelta(minutes=30)) + 1)
    dates = pd.Series([f"{day:%Y-%m-%d}"] * len(periods))
    return pd.DataFrame({
        "start_utc": period_start_utc(dates, periods), "Date": dates, "Settlement_Period": periods,
        "ND": np.nan, "Demand_Forecast": np.nan, "Demand_Outturn": np.nan,
    })


def solar_capacity_by_day(demand):
    """NESO's embedded solar capacity, as it would be known: shifted 28 days later."""
    capacity = demand.groupby("Date")["EMBEDDED_SOLAR_CAPACITY"].max()
    capacity.index = pd.to_datetime(capacity.index) + SOLAR_CAPACITY_DELAY
    return capacity


def build_base(rows, indo, capacity, bank_holiday_days=None):
    """Calendar, demand and solar-capacity features for each row (everything but the weather)."""
    bank_holiday_days = bank_holiday_days or bank_holidays()
    history = DemandHistory(indo)
    rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["Date"])
    rows["local"] = local_naive(rows["start_utc"])

    calendar = pd.DataFrame(index=rows.index)
    calendar["slot"] = rows["local"].dt.hour * 2 + rows["local"].dt.minute // 30
    calendar["weekday"] = rows["date"].dt.dayofweek
    calendar["month"] = rows["date"].dt.month
    calendar["day_of_year"] = rows["date"].dt.dayofyear
    calendar["holiday"] = rows["date"].isin(bank_holiday_days).astype(int)
    calendar["christmas"] = christmas_days(rows["date"])
    calendar["lockdown"] = 0
    for start, end in LOCKDOWNS:
        calendar.loc[rows["date"].between(start, end), "lockdown"] = 1

    lags = pd.DataFrame(index=rows.index)
    lags["lag7"] = history.days_before(rows, 7)
    lags["lag14"] = history.days_before(rows, 14)
    days = pd.Series(rows["date"].unique())
    back = dict(zip(days, days.map(lambda d: reference_days_back(d, bank_holiday_days)), strict=True))
    lags["similar_day"] = history.days_before(rows, rows["date"].map(back).to_numpy())
    levels = {d: history.latest_level(d) for d in days}
    lags["level24"] = rows["date"].map(lambda d: levels[d][0])
    lags["level_ratio"] = lags["level24"] / rows["date"].map(lambda d: levels[d][1])

    known = capacity.reindex(pd.date_range(capacity.index.min(), rows["date"].max())).ffill()
    base = pd.concat([rows[["start_utc", "Date", "ND", "Demand_Forecast", "Demand_Outturn"]],
                      calendar, lags], axis=1)
    base["solar_capacity"] = rows["date"].map(known).to_numpy()
    return base


def add_weather(base, weather):
    """Join half-hourly weather and add the two weather features built from it."""
    table = base.merge(weather, on="start_utc", how="left")
    local_date = local_naive(table["start_utc"]).dt.date
    table["temp_day_mean"] = table.groupby(local_date)["temperature_2m"].transform("mean")
    table["radiation_x_capacity"] = table["shortwave_radiation"] * table["solar_capacity"] / 1000
    return table


def complete_weather_days(table):
    """True for rows whose local day has radiation for every half-hour."""
    covered = table.groupby("Date")["shortwave_radiation"].apply(lambda s: s.notna().all())
    return table["Date"].map(covered).fillna(False).astype(bool)
