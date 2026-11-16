import numpy as np
import pandas as pd
import pytest

from gb_demand.weather import to_half_hours


def hourly(values_by_variable, hours=6, cities=("A", "B")):
    times = pd.date_range("2025-06-01 00:00", periods=hours, freq="h", tz="UTC")
    frames = []
    for i, city in enumerate(cities):
        frame = pd.DataFrame({"time": times, "city": city})
        for name, values in values_by_variable.items():
            frame[name] = np.asarray(values, dtype=float) + i * 2  # second city 2 units higher
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_radiation_comes_from_the_hour_that_contains_the_half_hour():
    raw = hourly({"temperature_2m": range(6), "wind_speed_10m": range(6),
                  "shortwave_radiation": [0, 100, 200, 300, 400, 500]})
    out = to_half_hours(raw).set_index("start_utc")
    # 01:00 and 01:30 lie in the hour ending 02:00, whose mean over two cities is 201
    assert out.loc[pd.Timestamp("2025-06-01 01:00", tz="UTC"), "shortwave_radiation"] == 201
    assert out.loc[pd.Timestamp("2025-06-01 01:30", tz="UTC"), "shortwave_radiation"] == 201


def test_temperature_is_interpolated_to_the_midpoint():
    raw = hourly({"temperature_2m": [0, 10, 20, 30, 40, 50], "wind_speed_10m": range(6),
                  "shortwave_radiation": range(6)})
    out = to_half_hours(raw).set_index("start_utc")
    # midpoint of 02:30 to 03:00 is 02:45: 25 + 0.75 * 10 = 27.5, plus the city mean offset 1
    assert out.loc[pd.Timestamp("2025-06-01 02:30", tz="UTC"), "temperature_2m"] == pytest.approx(28.5)
