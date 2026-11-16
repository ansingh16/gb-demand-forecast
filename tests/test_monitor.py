import numpy as np
import pandas as pd

from gb_demand.monitor import alerts, feature_drift, psi


def test_psi_is_near_zero_for_the_same_distribution_and_large_for_a_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(size=20_000)
    assert psi(base, rng.normal(size=5_000)) < 0.02
    assert psi(base, rng.normal(1.0, 1, size=5_000)) > 0.25


def test_drift_compares_with_the_same_calendar_month():
    """A normal July is stable against past Julys, but would look like drift against whole years."""
    rng = np.random.default_rng(1)
    times = pd.date_range("2019-01-01", "2023-12-31 23:30", freq="30min")
    seasonal = 10 - 8 * np.cos((times.dayofyear - 15) / 365 * 2 * np.pi)
    reference = pd.DataFrame({"Date": times.strftime("%Y-%m-%d"), "month": times.month,
                              "temperature_2m": seasonal + rng.normal(0, 2, len(times))})
    july = pd.date_range("2025-07-01", "2025-07-31 23:30", freq="30min")
    current = pd.DataFrame({"Date": july.strftime("%Y-%m-%d"), "month": 7,
                            "temperature_2m": 10 - 8 * np.cos((july.dayofyear - 15) / 365 * 2 * np.pi)
                            + rng.normal(0, 2, len(july))})
    drift = feature_drift(reference, current, ["temperature_2m"])
    assert drift.loc["2025-07", "temperature_2m"] < 0.1
    assert psi(reference["temperature_2m"], current["temperature_2m"]) > 0.25


def test_alerts_flag_coverage_and_gap():
    drift = pd.DataFrame({"temperature_2m": [0.05, 0.4], "solar_capacity": [3.0, 3.0]},
                         index=["2025-01", "2025-02"])
    quality = pd.DataFrame({"90% coverage": [0.9, 0.7], "gap to NESO": [100, 500]},
                           index=["2025-01", "2025-02"])
    flagged = alerts(drift, quality, reference_gap=174)
    assert set(flagged["month"]) == {"2025-02"}
    assert len(flagged) == 3
