"""Paths, settings and the fixed choices made in the notebooks."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

FIRST_YEAR = 2018
FEATURES_FROM = "2019-01-01"
TEST_START = "2024-06-01"

# Notebook 04: ten spread-out places, equal weights.
CITIES = ["London", "Birmingham", "Manchester", "Leeds", "Glasgow", "Bristol",
          "Newcastle upon Tyne", "Cardiff", "Plymouth", "Norwich"]
VARIABLES = ["temperature_2m", "shortwave_radiation", "wind_speed_10m"]
FORECAST_SUFFIX = "_previous_day2"

# Notebook 06: England's stay-at-home periods (kept in the table, not used by the model).
LOCKDOWNS = [("2020-03-23", "2020-06-30"), ("2020-11-05", "2020-12-02"), ("2021-01-06", "2021-03-28")]

# Notebook 06: the 17 features of the chosen model.
COLUMNS = ["slot", "weekday", "month", "day_of_year", "holiday", "christmas", "lag7", "lag14",
           "similar_day", "level24", "level_ratio", "solar_capacity", "temperature_2m",
           "temp_day_mean", "wind_speed_10m", "shortwave_radiation", "radiation_x_capacity"]

QUANTILES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
INTERVALS = [(0.25, 0.75), (0.1, 0.9), (0.05, 0.95)]


def load_params(path=ROOT / "params.yaml"):
    """Settings tracked by DVC: LightGBM parameters and the backtest rules."""
    with open(path) as handle:
        return yaml.safe_load(handle)
