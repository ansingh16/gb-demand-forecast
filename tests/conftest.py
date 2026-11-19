import numpy as np
import pandas as pd
import pytest
from fakes import make_model_folder


@pytest.fixture
def indo():
    """Five weeks of synthetic live demand: a daily cycle plus a slow trend."""
    start = pd.Timestamp("2025-01-01", tz="UTC")
    times = pd.date_range(start, periods=48 * 35, freq="30min")
    hours = times.hour + times.minute / 60
    values = 25_000 + 5_000 * np.sin((hours - 6) / 24 * 2 * np.pi) + np.arange(len(times))
    return pd.DataFrame({"start_utc": times, "INDO": values.round(),
                         "publish_utc": times + pd.Timedelta(minutes=30)})


@pytest.fixture
def model_folder(tmp_path):
    return make_model_folder(tmp_path / "models" / "latest")
