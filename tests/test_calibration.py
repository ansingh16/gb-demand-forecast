import numpy as np
import pandas as pd
import pytest

from gb_demand.calibration import conformal_shift, reference
from gb_demand.metrics import coverage, interval_score, pinball, scores


def test_conformal_shift_gives_at_least_nominal_coverage():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(200):
        past, future = rng.normal(size=200), rng.normal(size=200)
        shift = conformal_shift(np.abs(past), 0.9)
        hits.append((np.abs(future) <= shift).mean())
    assert np.mean(hits) >= 0.89


def test_reference_intervals_cover_on_exchangeable_errors():
    rng = np.random.default_rng(1)
    def frame(n):
        slot = np.tile(np.arange(48), n // 48)
        point = 25_000 + 100 * slot
        return pd.DataFrame({"slot": slot, "point": point,
                             "Demand_Outturn": point + rng.normal(0, 50 + 10 * slot)})
    out = reference(frame(48 * 400), frame(48 * 200))
    assert coverage(out["Demand_Outturn"], out["c0.05"], out["c0.95"]) == pytest.approx(0.9, abs=0.02)
    assert (out["c0.05"] < out["c0.25"]).all() and (out["c0.75"] < out["c0.95"]).all()


def test_interval_score_by_hand():
    # width 10; second outcome misses by 5, penalty 2 / 0.1 * 5 = 100
    assert interval_score([5, 15], [0, 0], [10, 10], 0.9) == pytest.approx((10 + 110) / 2)


def test_pinball_and_scores():
    assert pinball([10], [8], 0.9) == pytest.approx(1.8)
    assert pinball([10], [12], 0.9) == pytest.approx(0.2)
    s = scores([110, 90], [100, 100])
    assert s["MAE"] == 10 and s["bias"] == 0 and s["MAPE"] == pytest.approx(10)
