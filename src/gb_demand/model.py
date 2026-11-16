"""LightGBM point and quantile models (notebooks 06 and 07).

Trees cannot predict outside the range they were trained on, and GB demand has fallen about
4 GW since 2018. So every model predicts demand minus the latest 24 hours of live demand
(`level24`), and the level is added back afterwards.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import COLUMNS, QUANTILES


def target(frame):
    """What the trees learn: demand relative to the latest known level."""
    return frame["ND"] - frame["level24"]


def fit(train, params, quantiles=QUANTILES):
    """Fit the point model and one model per quantile on the same rows."""
    y = target(train)
    models = {"point": lgb.LGBMRegressor(**params).fit(train[COLUMNS], y)}
    for q in quantiles:
        models[q] = lgb.LGBMRegressor(objective="quantile", alpha=q, **params).fit(train[COLUMNS], y)
    return models


def predict(models, frame):
    """Point forecast and quantiles in MW. Quantiles are sorted so they never cross."""
    out = pd.DataFrame(index=frame.index)
    level = frame["level24"].to_numpy()
    out["point"] = models["point"].predict(frame[COLUMNS]) + level
    quantiles = [k for k in models if k != "point"]
    if quantiles:
        values = np.column_stack([models[q].predict(frame[COLUMNS]) + level for q in quantiles])
        out[[f"q{q}" for q in quantiles]] = np.sort(values, axis=1)
    return out


def save(models, folder):
    """Save each model as a LightGBM text file."""
    folder.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        model.booster_.save_model(folder / f"{key}.txt")


def load(folder, quantiles=QUANTILES):
    """Load models saved by `save` as boosters (they have the same `predict`)."""
    models = {"point": lgb.Booster(model_file=folder / "point.txt")}
    for q in quantiles:
        models[q] = lgb.Booster(model_file=folder / f"{q}.txt")
    return models
