"""HTTP service for live forecasts.

    uvicorn gb_demand.api:app

GET /health                 model details
GET /forecast?date=YYYY-MM-DD   half-hourly point forecast and 50%, 80% and 90% intervals (MW)
"""

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from .config import MODELS
from .cutoff import forecast_origin
from .live import Forecaster, NotReady

app = FastAPI(title="GB demand forecast", version="0.1.0")
_forecaster = None


def forecaster():
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster(Path(os.environ.get("GB_DEMAND_MODELS", MODELS / "latest")))
    return _forecaster


def tomorrow():
    return f"{pd.Timestamp.now(tz='Europe/London').normalize() + pd.Timedelta(days=1):%Y-%m-%d}"


@app.get("/health")
def health():
    return {"status": "ok", "model": forecaster().meta}


@app.get("/forecast")
def forecast(date: str = Query(default_factory=tomorrow, pattern=r"^\d{4}-\d{2}-\d{2}$")):
    made_at = pd.Timestamp.now(tz="UTC")
    try:
        frame = forecaster().forecast(date, now=made_at)
    except NotReady as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    columns = ["point", "c0.25", "c0.75", "c0.1", "c0.9", "c0.05", "c0.95"]
    names = ["point", "p25", "p75", "p10", "p90", "p05", "p95"]
    renamed = dict(zip(columns, names, strict=True))
    periods = frame[["start_utc", "local_time", *columns]].rename(columns=renamed)
    periods["start_utc"] = periods["start_utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
    periods[names] = periods[names].round(0)
    return {
        "date": date,
        "origin_utc": f"{forecast_origin(date):%Y-%m-%dT%H:%MZ}",
        "made_at_utc": f"{made_at:%Y-%m-%dT%H:%MZ}",
        "units": "MW, GB national demand (ND)",
        "periods": periods.to_dict(orient="records"),
    }
