"""Monitoring: input drift and forecast quality over time (notebook 09).

Two questions, checked month by month:

1. **Have the inputs moved away from what the model was trained on?** Measured with the
   population stability index (PSI) of each feature. Demand and weather are seasonal, so a
   month is compared with the *same calendar month* in the training years, not with the whole
   year; otherwise every July would look like drift.
2. **Is the forecast still as good as it was?** Monthly MAE against NESO, and whether the 90%
   interval still contains about 90% of outcomes. Outturn is published the next day, so this
   can be checked daily in live use.
"""

import numpy as np
import pandas as pd

from .metrics import coverage

PSI_MODERATE, PSI_LARGE = 0.1, 0.25
DRIFT_FEATURES = ["temperature_2m", "wind_speed_10m", "shortwave_radiation", "level24",
                  "lag7", "solar_capacity", "radiation_x_capacity"]
# Inputs that follow known trends (falling demand, growing solar). Their drift is expected and
# handled by the monthly refit, so it is reported but not raised as an alert.
TREND_FEATURES = ["level24", "lag7", "solar_capacity", "radiation_x_capacity"]
COVERAGE_BAND = (0.85, 0.95)
VALIDATION_GAP_MW = 174  # LightGBM minus NESO MAE on validation, notebook 06


def psi(expected, actual, bins=10):
    """Population stability index of `actual` against `expected`, on deciles of `expected`."""
    expected = pd.Series(expected).dropna().to_numpy()
    actual = pd.Series(actual).dropna().to_numpy()
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(reference, current, features=DRIFT_FEATURES):
    """PSI per feature for each month of `current`, against the same calendar month of `reference`."""
    current_month = pd.to_datetime(current["Date"]).dt.to_period("M")
    rows = {}
    for period, frame in current.groupby(current_month):
        same_month = reference[reference["month"] == period.month]
        rows[str(period)] = {f: psi(same_month[f], frame[f]) for f in features}
    return pd.DataFrame(rows).T


def performance(test):
    """Monthly MAE for this model and NESO, and 90% interval coverage."""
    y = test["Demand_Outturn"]
    month = pd.to_datetime(test["Date"]).dt.to_period("M").astype(str)
    out = pd.DataFrame({
        "MAE": (test["point"] - y).abs().groupby(month).mean(),
        "NESO MAE": (test["Demand_Forecast"] - y).abs().groupby(month).mean(),
        "90% coverage": test.groupby(month).apply(
            lambda g: coverage(g["Demand_Outturn"], g["c0.05"], g["c0.95"]), include_groups=False),
    })
    out["gap to NESO"] = out["MAE"] - out["NESO MAE"]
    return out


def alerts(drift, quality, reference_gap):
    """Plain-language flags: large drift, coverage outside its band, an unusual gap to NESO."""
    found = []
    for month, row in drift.iterrows():
        for feature, value in row.items():
            if value >= PSI_LARGE and feature not in TREND_FEATURES:
                found.append({"month": month, "check": f"drift: {feature}", "value": round(value, 2)})
    low, high = COVERAGE_BAND
    for month, row in quality.iterrows():
        if not low <= row["90% coverage"] <= high:
            found.append({"month": month, "check": "90% coverage outside 85-95%",
                          "value": round(row["90% coverage"], 3)})
        if row["gap to NESO"] > 2 * reference_gap:
            found.append({"month": month, "check": "gap to NESO over twice the validation gap",
                          "value": round(row["gap to NESO"])})
    return pd.DataFrame(found, columns=["month", "check", "value"])


def markdown_report(drift, quality, flagged):
    """A short Markdown report for `reports/monitoring.md`."""
    def table(frame, digits):
        return frame.round(digits).to_markdown()

    return "\n".join([
        "# Monitoring report", "",
        "Built by `python -m gb_demand.monitor` from the backtest output.",
        "Notebook 09 explains each check.", "",
        f"PSI bands: under {PSI_MODERATE} stable, {PSI_MODERATE} to {PSI_LARGE} moderate, "
        f"over {PSI_LARGE} large. Each month is compared with the same calendar month in 2019 to 2023.", "",
        "## Alerts", "", flagged.round(3).to_markdown(index=False) if len(flagged) else "None.", "",
        "## Forecast quality by month (MW)", "", table(quality, 3), "",
        "## Input drift by month (PSI)", "", table(drift, 2), "",
    ])


def main():
    from .config import PROCESSED, REPORTS

    features = pd.read_parquet(PROCESSED / "features_actual.parquet")
    forecast_features = pd.read_parquet(PROCESSED / "features_forecast.parquet")
    test = pd.read_parquet(PROCESSED / "test_forecasts.parquet")
    dates = pd.to_datetime(features["Date"])
    reference = features[(dates >= "2019-01-01") & (dates < "2024-01-01")]
    current = forecast_features[forecast_features["start_utc"].isin(test["start_utc"])]
    drift = feature_drift(reference, current)
    quality = performance(test)
    flagged = alerts(drift, quality, reference_gap=VALIDATION_GAP_MW)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "monitoring.md").write_text(markdown_report(drift, quality, flagged))
    print(flagged.to_string(index=False) if len(flagged) else "no alerts")


if __name__ == "__main__":
    main()
