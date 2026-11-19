"""Downloads from NESO, Elexon and Open-Meteo (notebooks 01, 02 and 04).

Files are saved under `data/raw` and are only downloaded when missing, or when `refresh` asks
for the latest data. Open-Meteo's free API allows 600 weighted calls a minute and 10,000 a day
for non-commercial use, so requests are spaced out and retried after a minute if refused.
"""

import time

import pandas as pd
import requests

from .config import CITIES, FIRST_YEAR, FORECAST_SUFFIX, VARIABLES
from .weather import to_long

NESO_API = "https://api.neso.energy/api/3/action/package_show"
ELEXON_INDO = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
GEOCODING = "https://geocoding-api.open-meteo.com/v1/search"
ERA5_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
FORECAST_ARCHIVE_START = "2024-01-01"


def get_json(url, params, tries=5, wait=65, timeout=120):
    """GET with retries: after HTTP 429 (too many requests), a timeout or a dropped connection.

    The wait grows with each attempt. The last error is raised if every attempt fails.
    """
    for attempt in range(1, tries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt == tries:
                raise
        else:
            if response.status_code != 429:
                response.raise_for_status()
                return response.json()
            if attempt == tries:
                response.raise_for_status()
        time.sleep(wait * attempt)


def merge_refresh(path, new, keys=("time", "city")):
    """Add newly downloaded rows to a stored file; where both have a row, the new one wins."""
    old = pd.read_parquet(path)
    both = pd.concat([old, new], ignore_index=True).drop_duplicates(list(keys), keep="last")
    both.sort_values(list(keys)).reset_index(drop=True).to_parquet(path, index=False)
    return both


def neso_package(package_id):
    return requests.get(NESO_API, params={"id": package_id}, timeout=60).json()["result"]


def download_historic(raw, refresh=False):
    """NESO historic demand data, one CSV per year from 2018."""
    package = neso_package("historic-demand-data")
    this_year = pd.Timestamp.now().year
    for resource in package["resources"]:
        name = resource["name"]
        if not (name.startswith("Historic Demand Data") and name[-4:].isdigit()):
            continue
        year = int(name[-4:])
        path = raw / f"demanddata_{year}.csv"
        if year < FIRST_YEAR or (path.exists() and not (refresh and year == this_year)):
            continue
        response = requests.get(resource["url"], timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)


def download_benchmark(raw, refresh=False):
    """NESO's day-ahead half-hourly demand forecast performance (the benchmark)."""
    path = raw / "neso_day_ahead_forecast.csv"
    if path.exists() and not refresh:
        return
    resource = neso_package("day-ahead-half-hourly-demand-forecast-performance")["resources"][0]
    response = requests.get(resource["url"], timeout=300)  # the URL redirects; requests follows it
    response.raise_for_status()
    path.write_bytes(response.content)
    with path.open(encoding="utf-8-sig") as handle:
        if not handle.readline().startswith("Month,Date"):
            raise ValueError("benchmark download is not the expected CSV")


def fetch_indo(start, end):
    """Elexon initial national demand outturn between two settlement dates (inclusive)."""
    chunks = []
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    while start <= end:
        stop = min(start + pd.Timedelta(days=27), end)  # the API allows 28 days per request
        response = requests.get(ELEXON_INDO, params={
            "settlementDateFrom": f"{start:%Y-%m-%d}", "settlementDateTo": f"{stop:%Y-%m-%d}",
            "format": "json"}, timeout=60)
        response.raise_for_status()
        chunks.append(pd.DataFrame(response.json()["data"]))
        start = stop + pd.Timedelta(days=1)
        time.sleep(0.2)
    return pd.concat(chunks, ignore_index=True)


def download_indo(raw, refresh=False):
    path = raw / "elexon_indo.parquet"
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    if not path.exists():
        fetch_indo(f"{FIRST_YEAR}-01-01", end).to_parquet(path, index=False)
    elif refresh:
        old = pd.read_parquet(path)
        new = fetch_indo(pd.Timestamp(old["settlementDate"].max()) - pd.Timedelta(days=1), end)
        both = pd.concat([old, new], ignore_index=True)
        both.drop_duplicates(["startTime", "publishTime"], keep="last").to_parquet(path, index=False)


def download_locations(raw):
    """Coordinates of the ten weather locations, from Open-Meteo's geocoding (GeoNames)."""
    path = raw / "weather_locations.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for city in CITIES:
        best = requests.get(GEOCODING, params={"name": city, "count": 5, "countryCode": "GB"},
                            timeout=30).json()["results"][0]
        rows.append({"city": city, "latitude": best["latitude"], "longitude": best["longitude"],
                     "population": best.get("population"), "nation": best.get("admin1")})
    pd.DataFrame(rows).to_csv(path, index=False)
    return pd.read_csv(path)


def coordinates(locations):
    return {"latitude": ",".join(locations["latitude"].astype(str)),
            "longitude": ",".join(locations["longitude"].astype(str))}


def download_era5(raw, locations, refresh=False, revised_days=120):
    """Hourly ERA5 reanalysis (actual weather) for each location, one year per request.

    A refresh only fetches the latest `revised_days`: the most recent ERA5 data is preliminary
    and is replaced by the final version a few months later, while older data does not change.
    """
    path = raw / "weather_era5.parquet"
    if path.exists() and not refresh:
        return
    today = pd.Timestamp.now().normalize()
    first = pd.Timestamp(f"{FIRST_YEAR}-01-01")
    if path.exists():
        first = pd.read_parquet(path, columns=["time"])["time"].max().tz_localize(None).normalize()
        first -= pd.Timedelta(days=revised_days)
    parts = []
    for year in range(first.year, today.year + 1):
        start, stop = max(first, pd.Timestamp(f"{year}-01-01")), min(today, pd.Timestamp(f"{year}-12-31"))
        payload = get_json(ERA5_ARCHIVE, {
            **coordinates(locations), "hourly": ",".join(VARIABLES), "models": "era5",
            "start_date": f"{start:%Y-%m-%d}", "end_date": f"{stop:%Y-%m-%d}", "timezone": "GMT"})
        parts.append(to_long(payload, locations["city"]))
        time.sleep(30)
    new = pd.concat(parts, ignore_index=True).dropna(subset=VARIABLES, how="all")
    if path.exists():
        merge_refresh(path, new)
    else:
        new.to_parquet(path, index=False)


def download_forecast_weather(raw, locations, refresh=False):
    """Archived forecasts made 48 hours before valid time (`_previous_day2`), from 2024.

    `_previous_day1` (24 hours before) would leak: for an evening target hour its model run
    starts after the 08:30 UTC origin on the day before.

    A forecast already archived never changes, so a refresh only fetches the days after the
    latest one stored (plus two days of overlap).
    """
    path = raw / "weather_forecast_day2.parquet"
    if path.exists() and not refresh:
        return
    end = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
    first = pd.Timestamp(FORECAST_ARCHIVE_START)
    if path.exists():
        first = pd.read_parquet(path, columns=["time"])["time"].max().tz_localize(None).normalize()
        first -= pd.Timedelta(days=2)
    if first > end:
        return
    parts = []
    for start in [first, *pd.date_range(first + pd.offsets.QuarterBegin(1, startingMonth=1), end, freq="QS")]:
        stop = min(start + pd.offsets.QuarterEnd(0), end)
        payload = get_json(PREVIOUS_RUNS, {
            **coordinates(locations), "hourly": ",".join(v + FORECAST_SUFFIX for v in VARIABLES),
            "start_date": f"{start:%Y-%m-%d}", "end_date": f"{stop:%Y-%m-%d}", "timezone": "GMT"})
        parts.append(to_long(payload, locations["city"], suffix=FORECAST_SUFFIX))
        time.sleep(20)
    new = pd.concat(parts, ignore_index=True)
    if path.exists():
        merge_refresh(path, new)
    else:
        new.to_parquet(path, index=False)


def fetch_day2_forecast(locations, target_date):
    """`_previous_day2` forecasts for one local day, hourly, every location.

    The same source as the backtest. Every hour of day D comes from a model run started 48
    hours earlier, before the 08:30 UTC origin on D-1, so the values are the same whether they
    are fetched at the origin or replayed later. The day before and after are included so the
    half-hours at the edges can be interpolated; hours not yet forecast are dropped.
    """
    day = pd.Timestamp(target_date)
    payload = get_json(PREVIOUS_RUNS, {
        **coordinates(locations), "hourly": ",".join(v + FORECAST_SUFFIX for v in VARIABLES),
        "timezone": "GMT", "start_date": f"{day - pd.Timedelta(days=1):%Y-%m-%d}",
        "end_date": f"{day + pd.Timedelta(days=1):%Y-%m-%d}"})
    return to_long(payload, locations["city"], suffix=FORECAST_SUFFIX).dropna(subset=VARIABLES)


def download_all(raw, refresh=False):
    raw.mkdir(parents=True, exist_ok=True)
    download_historic(raw, refresh)
    download_benchmark(raw, refresh)
    download_indo(raw, refresh)
    locations = download_locations(raw)
    download_era5(raw, locations, refresh)
    download_forecast_weather(raw, locations, refresh)
