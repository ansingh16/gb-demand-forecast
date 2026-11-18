"""Airflow DAGs for the daily forecast.

Each task runs a command from `gb_demand.ops` in the project's own environment (`.venv`), so
Airflow and the model never share dependencies. Every task works from the run's logical date
and only uses data known at the forecast origin, so a missed day can be filled in later
(catchup) and gives the same forecast it would have given on time.
"""

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.trigger import CronTriggerTimetable

PROJECT = Path(__file__).resolve().parents[1]
OPS = f"cd {PROJECT} && env -u VIRTUAL_ENV .venv/bin/python -m gb_demand.ops"

DEFAULTS = {
    "owner": "gb-demand",
    "retries": 3,
    "retry_delay": timedelta(minutes=15),  # Open-Meteo rate limits, late data
}

with DAG(
    dag_id="daily_forecast",
    description="Forecast tomorrow's half-hourly GB demand at 08:35 UTC, after the 08:30 origin",
    schedule=CronTriggerTimetable("35 8 * * *", timezone="UTC"),
    start_date=pendulum.datetime(2026, 11, 17, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    default_args=DEFAULTS,
    tags=["gb-demand"],
) as daily_forecast:
    BashOperator(
        task_id="forecast_tomorrow",
        bash_command=f"{OPS} forecast --date {{{{ macros.ds_add(ds, 1) }}}}",
    )
