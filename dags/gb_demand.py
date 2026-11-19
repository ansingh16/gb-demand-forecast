"""Airflow DAGs: forecast every day, score every day, retrain every month.

Each task runs a command from `gb_demand.ops` or `gb_demand.pipeline` in the project's own
environment (`.venv`), so Airflow and the model never share dependencies. The forecast task
works from the run's logical date and only uses data known at the forecast origin, so a
missed day can be filled in later (catchup) and gives the same forecast it would have given
on time. A task that exits with 99 is marked skipped (BashOperator's default), which the ops
commands use for "nothing to do yet".
"""

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import DAG
from airflow.timetables.trigger import CronTriggerTimetable

PROJECT = Path(__file__).resolve().parents[1]
OPS = f"cd {PROJECT} && env -u VIRTUAL_ENV .venv/bin/python -m gb_demand.ops"
PIPELINE = f"cd {PROJECT} && env -u VIRTUAL_ENV .venv/bin/python -m gb_demand.pipeline"
# PATH is quoted: under WSL it holds Windows folders with spaces ("/mnt/c/Program Files/...")
DVC = f'cd {PROJECT} && env -u VIRTUAL_ENV PATH="{PROJECT}/.venv/bin:$PATH" dvc'

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

with DAG(
    dag_id="daily_score",
    description="Score yesterday's forecast against Elexon outturn, then check recent coverage",
    schedule=CronTriggerTimetable("0 2 * * *", timezone="UTC"),
    start_date=pendulum.datetime(2026, 11, 18, tz="UTC"),
    catchup=True,
    max_active_runs=1,
    default_args=DEFAULTS,
    tags=["gb-demand"],
) as daily_score:
    score = BashOperator(
        task_id="score_yesterday",
        bash_command=f"{OPS} score --date {{{{ macros.ds_add(ds, -1) }}}}",
        retries=6,
        retry_delay=timedelta(hours=1),  # outturn for the last periods can arrive late
    )
    check = BashOperator(
        task_id="check_recent_scores",
        bash_command=f"{OPS} check",
        retries=0,  # a failure here is the alert
    )
    score >> check

with DAG(
    dag_id="monthly_retrain",
    description="Refresh the data, rerun the backtest, and promote a retrained model if it passes the gate",
    schedule=CronTriggerTimetable("0 3 2 * *", timezone="UTC"),
    start_date=pendulum.datetime(2026, 12, 1, tz="UTC"),
    catchup=False,  # retraining for a past month has no value
    max_active_runs=1,
    default_args={**DEFAULTS, "retries": 1},
    tags=["gb-demand"],
) as monthly_retrain:
    refresh = BashOperator(task_id="refresh_data", bash_command=f"{PIPELINE} download --refresh",
                           execution_timeout=timedelta(hours=1))
    rebuild = BashOperator(task_id="rebuild_and_backtest", bash_command=f"{DVC} repro backtest",
                           execution_timeout=timedelta(hours=2))
    gate = BashOperator(task_id="gate", bash_command=f"{OPS} gate", retries=0)
    stage = BashOperator(task_id="train_staging", bash_command=f"{PIPELINE} train --out models/staging")
    promote = BashOperator(task_id="promote", bash_command=f"{OPS} promote")
    refresh >> rebuild >> gate >> stage >> promote
