#!/bin/sh
# Run Airflow for this project from its own environment, with state kept in airflow_home/.
#   orchestration/airflow.sh standalone                         scheduler and UI on localhost:8080
#   orchestration/airflow.sh dags test daily_forecast 2026-11-18  one run, no scheduler
HERE=$(cd "$(dirname "$0")" && pwd)
export AIRFLOW_HOME="$HERE/airflow_home"
export AIRFLOW__CORE__DAGS_FOLDER="$(cd "$HERE/../dags" && pwd)"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
# standalone starts its parts by calling `airflow`, so the environment goes first on PATH
export PATH="$HERE/.venv/bin:$PATH"
exec env -u VIRTUAL_ENV airflow "$@"
