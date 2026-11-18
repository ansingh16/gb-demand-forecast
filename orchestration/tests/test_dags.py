from pathlib import Path

from airflow.dag_processing.dagbag import DagBag

DAGS = Path(__file__).resolve().parents[2] / "dags"


def test_dags_import_without_errors():
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    assert "daily_forecast" in bag.dag_ids


def test_forecast_targets_the_day_after_the_run():
    dag = DagBag(dag_folder=str(DAGS)).dags["daily_forecast"]
    task = dag.get_task("forecast_tomorrow")
    assert "forecast --date {{ macros.ds_add(ds, 1) }}" in task.bash_command
    assert dag.catchup and dag.max_active_runs == 1
