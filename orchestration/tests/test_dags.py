from itertools import pairwise
from pathlib import Path

from airflow.dag_processing.dagbag import DagBag

DAGS = Path(__file__).resolve().parents[2] / "dags"


def test_dags_import_without_errors():
    bag = DagBag(dag_folder=str(DAGS))
    assert bag.import_errors == {}
    assert set(bag.dag_ids) == {"daily_forecast", "daily_score", "monthly_retrain"}


def test_forecast_targets_the_day_after_the_run():
    dag = DagBag(dag_folder=str(DAGS)).dags["daily_forecast"]
    task = dag.get_task("forecast_tomorrow")
    assert "forecast --date {{ macros.ds_add(ds, 1) }}" in task.bash_command
    assert dag.catchup and dag.max_active_runs == 1


def test_score_runs_on_the_day_before_and_check_follows():
    dag = DagBag(dag_folder=str(DAGS)).dags["daily_score"]
    assert "score --date {{ macros.ds_add(ds, -1) }}" in dag.get_task("score_yesterday").bash_command
    assert dag.get_task("check_recent_scores").upstream_task_ids == {"score_yesterday"}


def test_retrain_promotes_only_after_the_gate():
    dag = DagBag(dag_folder=str(DAGS)).dags["monthly_retrain"]
    order = ["refresh_data", "rebuild_and_backtest", "gate", "train_staging", "promote"]
    for before, after in pairwise(order):
        assert dag.get_task(after).upstream_task_ids == {before}
    assert not dag.catchup


def test_dvc_command_survives_a_path_with_spaces():
    dag = DagBag(dag_folder=str(DAGS)).dags["monthly_retrain"]
    assert 'PATH="' in dag.get_task("rebuild_and_backtest").bash_command
