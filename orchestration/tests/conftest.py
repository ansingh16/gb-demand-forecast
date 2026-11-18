import os
from pathlib import Path

os.environ.setdefault("AIRFLOW_HOME", str(Path(__file__).resolve().parents[1] / "airflow_home"))
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")
