from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="week17_mlops_monitoring",
    start_date=datetime(2026, 1, 1),
    schedule="@weekly",
    catchup=False,
    tags=["mlops", "evidently", "telco"],
) as dag:
    drift_check = BashOperator(
        task_id="run_drift_check",
        bash_command="cd $AIRFLOW_HOME/../../week\\ 17 && uv run python scripts/monitor.py",
    )
    retraining_recommendation = BashOperator(
        task_id="recommend_retraining",
        bash_command="echo 'Review artifacts/reports/drift_summary.json; retrain when drift persists.'",
    )
    drift_check >> retraining_recommendation

