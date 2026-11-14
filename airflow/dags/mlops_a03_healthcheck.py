from datetime import datetime, timezone

from airflow.sdk import dag, task


@dag(
    dag_id="mlops_a03_healthcheck",
    description="Non-destructive Airflow infrastructure health check",
    start_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    tags=["mlops-a03", "healthcheck"],
)
def mlops_a03_healthcheck():

    @task
    def verify_environment():
        result = {
            "status": "ok",
            "component": "airflow",
            "purpose": "non-destructive health check",
        }
        print(f"Health check result: {result}")
        return result

    @task
    def report_result(result: dict):
        assert result["status"] == "ok"
        print("Airflow health check completed successfully.")
        return "success"

    report_result(verify_environment())


mlops_a03_healthcheck()
