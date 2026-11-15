"""Airflow orchestration for the MLOps assignment training pipeline."""

from datetime import UTC, datetime, timedelta

from airflow.sdk import dag, task

from training.airflow_pipeline import (
    evaluate_and_log,
    preprocess_dataset,
    quality_gate,
    train_model,
    validate_dataset,
)

WORKSPACE = "/opt/mlops-workspace"
TRACKING_URI = f"sqlite:///{WORKSPACE}/mlflow.db"
ARTIFACT_LOCATION = f"file://{WORKSPACE}/mlflow-artifacts"
EXPERIMENT_NAME = "airflow-student-pass-prediction"


@dag(
    dag_id="mlops_a03_training_pipeline",
    description="Validate, preprocess, train, evaluate, and quality-gate the model",
    start_date=datetime(2025, 1, 1, tzinfo=UTC),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "training", "mlflow"],
)
def mlops_training_pipeline():

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def validate_stage():
        return validate_dataset()

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def preprocess_stage(validation):
        return preprocess_dataset(
            validation["dataset_path"],
            f"{WORKSPACE}/preprocessed.csv",
        )

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def train_stage(preprocessing):
        return train_model(
            preprocessing["preprocessed_path"],
            f"{WORKSPACE}/model.joblib",
        )

    @task(retries=1, retry_delay=timedelta(minutes=1))
    def evaluate_stage(validation, preprocessing, training):
        return evaluate_and_log(
            preprocessing["preprocessed_path"],
            training["model_path"],
            validation["dataset_provenance"],
            TRACKING_URI,
            EXPERIMENT_NAME,
            f"{WORKSPACE}/metrics.json",
            ARTIFACT_LOCATION,
        )

    @task
    def quality_gate_stage(evaluation):
        return quality_gate(f"{WORKSPACE}/metrics.json")

    @task
    def promotion_eligibility_stage(gate_result):
        if gate_result.get("eligible") is not True:
            raise ValueError("Model is not eligible for promotion")
        return {
            "promotion_eligible": True,
            "action": "decision_only_no_registry_or_deployment",
            "dataset_rows": gate_result["dataset_rows"],
        }

    validation = validate_stage()
    preprocessing = preprocess_stage(validation)
    training = train_stage(preprocessing)
    evaluation = evaluate_stage(validation, preprocessing, training)
    gate = quality_gate_stage(evaluation)
    promotion_eligibility_stage(gate)


mlops_training_pipeline()
