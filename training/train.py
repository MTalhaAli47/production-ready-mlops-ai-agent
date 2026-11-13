"""Train a student pass-prediction model and track it with MLflow."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path("data/raw/training_data.csv")
METADATA_PATH = Path("data/raw/training_data.csv.dvc")
METRICS_PATH = PROJECT_ROOT / "training" / "metrics.json"

FEATURES = ["hours_studied", "attendance", "previous_score"]
TARGET = "passed"
RANDOM_STATE = 42
N_SPLITS = 3


def get_dataset_provenance():
    """Capture and validate the exact DVC dataset version used for training."""
    dataset_file = PROJECT_ROOT / DATASET_PATH
    metadata_file = PROJECT_ROOT / METADATA_PATH

    if not dataset_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError("Dataset or DVC metadata file is missing")

    git_status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            METADATA_PATH.as_posix(),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )

    if git_status.stdout.strip():
        raise RuntimeError(
            "DVC metadata has uncommitted changes; commit it before training"
        )

    metadata = yaml.safe_load(metadata_file.read_text())

    if not isinstance(metadata, dict):
        raise TypeError("DVC metadata must be a YAML mapping")

    outputs = metadata.get("outs")

    if not isinstance(outputs, list):
        raise TypeError("DVC metadata has no valid outputs list")

    matching_outputs = [
        output
        for output in outputs
        if isinstance(output, dict)
        and output.get("path") == DATASET_PATH.name
    ]

    if len(matching_outputs) != 1:
        raise ValueError("Expected exactly one matching DVC dataset output")

    output = matching_outputs[0]
    dvc_hash = output.get("md5")

    if not isinstance(dvc_hash, str) or not dvc_hash.strip():
        raise ValueError("DVC dataset hash is missing")

    actual_hash = hashlib.md5(
        dataset_file.read_bytes(),
        usedforsecurity=False,
    ).hexdigest()

    if actual_hash != dvc_hash:
        raise ValueError("Dataset content does not match its DVC metadata hash")

    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    ).stdout.strip()

    return {
        "dataset_path": DATASET_PATH.as_posix(),
        "dataset_git_revision": git_revision,
        "dataset_dvc_hash": dvc_hash,
        "dataset_dvc_metadata_path": METADATA_PATH.as_posix(),
        "dataset_version": f"md5:{dvc_hash}",
    }


def create_model():
    """Build the classification pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train():
    """Evaluate, train, and log the model and dataset provenance."""
    provenance = get_dataset_provenance()

    dataset = pd.read_csv(PROJECT_ROOT / DATASET_PATH)

    required_columns = FEATURES + [TARGET]

    if not set(required_columns).issubset(dataset.columns):
        raise ValueError("Dataset is missing required training columns")

    if dataset[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing training values")

    X = dataset[FEATURES]
    y = dataset[TARGET]

    if y.nunique() != 2:
        raise ValueError("Target must contain exactly two classes")

    if y.value_counts().min() < N_SPLITS:
        raise ValueError(
            f"Each target class needs at least {N_SPLITS} rows "
            "for stratified cross-validation"
        )

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}",
    )
    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "student-pass-prediction",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = create_model()

    with mlflow.start_run(
        run_name="student-pass-logistic-regression"
    ) as active_run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "scaler": "StandardScaler",
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "cv_strategy": "StratifiedKFold",
                "cv_splits": N_SPLITS,
                "dataset_rows": len(dataset),
                "feature_count": len(FEATURES),
            }
        )

        mlflow.set_tags(
            {
                **provenance,
                "target_column": TARGET,
                "evaluation_method": "stratified_cross_validation",
            }
        )

        scores = cross_validate(
            model,
            X,
            y,
            cv=cross_validation,
            scoring=scoring,
            return_train_score=False,
            error_score="raise",
        )

        metrics = {}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            metrics[f"cv_{metric}_mean"] = float(values.mean())
            metrics[f"cv_{metric}_std"] = float(values.std())

        mlflow.log_metrics(metrics)

        # Train the final model on all available rows after evaluation.
        model.fit(X, y)

        signature = infer_signature(X, model.predict(X))

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="student_pass_prediction_model",
            signature=signature,
            input_example=X.head(3),
        )

        run_id = active_run.info.run_id

    result = {
        "success": True,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "model_uri": model_info.model_uri,
        "dataset_rows": len(dataset),
        "cross_validation_metrics": metrics,
        "dataset_provenance": provenance,
        "warning": (
            "Only nine dataset rows are available. "
            "Metrics are for demonstration, not reliable "
            "estimates of real-world generalization."
        ),
    }

    METRICS_PATH.write_text(json.dumps(result, indent=2) + "\n")

    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
