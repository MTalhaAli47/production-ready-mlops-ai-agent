"""Composable, independently testable stages for the Airflow training DAG."""

import json
import math
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import StratifiedKFold, cross_validate

from training.train import (
    DATASET_PATH,
    FEATURES,
    N_SPLITS,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET,
    create_model,
    get_dataset_provenance,
)

SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
DEFAULT_MIN_DATASET_ROWS = 30


def validate_dataset():
    """Validate dataset structure, class distribution, and DVC provenance."""
    provenance = get_dataset_provenance()
    dataset_path = PROJECT_ROOT / DATASET_PATH
    dataset = pd.read_csv(dataset_path)

    required_columns = FEATURES + [TARGET]
    if not set(required_columns).issubset(dataset.columns):
        raise ValueError("Dataset is missing required training columns")

    if dataset[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing training values")

    target = dataset[TARGET]
    if target.nunique() != 2:
        raise ValueError("Target must contain exactly two classes")

    if target.value_counts().min() < N_SPLITS:
        raise ValueError(
            f"Each target class needs at least {N_SPLITS} rows "
            "for stratified cross-validation"
        )

    return {
        "dataset_path": str(dataset_path),
        "dataset_rows": len(dataset),
        "features": FEATURES,
        "target": TARGET,
        "dataset_provenance": provenance,
    }


def preprocess_dataset(dataset_path, output_path):
    """Select required columns without fitting transformations outside CV."""
    dataset = pd.read_csv(dataset_path)
    required_columns = FEATURES + [TARGET]

    if not set(required_columns).issubset(dataset.columns):
        raise ValueError("Dataset is missing required training columns")

    if dataset[required_columns].isnull().any().any():
        raise ValueError("Dataset contains missing training values")

    if dataset[TARGET].nunique() != 2:
        raise ValueError("Target must contain exactly two classes")

    if dataset[TARGET].value_counts().min() < N_SPLITS:
        raise ValueError(
            f"Each target class needs at least {N_SPLITS} rows "
            "for stratified cross-validation"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset[required_columns].to_csv(output, index=False)

    return {
        "preprocessed_path": str(output),
        "dataset_rows": len(dataset),
    }


def train_model(preprocessed_path, model_path):
    """Fit the existing sklearn pipeline and persist the fitted model."""
    dataset = pd.read_csv(preprocessed_path)
    X = dataset[FEATURES]
    y = dataset[TARGET]

    model = create_model()
    model.fit(X, y)

    output = Path(model_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)

    return {
        "model_path": str(output),
        "dataset_rows": len(dataset),
        "feature_count": len(FEATURES),
    }


def evaluate_and_log(
    preprocessed_path,
    model_path,
    dataset_provenance,
    tracking_uri,
    experiment_name,
    metrics_path,
    artifact_location,
):
    """Run stratified CV and log metrics and the fitted model to MLflow."""
    dataset = pd.read_csv(preprocessed_path)
    X = dataset[FEATURES]
    y = dataset[TARGET]
    model = joblib.load(model_path)

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_location,
        )
    elif experiment.artifact_location != artifact_location:
        raise ValueError(
            "Existing MLflow experiment uses an unexpected artifact location"
        )

    mlflow.set_experiment(experiment_name)

    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scores = cross_validate(
        create_model(),
        X,
        y,
        cv=cross_validation,
        scoring=SCORING,
        return_train_score=False,
        error_score="raise",
    )

    metrics = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        metrics[f"cv_{metric}_mean"] = float(values.mean())
        metrics[f"cv_{metric}_std"] = float(values.std())

    with mlflow.start_run(
        run_name="airflow-student-pass-logistic-regression"
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
                **dataset_provenance,
                "target_column": TARGET,
                "evaluation_method": "stratified_cross_validation",
                "pipeline": "airflow",
            }
        )
        mlflow.log_metrics(metrics)

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
        "dataset_provenance": dataset_provenance,
    }

    output = Path(metrics_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    return result


def quality_gate(
    metrics_path,
    min_dataset_rows=DEFAULT_MIN_DATASET_ROWS,
):
    """Reject promotion eligibility when evidence is insufficient or invalid."""
    if (
        isinstance(min_dataset_rows, bool)
        or not isinstance(min_dataset_rows, int)
        or min_dataset_rows < 1
    ):
        raise ValueError("Minimum dataset rows must be a positive integer")

    result = json.loads(Path(metrics_path).read_text())
    dataset_rows = result.get("dataset_rows")
    metrics = result.get("cross_validation_metrics")

    if (
        isinstance(dataset_rows, bool)
        or not isinstance(dataset_rows, int)
        or dataset_rows < 0
    ):
        raise ValueError("Evaluation result has an invalid dataset row count")

    if not isinstance(metrics, dict):
        raise TypeError("Evaluation result has no valid CV metrics")

    required_metrics = [f"cv_{metric}_mean" for metric in SCORING]
    for name in required_metrics:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"Evaluation metric is invalid: {name}")

    if dataset_rows < min_dataset_rows:
        raise ValueError(
            "Promotion eligibility rejected: "
            f"{dataset_rows} dataset rows; "
            f"at least {min_dataset_rows} are required"
        )

    return {
        "eligible": True,
        "dataset_rows": dataset_rows,
        "minimum_dataset_rows": min_dataset_rows,
        "reason": "Dataset size and evaluation metrics passed the configured gate",
    }
