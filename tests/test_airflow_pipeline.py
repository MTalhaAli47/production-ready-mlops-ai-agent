"""Unit tests for the composable Airflow training stages."""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from training import airflow_pipeline as pipeline


@pytest.fixture
def sample_dataset(tmp_path):
    path = tmp_path / "training_data.csv"
    pd.DataFrame(
        {
            "hours_studied": [1, 2, 3, 4, 5, 6],
            "attendance": [60, 65, 70, 75, 80, 85],
            "previous_score": [40, 45, 50, 55, 60, 65],
            "passed": [0, 0, 0, 1, 1, 1],
        }
    ).to_csv(path, index=False)
    return path


def test_validate_dataset_returns_provenance(
    tmp_path, sample_dataset, monkeypatch
):
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline, "DATASET_PATH", Path(sample_dataset.name)
    )
    monkeypatch.setattr(
        pipeline,
        "get_dataset_provenance",
        lambda: {"dataset_dvc_hash": "test-hash"},
    )

    result = pipeline.validate_dataset()

    assert result["dataset_rows"] == 6
    assert result["features"] == pipeline.FEATURES
    assert result["target"] == pipeline.TARGET
    assert result["dataset_provenance"]["dataset_dvc_hash"] == "test-hash"


def test_validate_dataset_rejects_missing_columns(
    tmp_path, sample_dataset, monkeypatch
):
    pd.DataFrame({"hours_studied": [1]}).to_csv(
        sample_dataset, index=False
    )
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline, "DATASET_PATH", Path(sample_dataset.name)
    )
    monkeypatch.setattr(pipeline, "get_dataset_provenance", dict)

    with pytest.raises(ValueError, match="missing required"):
        pipeline.validate_dataset()


def test_preprocess_dataset_writes_selected_columns(
    sample_dataset, tmp_path
):
    output = tmp_path / "workspace" / "prepared.csv"

    result = pipeline.preprocess_dataset(sample_dataset, output)

    prepared = pd.read_csv(output)
    assert result["dataset_rows"] == 6
    assert list(prepared.columns) == pipeline.FEATURES + [pipeline.TARGET]


def test_preprocess_dataset_rejects_missing_values(
    sample_dataset, tmp_path
):
    dataset = pd.read_csv(sample_dataset)
    dataset.loc[0, "hours_studied"] = np.nan
    dataset.to_csv(sample_dataset, index=False)

    with pytest.raises(ValueError, match="missing training values"):
        pipeline.preprocess_dataset(
            sample_dataset, tmp_path / "prepared.csv"
        )


def test_train_model_saves_fitted_pipeline(sample_dataset, tmp_path):
    model_path = tmp_path / "workspace" / "model.joblib"

    pipeline.train_model(sample_dataset, model_path)

    import joblib

    model = joblib.load(model_path)
    assert model_path.is_file()
    assert model.named_steps["scaler"].mean_.shape == (3,)
    assert model.named_steps["classifier"].classes_.tolist() == [0, 1]


def test_evaluate_and_log_uses_isolated_tracking_uri(
    sample_dataset, tmp_path, monkeypatch
):
    import joblib

    model = pipeline.create_model()
    dataset = pd.read_csv(sample_dataset)
    model.fit(dataset[pipeline.FEATURES], dataset[pipeline.TARGET])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)

    calls = {}
    active_run = SimpleNamespace(
        info=SimpleNamespace(run_id="airflow-test-run")
    )

    monkeypatch.setattr(
        pipeline,
        "cross_validate",
        lambda *args, **kwargs: {
            f"test_{metric}": np.array([0.8, 0.9, 1.0])
            for metric in pipeline.SCORING
        },
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update(tracking_uri=uri),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "get_experiment_by_name",
        lambda name: None,
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "create_experiment",
        lambda name, artifact_location: calls.update(
            experiment_name=name,
            artifact_location=artifact_location,
        ),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_experiment",
        lambda name: calls.update(experiment_name=name),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_params",
        lambda params: calls.update(params=params),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "set_tags",
        lambda tags: calls.update(tags=tags),
    )
    monkeypatch.setattr(
        pipeline.mlflow,
        "log_metrics",
        lambda metrics: calls.update(metrics=metrics),
    )
    monkeypatch.setattr(
        pipeline,
        "infer_signature",
        lambda *args: "test-signature",
    )

    @contextmanager
    def fake_start_run(**kwargs):
        calls["run_name"] = kwargs["run_name"]
        yield active_run

    monkeypatch.setattr(pipeline.mlflow, "start_run", fake_start_run)
    monkeypatch.setattr(
        pipeline.mlflow.sklearn,
        "log_model",
        lambda **kwargs: SimpleNamespace(model_uri="runs:/test/model"),
    )

    metrics_path = tmp_path / "metrics" / "result.json"
    result = pipeline.evaluate_and_log(
        sample_dataset,
        model_path,
        {"dataset_dvc_hash": "test-hash"},
        "sqlite:////isolated/tracking.db",
        "airflow-test",
        metrics_path,
        "file:///isolated/mlflow-artifacts",
    )

    assert result["run_id"] == "airflow-test-run"
    assert calls["tracking_uri"] == "sqlite:////isolated/tracking.db"
    assert calls["experiment_name"] == "airflow-test"
    assert calls["artifact_location"] == "file:///isolated/mlflow-artifacts"
    assert result["cross_validation_metrics"]["cv_f1_mean"] == pytest.approx(
        0.9
    )
    assert json.loads(metrics_path.read_text())["run_id"] == (
        "airflow-test-run"
    )


def _write_valid_metrics(path, rows=30):
    path.write_text(
        json.dumps(
            {
                "dataset_rows": rows,
                "cross_validation_metrics": {
                    f"cv_{metric}_mean": 0.8
                    for metric in pipeline.SCORING
                },
            }
        )
    )


def test_quality_gate_accepts_valid_metrics(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    _write_valid_metrics(metrics_path)

    result = pipeline.quality_gate(metrics_path)

    assert result["eligible"] is True
    assert result["minimum_dataset_rows"] == 30


def test_quality_gate_rejects_nine_row_dataset(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    _write_valid_metrics(metrics_path, rows=9)

    with pytest.raises(ValueError, match="at least 30"):
        pipeline.quality_gate(metrics_path)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "30"])
def test_quality_gate_rejects_invalid_threshold(tmp_path, value):
    metrics_path = tmp_path / "metrics.json"
    _write_valid_metrics(metrics_path)

    with pytest.raises(ValueError, match="positive integer"):
        pipeline.quality_gate(metrics_path, min_dataset_rows=value)


def test_quality_gate_rejects_non_finite_metric(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    _write_valid_metrics(metrics_path)
    result = json.loads(metrics_path.read_text())
    result["cross_validation_metrics"]["cv_f1_mean"] = float("nan")
    metrics_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="metric is invalid"):
        pipeline.quality_gate(metrics_path)


def test_quality_gate_rejects_missing_metric(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    _write_valid_metrics(metrics_path)
    result = json.loads(metrics_path.read_text())
    del result["cross_validation_metrics"]["cv_f1_mean"]
    metrics_path.write_text(json.dumps(result))

    with pytest.raises(ValueError, match="metric is invalid"):
        pipeline.quality_gate(metrics_path)
