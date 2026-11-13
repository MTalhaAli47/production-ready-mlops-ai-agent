"""Automated tests for the MLflow tools."""

from pathlib import Path

import pytest
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sqlalchemy.exc import SQLAlchemyError

from agent.tools import mlflow_tools


@pytest.fixture
def mlflow_env(tmp_path, monkeypatch):
    """Provide an isolated SQLite tracking store for each test."""
    database = tmp_path / "mlflow.db"
    tracking_uri = f"sqlite:///{database.as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)

    experiment_id = client.create_experiment("test-experiment")

    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: client,
    )

    return client, experiment_id


def create_run(client, experiment_id, accuracy):
    """Create a completed run with a metric, parameter, and tag."""
    run = client.create_run(experiment_id)
    client.log_metric(run.info.run_id, "accuracy", accuracy)
    client.log_param(run.info.run_id, "model", "test-model")
    client.set_tag(run.info.run_id, "purpose", "automated-test")
    client.set_terminated(run.info.run_id, status="FINISHED")
    return run.info.run_id


@pytest.mark.parametrize(
    "limit",
    [0, -1, 1001, True, False, "10", None],
)
def test_validate_limit_rejects_invalid_values(limit):
    assert mlflow_tools._validate_limit(limit) is False


@pytest.mark.parametrize("limit", [1, 5, 100, 1000])
def test_validate_limit_accepts_valid_values(limit):
    assert mlflow_tools._validate_limit(limit) is True


@pytest.mark.parametrize("metric", ["accuracy", "val_loss", "f1.score", "m-1"])
def test_validate_metric_accepts_valid_names(metric):
    assert mlflow_tools._validate_metric(metric) is True


@pytest.mark.parametrize("metric", ["", " ", "accuracy score", "metric;DROP", None])
def test_validate_metric_rejects_invalid_names(metric):
    assert mlflow_tools._validate_metric(metric) is False


def test_list_experiments_returns_experiment(mlflow_env):
    _, experiment_id = mlflow_env

    result = mlflow_tools.list_experiments()

    assert result["success"] is True
    assert any(
        item["experiment_id"] == experiment_id
        for item in result["experiments"]
    )


def test_list_experiments_rejects_invalid_limit():
    result = mlflow_tools.list_experiments(max_results=0)

    assert result == {
        "success": False,
        "error": "Invalid max_results limit",
    }


def test_list_runs_by_experiment_name(mlflow_env):
    client, experiment_id = mlflow_env
    run_id = create_run(client, experiment_id, 0.92)

    result = mlflow_tools.list_runs(experiment_name="test-experiment")

    assert result["success"] is True
    assert result["experiment_id"] == experiment_id
    assert result["runs"][0]["run_id"] == run_id
    assert result["runs"][0]["metrics"]["accuracy"] == 0.92
    assert result["runs"][0]["params"]["model"] == "test-model"
    assert result["runs"][0]["tags"]["purpose"] == "automated-test"


def test_list_runs_by_experiment_id(mlflow_env):
    client, experiment_id = mlflow_env
    create_run(client, experiment_id, 0.85)

    result = mlflow_tools.list_runs(experiment_id=experiment_id)

    assert result["success"] is True
    assert len(result["runs"]) == 1


def test_list_runs_rejects_both_experiment_identifiers():
    result = mlflow_tools.list_runs(
        experiment_name="test-experiment",
        experiment_id="1",
    )

    assert result["success"] is False
    assert "not both" in result["error"]


def test_list_runs_reports_missing_experiment(mlflow_env):
    result = mlflow_tools.list_runs(experiment_name="missing-experiment")

    assert result["success"] is False
    assert result["error"] == "Experiment not found"


def test_list_runs_rejects_invalid_limit():
    result = mlflow_tools.list_runs(max_results=0)

    assert result["success"] is False


def test_get_run_details_returns_run(mlflow_env):
    client, experiment_id = mlflow_env
    run_id = create_run(client, experiment_id, 0.95)

    result = mlflow_tools.get_run_details(run_id)

    assert result["success"] is True
    assert result["run"]["run_id"] == run_id
    assert result["run"]["metrics"]["accuracy"] == 0.95


@pytest.mark.parametrize("run_id", ["", " ", None, 123])
def test_get_run_details_rejects_invalid_run_id(run_id):
    result = mlflow_tools.get_run_details(run_id)

    assert result["success"] is False
    assert result["error"] == "Invalid run_id"


def test_get_best_run_returns_highest_metric(mlflow_env):
    client, experiment_id = mlflow_env
    create_run(client, experiment_id, 0.80)
    best_run_id = create_run(client, experiment_id, 0.97)

    result = mlflow_tools.get_best_run("test-experiment", "accuracy")

    assert result["success"] is True
    assert result["direction"] == "DESC"
    assert result["best_run"]["run_id"] == best_run_id


def test_get_best_run_supports_ascending_order(mlflow_env):
    client, experiment_id = mlflow_env
    best_run_id = create_run(client, experiment_id, 0.60)
    create_run(client, experiment_id, 0.90)

    result = mlflow_tools.get_best_run(
        "test-experiment",
        "accuracy",
        ascending=True,
    )

    assert result["success"] is True
    assert result["direction"] == "ASC"
    assert result["best_run"]["run_id"] == best_run_id


def test_get_best_run_reports_missing_experiment(mlflow_env):
    result = mlflow_tools.get_best_run("missing-experiment", "accuracy")

    assert result["success"] is False
    assert result["error"] == "Experiment not found"


def test_get_best_run_reports_missing_metric(mlflow_env):

    result = mlflow_tools.get_best_run("test-experiment", "accuracy")

    assert result["success"] is False
    assert "No runs found" in result["error"]


def test_get_best_run_rejects_invalid_arguments():
    assert not mlflow_tools.get_best_run("", "accuracy")["success"]
    assert not mlflow_tools.get_best_run(
        "test-experiment", "accuracy score"
    )["success"]
    assert not mlflow_tools.get_best_run(
        "test-experiment", "accuracy", ascending="yes"
    )["success"]


def test_compare_runs_orders_runs_by_metric(mlflow_env):
    client, experiment_id = mlflow_env
    create_run(client, experiment_id, 0.75)
    create_run(client, experiment_id, 0.88)
    create_run(client, experiment_id, 0.96)

    result = mlflow_tools.compare_runs("test-experiment", "accuracy")

    assert result["success"] is True
    assert result["direction"] == "DESC"
    assert len(result["runs"]) == 3
    assert [
        run["metrics"]["accuracy"] for run in result["runs"]
    ] == [0.96, 0.88, 0.75]


def test_compare_runs_supports_ascending_order(mlflow_env):
    client, experiment_id = mlflow_env
    create_run(client, experiment_id, 0.75)
    create_run(client, experiment_id, 0.95)

    result = mlflow_tools.compare_runs(
        "test-experiment",
        "accuracy",
        ascending=True,
    )

    assert result["success"] is True
    assert result["direction"] == "ASC"
    assert result["runs"][0]["metrics"]["accuracy"] == 0.75


def test_compare_runs_rejects_invalid_arguments():
    assert not mlflow_tools.compare_runs("", "accuracy")["success"]
    assert not mlflow_tools.compare_runs(
        "test-experiment", "bad metric"
    )["success"]
    assert not mlflow_tools.compare_runs(
        "test-experiment", "accuracy", max_results=0
    )["success"]
    assert not mlflow_tools.compare_runs(
        "test-experiment", "accuracy", ascending=1
    )["success"]


def test_compare_runs_reports_missing_experiment(mlflow_env):
    result = mlflow_tools.compare_runs("missing-experiment", "accuracy")

    assert result["success"] is False
    assert result["error"] == "Experiment not found"


def test_list_registered_models_returns_registered_model(mlflow_env):
    client, _ = mlflow_env
    client.create_registered_model("test-model-registry")

    result = mlflow_tools.list_registered_models()

    assert result["success"] is True
    assert result["models"][0]["name"] == "test-model-registry"
    assert result["models"][0]["latest_versions"] == []


def test_list_registered_models_rejects_invalid_limit():
    result = mlflow_tools.list_registered_models(max_results=0)

    assert result["success"] is False


def test_list_run_artifacts_returns_artifact_list(mlflow_env):
    client, experiment_id = mlflow_env
    run_id = create_run(client, experiment_id, 0.90)

    result = mlflow_tools.list_run_artifacts(run_id)

    assert result["success"] is True
    assert result["run_id"] == run_id
    assert isinstance(result["artifacts"], list)


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../secret", "data/../../secret", "folder\\secret"],
)
def test_list_run_artifacts_rejects_unsafe_paths(path):
    result = mlflow_tools.list_run_artifacts("some-run-id", path=path)

    assert result["success"] is False
    assert result["error"] == "Invalid artifact path"


def test_list_run_artifacts_rejects_invalid_run_id():
    result = mlflow_tools.list_run_artifacts("")

    assert result["success"] is False
    assert result["error"] == "Invalid run_id"


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("MLflow unavailable"),
        SQLAlchemyError("Database unavailable"),
    ],
)
def test_list_experiments_handles_tracking_errors(monkeypatch, exception):
    def raise_error():
        raise exception

    monkeypatch.setattr(mlflow_tools, "_get_client", raise_error)

    result = mlflow_tools.list_experiments()

    assert result["success"] is False
    assert str(exception) in result["error"]


def test_get_client_uses_default_tracking_uri(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, tracking_uri):
            captured["tracking_uri"] = tracking_uri

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_tools, "MlflowClient", FakeClient)

    result = mlflow_tools._get_client()

    assert isinstance(result, FakeClient)
    assert captured["tracking_uri"] == mlflow_tools.DEFAULT_TRACKING_URI


def test_get_client_uses_environment_tracking_uri(monkeypatch):
    captured = {}
    configured_uri = "sqlite:///custom-mlflow.db"

    class FakeClient:
        def __init__(self, tracking_uri):
            captured["tracking_uri"] = tracking_uri

    monkeypatch.setenv("MLFLOW_TRACKING_URI", configured_uri)
    monkeypatch.setattr(mlflow_tools, "MlflowClient", FakeClient)

    result = mlflow_tools._get_client()

    assert isinstance(result, FakeClient)
    assert captured["tracking_uri"] == configured_uri


def test_list_runs_rejects_invalid_experiment_id(mlflow_env):
    result = mlflow_tools.list_runs(experiment_id=" ")

    assert result["success"] is False
    assert result["error"] == "Experiment not found"


def test_list_runs_rejects_missing_experiment_identifier(mlflow_env):
    result = mlflow_tools.list_runs()

    assert result["success"] is False
    assert result["error"] == "Experiment not found"


@pytest.mark.parametrize(
    "tool_call",
    [
        lambda: mlflow_tools.list_runs(experiment_name="test-experiment"),
        lambda: mlflow_tools.get_run_details("some-run-id"),
        lambda: mlflow_tools.get_best_run("test-experiment", "accuracy"),
        lambda: mlflow_tools.compare_runs("test-experiment", "accuracy"),
        lambda: mlflow_tools.list_registered_models(),
        lambda: mlflow_tools.list_run_artifacts("some-run-id"),
    ],
)
@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("MLflow unavailable"),
        SQLAlchemyError("Database unavailable"),
    ],
)
def test_mlflow_tools_handle_tracking_errors(
    monkeypatch,
    tool_call,
    exception,
):
    def raise_error():
        raise exception

    monkeypatch.setattr(mlflow_tools, "_get_client", raise_error)

    result = tool_call()

    assert result["success"] is False
    assert str(exception) in result["error"]


def test_compare_runs_across_experiments_by_names(mlflow_env):
    client, first_experiment_id = mlflow_env
    second_experiment_id = client.create_experiment("second-experiment")

    first_run_id = create_run(client, first_experiment_id, 0.82)
    second_run_id = create_run(client, second_experiment_id, 0.96)

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_names=["test-experiment", "second-experiment"],
    )

    assert result["success"] is True
    assert result["experiment_names"] == [
        "test-experiment",
        "second-experiment",
    ]
    assert result["experiment_ids"] == [
        first_experiment_id,
        second_experiment_id,
    ]
    assert [run["run_id"] for run in result["runs"]] == [
        second_run_id,
        first_run_id,
    ]
    assert result["scanned_runs"] == 2
    assert result["truncated"] is False


def test_compare_runs_across_experiments_by_ids(mlflow_env):
    client, first_experiment_id = mlflow_env
    second_experiment_id = client.create_experiment("second-experiment")

    create_run(client, first_experiment_id, 0.82)
    create_run(client, second_experiment_id, 0.96)

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_ids=[first_experiment_id, second_experiment_id],
    )

    assert result["success"] is True
    assert result["experiment_ids"] == [
        first_experiment_id,
        second_experiment_id,
    ]
    assert result["direction"] == "DESC"
    assert len(result["runs"]) == 2


@pytest.mark.parametrize(
    ("ascending", "expected_metrics", "expected_direction"),
    [
        (False, [0.90, 0.80, None], "DESC"),
        (True, [0.80, 0.90, None], "ASC"),
    ],
)
def test_compare_runs_across_experiments_sorting_and_missing_metric(
    mlflow_env,
    ascending,
    expected_metrics,
    expected_direction,
):
    client, experiment_id = mlflow_env

    create_run(client, experiment_id, 0.80)
    create_run(client, experiment_id, 0.90)

    missing_metric_run = client.create_run(experiment_id)
    client.set_terminated(
        missing_metric_run.info.run_id,
        status="FINISHED",
    )

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_ids=[experiment_id],
        ascending=ascending,
    )

    assert result["success"] is True
    assert result["direction"] == expected_direction
    assert [
        run["metrics"].get("accuracy")
        for run in result["runs"]
    ] == expected_metrics
    assert result["runs"][-1]["run_id"] == missing_metric_run.info.run_id


@pytest.mark.parametrize(
    ("kwargs", "expected_error"),
    [
        (
            {},
            "Provide exactly one of experiment_names or experiment_ids",
        ),
        (
            {
                "experiment_names": ["test-experiment"],
                "experiment_ids": ["1"],
            },
            "Provide exactly one of experiment_names or experiment_ids",
        ),
        (
            {"experiment_names": []},
            "Experiment identifiers must be a non-empty list of strings",
        ),
        (
            {"experiment_ids": []},
            "Experiment identifiers must be a non-empty list of strings",
        ),
        (
            {"experiment_names": "test-experiment"},
            "Experiment identifiers must be a non-empty list of strings",
        ),
        (
            {"experiment_ids": [1]},
            "Experiment identifiers must be a non-empty list of strings",
        ),
        (
            {"experiment_names": [""]},
            "Experiment identifiers must be a non-empty list of strings",
        ),
        (
            {"experiment_names": ["   "]},
            "Experiment identifiers must be a non-empty list of strings",
        ),
        (
            {
                "experiment_names": [
                    "test-experiment",
                    "test-experiment",
                ],
            },
            "Duplicate experiment identifiers are not allowed",
        ),
        (
            {"experiment_names": ["test-experiment"], "metric": ""},
            "Invalid metric name",
        ),
        (
            {
                "experiment_names": ["test-experiment"],
                "max_results": 0,
            },
            "Invalid max_results limit",
        ),
        (
            {
                "experiment_names": ["test-experiment"],
                "ascending": "yes",
            },
            "ascending must be a boolean",
        ),
    ],
)
def test_compare_runs_across_experiments_rejects_invalid_arguments(
    kwargs,
    expected_error,
):
    kwargs.setdefault("metric", "accuracy")

    result = mlflow_tools.compare_runs_across_experiments(**kwargs)

    assert result == {
        "success": False,
        "error": expected_error,
    }


def test_compare_runs_across_experiments_rejects_missing_experiment_name(
    mlflow_env,
):
    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_names=["does-not-exist"],
    )

    assert result == {
        "success": False,
        "error": "Experiment not found: does-not-exist",
    }


def test_compare_runs_across_experiments_rejects_missing_experiment_id(
    mlflow_env,
):
    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_ids=["999999"],
    )

    assert result["success"] is False
    assert result["error"] == "No Experiment with id=999999 exists"


def test_compare_runs_across_experiments_with_no_runs(mlflow_env):
    _, experiment_id = mlflow_env

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_ids=[experiment_id],
    )

    assert result["success"] is True
    assert result["runs"] == []
    assert result["scanned_runs"] == 0
    assert result["truncated"] is False


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("MLflow unavailable"),
        SQLAlchemyError("Database unavailable"),
    ],
)
def test_compare_runs_across_experiments_handles_client_errors(
    monkeypatch,
    exception,
):
    def raise_error():
        raise exception

    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        raise_error,
    )

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_names=["test-experiment"],
    )

    assert result == {
        "success": False,
        "error": str(exception),
    }


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("MLflow search failed"),
        SQLAlchemyError("Database search failed"),
    ],
)
def test_compare_runs_across_experiments_handles_search_errors(
    mlflow_env,
    monkeypatch,
    exception,
):
    client, experiment_id = mlflow_env

    def raise_error(**kwargs):
        raise exception

    monkeypatch.setattr(client, "search_runs", raise_error)

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_ids=[experiment_id],
    )

    assert result == {
        "success": False,
        "error": str(exception),
    }


def test_compare_runs_across_experiments_reports_scan_truncation(
    mlflow_env,
    monkeypatch,
):
    client, experiment_id = mlflow_env

    create_run(client, experiment_id, 0.70)
    create_run(client, experiment_id, 0.80)
    create_run(client, experiment_id, 0.90)

    monkeypatch.setattr(
        mlflow_tools,
        "MAX_SCAN_RESULTS",
        2,
    )

    result = mlflow_tools.compare_runs_across_experiments(
        metric="accuracy",
        experiment_ids=[experiment_id],
    )

    assert result["success"] is True
    assert result["scanned_runs"] == 2
    assert result["scan_limit"] == 2
    assert len(result["runs"]) == 2
    assert result["truncated"] is True


def create_registered_version(
    client,
    experiment_id,
    model_name,
    *,
    run_id=None,
    source=None,
    tags=None,
    description=None,
):
    """Create a real registered-model version in the isolated test store."""
    if run_id is None and source is None:
        run_id = create_run(client, experiment_id, 0.90)

    if source is None:
        source = f"runs:/{run_id}/model"

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(model_name)

    return client.create_model_version(
        name=model_name,
        source=source,
        run_id=run_id,
        tags=tags,
        description=description,
        await_creation_for=0,
    )


def test_get_registered_model_details_without_versions(mlflow_env):
    client, _ = mlflow_env
    client.create_registered_model("empty-registry-model")

    result = mlflow_tools.get_registered_model_details(
        "empty-registry-model"
    )

    assert result["success"] is True
    assert result["model"]["name"] == "empty-registry-model"
    assert result["model"]["versions"] == []
    assert isinstance(result["model"]["tags"], dict)
    assert isinstance(result["model"]["aliases"], dict)


def test_get_registered_model_details_returns_metadata_and_aliases(
    mlflow_env,
):
    client, experiment_id = mlflow_env
    name = "metadata-registry-model"

    create_registered_version(client, experiment_id, name)

    client.update_registered_model(
        name,
        description="Registry metadata test.",
    )
    client.set_registered_model_tag(name, "team", "ml-platform")
    client.set_registered_model_alias(name, "champion", "1")

    result = mlflow_tools.get_registered_model_details(name)

    assert result["success"] is True
    model = result["model"]
    assert model["description"] == "Registry metadata test."
    assert model["creation_timestamp"] is not None
    assert model["last_updated_timestamp"] is not None
    assert model["tags"] == {"team": "ml-platform"}
    assert model["aliases"] == {"champion": "1"}


def test_get_registered_model_details_returns_version(mlflow_env):
    client, experiment_id = mlflow_env
    name = "single-version-model"

    create_registered_version(
        client,
        experiment_id,
        name,
        tags={"validation": "passed"},
        description="First registered version.",
    )

    result = mlflow_tools.get_registered_model_details(name)

    assert result["success"] is True
    versions = result["model"]["versions"]
    assert len(versions) == 1
    assert versions[0]["version"] == "1"
    assert versions[0]["description"] == "First registered version."
    assert versions[0]["creation_timestamp"] is not None
    assert versions[0]["last_updated_timestamp"] is not None
    assert versions[0]["run_id"] is not None
    assert versions[0]["source"].startswith("runs:/")
    assert versions[0]["tags"] == {"validation": "passed"}
    assert versions[0]["status"]


def test_get_registered_model_details_sorts_versions_numerically(
    mlflow_env,
):
    client, experiment_id = mlflow_env
    name = "multiple-version-model"

    for _ in range(12):
        create_registered_version(client, experiment_id, name)

    result = mlflow_tools.get_registered_model_details(name)

    assert result["success"] is True
    versions = result["model"]["versions"]
    assert [int(item["version"]) for item in versions] == list(range(1, 13))


def test_get_registered_model_details_paginates_versions(
    mlflow_env,
    monkeypatch,
):
    client, experiment_id = mlflow_env
    name = "paginated-registry-model"

    for _ in range(5):
        create_registered_version(client, experiment_id, name)

    monkeypatch.setattr(mlflow_tools, "MAX_RESULTS", 2)

    result = mlflow_tools.get_registered_model_details(name)

    assert result["success"] is True
    assert result["model"]["scanned_versions"] == 5
    assert result["model"]["scan_limit"] == mlflow_tools.MAX_SCAN_RESULTS
    assert result["model"]["truncated"] is False
    assert [
        int(item["version"])
        for item in result["model"]["versions"]
    ] == [1, 2, 3, 4, 5]


def test_get_registered_model_details_reports_scan_truncation(
    mlflow_env,
    monkeypatch,
):
    client, experiment_id = mlflow_env
    name = "truncated-registry-model"

    for _ in range(3):
        create_registered_version(client, experiment_id, name)

    monkeypatch.setattr(mlflow_tools, "MAX_RESULTS", 1)
    monkeypatch.setattr(mlflow_tools, "MAX_SCAN_RESULTS", 2)

    result = mlflow_tools.get_registered_model_details(name)

    assert result["success"] is True
    assert result["model"]["scanned_versions"] == 2
    assert result["model"]["scan_limit"] == 2
    assert result["model"]["truncated"] is True
    assert len(result["model"]["versions"]) == 2


def test_get_model_version_details_returns_run_status(mlflow_env):
    client, experiment_id = mlflow_env
    name = "version-details-model"
    run_id = create_run(client, experiment_id, 0.95)

    created = create_registered_version(
        client,
        experiment_id,
        name,
        run_id=run_id,
        tags={"environment": "test"},
        description="Completed source run.",
    )

    result = mlflow_tools.get_model_version_details(name, 1)

    assert result["success"] is True
    version = result["version"]
    assert version["name"] == name
    assert version["version"] == str(created.version)
    assert version["description"] == "Completed source run."
    assert version["creation_timestamp"] is not None
    assert version["last_updated_timestamp"] is not None
    assert version["run_id"] == run_id
    assert version["run_status"] == "FINISHED"
    assert version["source"] == f"runs:/{run_id}/model"
    assert version["tags"] == {"environment": "test"}
    assert version["status"]


def test_get_model_version_details_without_associated_run(mlflow_env):
    client, experiment_id = mlflow_env
    name = "version-without-run-model"

    create_registered_version(
        client,
        experiment_id,
        name,
        source="file:///tmp/test-model",
    )

    result = mlflow_tools.get_model_version_details(name, 1)

    assert result["success"] is True
    assert result["version"]["run_id"] is None
    assert result["version"]["run_status"] is None
    assert result["version"]["source"] == "file:///tmp/test-model"


@pytest.mark.parametrize(
    "name",
    [
        "",
        " ",
        "../model",
        "/tmp/model",
        "model/name",
        "model\\name",
        "x' or '1'='1",
        None,
        123,
        [],
    ],
)
def test_registry_details_reject_invalid_model_names(name):
    assert mlflow_tools.get_registered_model_details(name) == {
        "success": False,
        "error": "Invalid registered model name",
    }

    assert mlflow_tools.get_model_version_details(name, 1) == {
        "success": False,
        "error": "Invalid registered model name",
    }


@pytest.mark.parametrize(
    "version",
    [0, -1, True, False, "1", None, 1.5, [], {}],
)
def test_get_model_version_details_rejects_invalid_versions(version):
    result = mlflow_tools.get_model_version_details(
        "valid-model",
        version,
    )

    assert result == {
        "success": False,
        "error": "Invalid model version",
    }


def test_get_registered_model_details_reports_missing_model(mlflow_env):
    result = mlflow_tools.get_registered_model_details("missing-model")

    assert result["success"] is False
    assert result["error"]


def test_get_model_version_details_reports_missing_version(mlflow_env):
    client, _ = mlflow_env
    client.create_registered_model("model-without-version")

    result = mlflow_tools.get_model_version_details(
        "model-without-version",
        1,
    )

    assert result["success"] is False
    assert result["error"]


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("MLflow unavailable"),
        SQLAlchemyError("Database unavailable"),
    ],
)
def test_get_registered_model_details_handles_client_errors(
    monkeypatch,
    exception,
):
    def raise_error():
        raise exception

    monkeypatch.setattr(mlflow_tools, "_get_client", raise_error)

    result = mlflow_tools.get_registered_model_details("valid-model")

    assert result == {"success": False, "error": str(exception)}


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("MLflow unavailable"),
        SQLAlchemyError("Database unavailable"),
    ],
)
def test_get_model_version_details_handles_client_errors(
    monkeypatch,
    exception,
):
    def raise_error():
        raise exception

    monkeypatch.setattr(mlflow_tools, "_get_client", raise_error)

    result = mlflow_tools.get_model_version_details("valid-model", 1)

    assert result == {"success": False, "error": str(exception)}


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("Model lookup failed"),
        SQLAlchemyError("Model database lookup failed"),
    ],
)
def test_get_registered_model_details_handles_lookup_errors(
    mlflow_env,
    monkeypatch,
    exception,
):
    client, _ = mlflow_env

    def raise_error(name):
        raise exception

    monkeypatch.setattr(client, "get_registered_model", raise_error)

    result = mlflow_tools.get_registered_model_details("valid-model")

    assert result == {"success": False, "error": str(exception)}


@pytest.mark.parametrize(
    "exception",
    [
        MlflowException("Version lookup failed"),
        SQLAlchemyError("Version database lookup failed"),
    ],
)
def test_get_model_version_details_handles_lookup_errors(
    mlflow_env,
    monkeypatch,
    exception,
):
    client, _ = mlflow_env

    def raise_error(name, version):
        raise exception

    monkeypatch.setattr(client, "get_model_version", raise_error)

    result = mlflow_tools.get_model_version_details("valid-model", 1)

    assert result == {"success": False, "error": str(exception)}


def test_get_run_dataset_provenance_returns_all_tags(mlflow_env):
    client, experiment_id = mlflow_env
    run_id = create_run(client, experiment_id, 0.91)

    expected = {
        "dataset_path": "data/raw/training_data.csv",
        "dataset_git_revision": "abc123",
        "dataset_dvc_hash": "0123456789abcdef",
        "dataset_dvc_metadata_path": "data/raw/training_data.csv.dvc",
        "dataset_version": "md5:0123456789abcdef",
    }

    for key, value in expected.items():
        client.set_tag(run_id, key, value)

    result = mlflow_tools.get_run_dataset_provenance(run_id)

    assert result["success"] is True
    assert result["run_id"] == run_id
    assert result["provenance"] == expected
    assert result["complete"] is True
    assert result["missing_tags"] == []


def test_get_run_dataset_provenance_reports_missing_tags(mlflow_env):
    client, experiment_id = mlflow_env
    run_id = create_run(client, experiment_id, 0.91)
    client.set_tag(run_id, "dataset_path", "data/training.csv")

    result = mlflow_tools.get_run_dataset_provenance(run_id)

    assert result["success"] is True
    assert result["complete"] is False
    assert result["provenance"]["dataset_path"] == "data/training.csv"
    assert result["missing_tags"] == [
        "dataset_git_revision",
        "dataset_dvc_hash",
        "dataset_dvc_metadata_path",
        "dataset_version",
    ]


@pytest.mark.parametrize("run_id", ["", " ", None, 123])
def test_get_run_dataset_provenance_rejects_invalid_run_id(run_id):
    result = mlflow_tools.get_run_dataset_provenance(run_id)

    assert result == {
        "success": False,
        "error": "Invalid run_id",
    }


def test_get_run_dataset_provenance_handles_missing_run(mlflow_env):
    result = mlflow_tools.get_run_dataset_provenance("missing-run-id")

    assert result["success"] is False
    assert result["error"]


def test_artifact_tree_size_rejects_unsafe_nested_directory_path():
    class FakeArtifact:
        path = "../outside"
        is_dir = True
        file_size = None

    class FakeClient:
        def list_artifacts(self, run_id, path=None):
            if path is None:
                return [FakeArtifact()]
            pytest.fail("Unsafe nested path must not be queried")

    with pytest.raises(ValueError, match="Invalid artifact path"):
        mlflow_tools._artifact_tree_size(
            FakeClient(),
            "test-run",
        )


def test_artifact_tree_size_rejects_boolean_file_size():
    class FakeArtifact:
        path = "artifact.bin"
        is_dir = False
        file_size = True

    class FakeClient:
        def list_artifacts(self, run_id, path=None):
            return [FakeArtifact()]

    with pytest.raises(ValueError, match="Artifact size is unavailable"):
        mlflow_tools._artifact_tree_size(
            FakeClient(),
            "test-run",
        )


def test_finish_download_rejects_symlinked_file(tmp_path):
    cache = tmp_path / "cache"
    destination = cache / "current"
    destination.mkdir(parents=True)

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside the download directory")

    linked_file = destination / "linked.txt"
    linked_file.symlink_to(outside_file)

    with pytest.raises(ValueError, match="symbolic link"):
        mlflow_tools._finish_download(
            destination,
            destination,
        )


def test_finish_download_preserves_current_directory_during_cache_pruning(
    tmp_path,
    monkeypatch,
):
    cache = tmp_path / "cache"
    destination = cache / "current"
    destination.mkdir(parents=True)

    downloaded_file = destination / "artifact.txt"
    downloaded_file.write_text("current artifact")

    monkeypatch.setattr(mlflow_tools, "ARTIFACT_DOWNLOAD_CACHE", cache)
    monkeypatch.setattr(mlflow_tools, "MAX_CACHED_DOWNLOADS", 2)

    # Simulate two other cache entries appearing newer than this download.
    for name in ("newer-1", "newer-2"):
        directory = cache / name
        directory.mkdir()
        (directory / "old.txt").write_text("old")
        import os
        os.utime(directory, (2_000_000_000, 2_000_000_000))

    result = mlflow_tools._finish_download(
        destination,
        downloaded_file,
    )

    assert result["success"] is True
    assert downloaded_file.is_file()
    assert destination.exists()


def test_artifact_tree_size_supports_logged_models():
    class FakeArtifact:
        path = "MLmodel"
        is_dir = False
        file_size = 120

    class FakeClient:
        def list_logged_model_artifacts(self, model_id, path=None):
            assert model_id == "model-1"
            assert path is None
            return [FakeArtifact()]

    assert mlflow_tools._artifact_tree_size(
        FakeClient(),
        "model-1",
        logged_model=True,
    ) == 120


def test_artifact_tree_size_rejects_scan_limit(monkeypatch):
    class FakeArtifact:
        path = "artifact.bin"
        is_dir = False
        file_size = 1

    class FakeClient:
        def list_artifacts(self, run_id, path=None):
            return [FakeArtifact()]

    monkeypatch.setattr(mlflow_tools, "MAX_SCAN_RESULTS", 0)

    with pytest.raises(ValueError, match="scan limit"):
        mlflow_tools._artifact_tree_size(FakeClient(), "run-1")


def test_artifact_tree_size_rejects_total_size_limit(monkeypatch):
    class FakeArtifact:
        path = "artifact.bin"
        is_dir = False
        file_size = 2

    class FakeClient:
        def list_artifacts(self, run_id, path=None):
            return [FakeArtifact()]

    monkeypatch.setattr(mlflow_tools, "MAX_ARTIFACT_DOWNLOAD_BYTES", 1)

    with pytest.raises(ValueError, match="50 MB"):
        mlflow_tools._artifact_tree_size(FakeClient(), "run-1")


def test_prepare_download_directory_rejects_symlinked_cache(
    tmp_path,
    monkeypatch,
):
    real_cache = tmp_path / "real-cache"
    real_cache.mkdir()
    linked_cache = tmp_path / "linked-cache"
    linked_cache.symlink_to(real_cache, target_is_directory=True)

    monkeypatch.setattr(
        mlflow_tools,
        "ARTIFACT_DOWNLOAD_CACHE",
        linked_cache,
    )

    with pytest.raises(ValueError, match="symbolic link"):
        mlflow_tools._prepare_download_directory()


def test_finish_download_rejects_path_outside_destination(tmp_path):
    destination = tmp_path / "cache" / "download"
    destination.mkdir(parents=True)

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside")

    with pytest.raises(ValueError, match="path is invalid"):
        mlflow_tools._finish_download(destination, outside_file)


def test_finish_download_rejects_missing_path(tmp_path):
    destination = tmp_path / "cache" / "download"
    destination.mkdir(parents=True)

    with pytest.raises(ValueError, match="path is invalid"):
        mlflow_tools._finish_download(
            destination,
            destination / "missing.txt",
        )


def test_finish_download_rejects_oversized_file(tmp_path, monkeypatch):
    destination = tmp_path / "cache" / "download"
    destination.mkdir(parents=True)
    downloaded_file = destination / "artifact.bin"
    downloaded_file.write_bytes(b"too large")

    monkeypatch.setattr(
        mlflow_tools,
        "MAX_ARTIFACT_DOWNLOAD_BYTES",
        1,
    )

    with pytest.raises(ValueError, match="50 MB"):
        mlflow_tools._finish_download(destination, downloaded_file)


def test_list_logged_model_artifacts_returns_metadata(monkeypatch):
    class FakeArtifact:
        path = "MLmodel"
        is_dir = False
        file_size = 120

    class FakeClient:
        def get_logged_model(self, model_id):
            assert model_id == "model-1"

        def list_logged_model_artifacts(self, model_id, path=None):
            return [FakeArtifact()]

    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.list_logged_model_artifacts("model-1")

    assert result == {
        "success": True,
        "model_id": "model-1",
        "artifacts": [
            {
                "path": "MLmodel",
                "is_dir": False,
                "file_size": 120,
            }
        ],
    }


def test_list_logged_model_artifacts_rejects_invalid_inputs():
    assert mlflow_tools.list_logged_model_artifacts("") == {
        "success": False,
        "error": "Invalid model_id",
    }
    assert mlflow_tools.list_logged_model_artifacts(
        "model-1",
        path="../outside",
    ) == {
        "success": False,
        "error": "Invalid artifact path",
    }


def test_list_logged_model_artifacts_handles_tracking_error(monkeypatch):
    class FakeClient:
        def get_logged_model(self, model_id):
            raise MlflowException("Model unavailable")

    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.list_logged_model_artifacts("model-1")

    assert result["success"] is False
    assert "Model unavailable" in result["error"]


def test_download_run_artifact_rejects_invalid_inputs():
    assert mlflow_tools.download_run_artifact("", "artifact.bin") == {
        "success": False,
        "error": "Invalid run_id",
    }
    assert mlflow_tools.download_run_artifact(
        "run-1",
        "../outside",
    ) == {
        "success": False,
        "error": "Invalid artifact path",
    }


def test_download_run_artifact_cleans_up_after_download_error(
    tmp_path,
    monkeypatch,
):
    class FakeArtifact:
        path = "artifact.bin"
        is_dir = False
        file_size = 10

    class FakeClient:
        def list_artifacts(self, run_id, path=None):
            return [FakeArtifact()]

        def download_artifacts(self, run_id, path, dst_path):
            raise MlflowException("Download failed")

    cache = tmp_path / "cache"
    monkeypatch.setattr(
        mlflow_tools,
        "ARTIFACT_DOWNLOAD_CACHE",
        cache,
    )
    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.download_run_artifact(
        "run-1",
        "artifact.bin",
    )

    assert result["success"] is False
    assert "Download failed" in result["error"]
    assert list(cache.iterdir()) == []


def test_download_logged_model_artifact_rejects_invalid_model_id():
    assert mlflow_tools.download_logged_model_artifact("") == {
        "success": False,
        "error": "Invalid model_id",
    }


def test_download_logged_model_artifact_downloads_model(
    tmp_path,
    monkeypatch,
):
    import mlflow

    class FakeArtifact:
        path = "MLmodel"
        is_dir = False
        file_size = 7

    class FakeClient:
        tracking_uri = "sqlite:///temporary.db"

        def get_logged_model(self, model_id):
            assert model_id == "model-1"

        def list_logged_model_artifacts(self, model_id, path=None):
            return [FakeArtifact()]

    cache = tmp_path / "cache"
    monkeypatch.setattr(
        mlflow_tools,
        "ARTIFACT_DOWNLOAD_CACHE",
        cache,
    )
    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    def fake_download_artifacts(artifact_uri, dst_path, tracking_uri):
        assert artifact_uri == "models:/model-1"
        assert tracking_uri == "sqlite:///temporary.db"
        downloaded = Path(dst_path) / "MLmodel"
        downloaded.write_text("model!")
        return str(downloaded)

    monkeypatch.setattr(
        mlflow.artifacts,
        "download_artifacts",
        fake_download_artifacts,
    )

    result = mlflow_tools.download_logged_model_artifact("model-1")

    assert result["success"] is True
    assert result["size_bytes"] == 6
    assert Path(result["local_path"]).read_text() == "model!"


def test_download_logged_model_artifact_cleans_up_after_error(
    tmp_path,
    monkeypatch,
):
    class FakeClient:
        def get_logged_model(self, model_id):
            raise MlflowException("Model unavailable")

    cache = tmp_path / "cache"
    monkeypatch.setattr(
        mlflow_tools,
        "ARTIFACT_DOWNLOAD_CACHE",
        cache,
    )
    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.download_logged_model_artifact("model-1")

    assert result["success"] is False
    assert "Model unavailable" in result["error"]
    assert not cache.exists()


def test_artifact_tree_size_traverses_nested_directories():
    class FakeArtifact:
        def __init__(self, path, is_dir, file_size):
            self.path = path
            self.is_dir = is_dir
            self.file_size = file_size

    class FakeClient:
        def __init__(self):
            self.visited = []

        def list_artifacts(self, run_id, path=None):
            self.visited.append(path)

            if path is None:
                return [FakeArtifact("reports", True, None)]

            if path == "reports":
                return [FakeArtifact("reports/summary.txt", False, 12)]

            return []

    client = FakeClient()

    result = mlflow_tools._artifact_tree_size(client, "run-1")

    assert result == 12
    assert client.visited == [None, "reports"]


def test_finish_download_rejects_symlinked_download_root(tmp_path):
    destination = tmp_path / "destination"
    destination.mkdir()

    target = tmp_path / "target.txt"
    target.write_text("outside")

    link = destination / "linked.txt"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="symbolic link"):
        mlflow_tools._finish_download(destination, link)


def test_finish_download_calculates_nested_directory_size(tmp_path):
    destination = tmp_path / "destination"
    nested = destination / "reports" / "archive"
    nested.mkdir(parents=True)

    (destination / "reports" / "summary.txt").write_text("hello")
    (nested / "details.txt").write_text("world!")

    result = mlflow_tools._finish_download(
        destination,
        destination / "reports",
    )

    assert result["success"] is True
    assert result["size_bytes"] == 11
    assert Path(result["local_path"]).is_dir()


def test_download_run_artifact_downloads_single_file(
    tmp_path,
    monkeypatch,
):
    class FakeArtifact:
        path = "summary.txt"
        is_dir = False
        file_size = 13

    class FakeClient:
        def list_artifacts(self, run_id, path=None):
            return [FakeArtifact()]

        def download_artifacts(self, run_id, path, dst_path):
            downloaded = Path(dst_path) / path
            downloaded.write_text("hello, world!")
            return str(downloaded)

    cache = tmp_path / "cache"

    monkeypatch.setattr(
        mlflow_tools,
        "ARTIFACT_DOWNLOAD_CACHE",
        cache,
    )
    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    result = mlflow_tools.download_run_artifact(
        "run-1",
        "summary.txt",
    )

    assert result["success"] is True
    assert Path(result["local_path"]).read_text() == "hello, world!"
    assert result["size_bytes"] == 13
    assert Path(result["local_path"]).stat().st_size == 13


def test_download_logged_model_artifact_cleans_up_after_download_error(
    tmp_path,
    monkeypatch,
):
    import mlflow

    class FakeArtifact:
        path = "MLmodel"
        is_dir = False
        file_size = 10

    class FakeClient:
        tracking_uri = "sqlite:///temporary.db"

        def get_logged_model(self, model_id):
            assert model_id == "model-1"

        def list_logged_model_artifacts(self, model_id, path=None):
            return [FakeArtifact()]

    cache = tmp_path / "cache"

    monkeypatch.setattr(
        mlflow_tools,
        "ARTIFACT_DOWNLOAD_CACHE",
        cache,
    )
    monkeypatch.setattr(
        mlflow_tools,
        "_get_client",
        lambda: FakeClient(),
    )

    def fake_download_artifacts(artifact_uri, dst_path, tracking_uri):
        partial = Path(dst_path) / "partial.txt"
        partial.write_text("partial")
        raise MlflowException("Download failed")

    monkeypatch.setattr(
        mlflow.artifacts,
        "download_artifacts",
        fake_download_artifacts,
    )

    result = mlflow_tools.download_logged_model_artifact("model-1")

    assert result["success"] is False
    assert "Download failed" in result["error"]
    assert cache.exists()
    assert list(cache.iterdir()) == []
