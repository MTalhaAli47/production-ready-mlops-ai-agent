"""Tests for deterministic MLOps orchestration."""

import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import requests

from agent import orchestrator
from agent.orchestrator import (
    INFERENCE_API_TIMEOUT_SECONDS,
    check_inference_api,
    select_best_recent_model,
)
from agent.schemas import ToolResult


def make_response(status_code, body):
    response = Mock()
    response.status_code = status_code
    response.json.return_value = body
    return response


def test_missing_inference_url_returns_unavailable(monkeypatch):
    monkeypatch.delenv("INFERENCE_API_URL", raising=False)

    result = check_inference_api()

    assert result.status == "unavailable"
    assert result.error == "INFERENCE_API_URL is not configured"
    assert result.checked_at is not None


def test_invalid_inference_url_returns_failed(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_URL", "ftp://localhost:8000")

    result = check_inference_api()

    assert result.status == "failed"
    assert result.error == "Invalid INFERENCE_API_URL"


def test_both_inference_endpoints_healthy(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_URL", "http://localhost:18000/")

    mock_get = Mock(
        side_effect=[
            make_response(200, {"status": "alive"}),
            make_response(200, {"status": "ready"}),
        ]
    )
    monkeypatch.setattr(requests, "get", mock_get)

    result = check_inference_api()

    assert result.status == "success"
    assert result.data["endpoints"]["health"]["success"] is True
    assert result.data["endpoints"]["ready"]["success"] is True
    assert result.error is None
    assert mock_get.call_count == 2

    for call in mock_get.call_args_list:
        assert call.kwargs["timeout"] == INFERENCE_API_TIMEOUT_SECONDS
        assert call.kwargs["allow_redirects"] is False


def test_one_healthy_endpoint_returns_partial(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_URL", "http://localhost:18000")

    monkeypatch.setattr(
        requests,
        "get",
        Mock(side_effect=[
            make_response(200, {"status": "alive"}),
            make_response(503, {"detail": "Model unavailable"}),
        ]),
    )

    result = check_inference_api()

    assert result.status == "partial"
    assert result.data["endpoints"]["health"]["success"] is True
    assert result.data["endpoints"]["ready"]["success"] is False
    assert "ready:" in result.error


def test_both_endpoints_failing_returns_failed(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_URL", "http://localhost:18000")

    monkeypatch.setattr(
        requests,
        "get",
        Mock(side_effect=[
            make_response(503, {"detail": "Unavailable"}),
            make_response(503, {"detail": "Unavailable"}),
        ]),
    )

    result = check_inference_api()

    assert result.status == "failed"
    assert result.data["endpoints"]["health"]["success"] is False
    assert result.data["endpoints"]["ready"]["success"] is False
    assert result.error is not None


def test_connection_errors_are_captured_independently(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_URL", "http://localhost:18000")

    monkeypatch.setattr(
        requests,
        "get",
        Mock(side_effect=requests.ConnectionError("Connection refused")),
    )

    result = check_inference_api()

    assert result.status == "failed"
    assert "Connection refused" in result.error
    assert result.data["endpoints"]["health"]["success"] is False
    assert result.data["endpoints"]["ready"]["success"] is False


def test_non_json_response_is_reported_as_failure(monkeypatch):
    monkeypatch.setenv("INFERENCE_API_URL", "http://localhost:18000")

    response = make_response(200, None)
    response.json.side_effect = ValueError("Invalid JSON")

    monkeypatch.setattr(
        requests,
        "get",
        Mock(side_effect=[response, make_response(200, {"status": "ready"})]),
    )

    result = check_inference_api()

    assert result.status == "partial"
    assert result.data["endpoints"]["health"]["response"] is None
    assert result.data["endpoints"]["health"]["success"] is False
    assert result.data["endpoints"]["ready"]["success"] is True



def make_run(run_id, f1, start_time, status="FINISHED"):
    return {
        "run_id": run_id,
        "experiment_id": "experiment-1",
        "status": status,
        "start_time": start_time,
        "metrics": {"f1": f1} if f1 is not None else {},
    }


def fixed_reference_time():
    return datetime(2026, 11, 15, 1, 0, tzinfo=UTC)


def patch_run_listing(monkeypatch, response):
    mock_listing = Mock(return_value=response)
    monkeypatch.setattr("agent.orchestrator.list_runs", mock_listing)
    return mock_listing


def test_select_best_recent_model_uses_highest_f1(monkeypatch):
    now = fixed_reference_time()
    recent = int((now - timedelta(days=2)).timestamp() * 1000)

    mock_listing = patch_run_listing(
        monkeypatch,
        {
            "success": True,
            "runs": [
                make_run("run-low", 0.81, recent),
                make_run("run-high", 0.95, recent),
                make_run("run-mid", 0.89, recent),
            ],
        },
    )

    result = select_best_recent_model(now=now)

    assert result.status == "success"
    assert result.data["selected_run"]["run_id"] == "run-high"
    assert result.data["selected_run"]["f1"] == 0.95
    mock_listing.assert_called_once_with(
        experiment_name="student-pass-prediction",
        max_results=100,
    )


def test_select_best_recent_model_uses_newest_run_for_f1_tie(monkeypatch):
    now = fixed_reference_time()
    older = int((now - timedelta(days=5)).timestamp() * 1000)
    newer = int((now - timedelta(days=1)).timestamp() * 1000)

    patch_run_listing(
        monkeypatch,
        {
            "success": True,
            "runs": [
                make_run("older-run", 0.95, older),
                make_run("newer-run", 0.95, newer),
            ],
        },
    )

    result = select_best_recent_model(now=now)

    assert result.status == "success"
    assert result.data["selected_run"]["run_id"] == "newer-run"


def test_select_best_recent_model_uses_run_id_for_complete_tie(monkeypatch):
    now = fixed_reference_time()
    timestamp = int((now - timedelta(days=1)).timestamp() * 1000)

    patch_run_listing(
        monkeypatch,
        {
            "success": True,
            "runs": [
                make_run("run-z", 0.95, timestamp),
                make_run("run-a", 0.95, timestamp),
                make_run("run-m", 0.95, timestamp),
            ],
        },
    )

    result = select_best_recent_model(now=now)

    assert result.status == "success"
    assert result.data["selected_run"]["run_id"] == "run-a"


def test_select_best_recent_model_excludes_ineligible_runs(monkeypatch):
    now = fixed_reference_time()
    recent = int((now - timedelta(days=2)).timestamp() * 1000)
    old = int((now - timedelta(days=31)).timestamp() * 1000)
    future = int((now + timedelta(days=1)).timestamp() * 1000)

    runs = [
        make_run("valid", 0.91, recent),
        make_run("running", 0.99, recent, status="RUNNING"),
        make_run("old", 0.98, old),
        make_run("future", 0.97, future),
        make_run("missing-f1", None, recent),
        make_run("string-f1", "0.99", recent),
        make_run("boolean-f1", True, recent),
        make_run("out-of-range-f1", 1.2, recent),
        None,
    ]

    patch_run_listing(monkeypatch, {"success": True, "runs": runs})

    result = select_best_recent_model(now=now)

    assert result.status == "success"
    assert result.data["selected_run"]["run_id"] == "valid"
    assert result.data["excluded_runs"] == {
        "invalid_run": 1,
        "not_finished": 1,
        "outside_recency_window": 2,
        "missing_or_invalid_f1": 4,
    }


def test_select_best_recent_model_returns_unavailable_without_fallback(
    monkeypatch,
):
    now = fixed_reference_time()
    old = int((now - timedelta(days=45)).timestamp() * 1000)

    patch_run_listing(
        monkeypatch,
        {
            "success": True,
            "runs": [make_run("old-run", 0.99, old)],
        },
    )

    result = select_best_recent_model(now=now)

    assert result.status == "unavailable"
    assert result.data["selected_run"] is None
    assert result.data["eligible_runs"] == 0
    assert "No qualifying FINISHED run" in result.error


def test_select_best_recent_model_handles_mlflow_listing_failure(monkeypatch):
    patch_run_listing(
        monkeypatch,
        {"success": False, "error": "MLflow database unavailable"},
    )

    result = select_best_recent_model(now=fixed_reference_time())

    assert result.status == "failed"
    assert result.error == "MLflow database unavailable"


def test_select_best_recent_model_rejects_invalid_listing_responses(
    monkeypatch,
):
    for response in (None, [], {"success": True, "runs": None}):
        patch_run_listing(monkeypatch, response)

        result = select_best_recent_model(now=fixed_reference_time())

        assert result.status == "failed"


def test_select_best_recent_model_rejects_naive_reference_time():
    naive_now = datetime(2026, 11, 15, 1, 0)

    result = select_best_recent_model(now=naive_now)

    assert result.status == "failed"
    assert result.error == "Reference time must be timezone-aware"


def test_select_best_recent_model_reports_sample_limitation(monkeypatch):
    now = fixed_reference_time()
    recent = int((now - timedelta(days=2)).timestamp() * 1000)
    runs = [
        make_run(f"run-{index:03}", 0.8, recent)
        for index in range(100)
    ]

    mock_listing = patch_run_listing(
        monkeypatch,
        {"success": True, "runs": runs},
    )

    result = select_best_recent_model(now=now)

    assert result.status == "success"
    assert result.data["runs_examined"] == 100
    assert result.data["max_runs_examined"] == 100
    assert "limited to the runs returned by MLflow" in (
        result.data["sample_limitation"]
    )
    mock_listing.assert_called_once()


def test_select_best_recent_model_handles_unexpected_run_data(monkeypatch):
    now = fixed_reference_time()
    recent = int((now - timedelta(days=2)).timestamp() * 1000)

    runs = [
        "not-a-run-dictionary",
        make_run("invalid-time", 0.9, "not-a-timestamp"),
        make_run("invalid-metrics", 0.9, recent),
    ]
    runs[-1]["metrics"] = ["unexpected", "metrics"]

    patch_run_listing(monkeypatch, {"success": True, "runs": runs})

    result = select_best_recent_model(now=now)

    assert result.status == "unavailable"
    assert result.data["eligible_runs"] == 0
    assert result.data["excluded_runs"]["invalid_run"] == 1
    assert result.data["excluded_runs"]["outside_recency_window"] == 1
    assert result.data["excluded_runs"]["missing_or_invalid_f1"] == 1


def test_run_diagnostics_all_checks_succeed(monkeypatch, caplog):
    checked_at = orchestrator.datetime(2026, 11, 15, tzinfo=orchestrator.UTC)
    calls = []

    model_result = ToolResult(
        status="success",
        data={
            "selected_run": {
                "run_id": "run-123",
                "metrics": {"f1": 0.93},
            }
        },
        checked_at=checked_at,
    )

    def mock_tool(name, response):
        def execute(**kwargs):
            calls.append((name, kwargs))
            return response

        return execute

    monkeypatch.setattr(
        orchestrator,
        "select_best_recent_model",
        lambda now=None: model_result,
    )

    monkeypatch.setattr(
        orchestrator,
        "get_run_dataset_provenance",
        mock_tool(
            "provenance",
            {
                "success": True,
                "run_id": "run-123",
                "complete": True,
                "provenance": {
                    "dataset_path": "data/raw/training_data.csv",
                    "dataset_version": "md5:abc123",
                },
            },
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "check_dataset_changes",
        mock_tool(
            "dvc",
            {
                "success": True,
                "local_status": {
                    "success": True,
                    "output": "Data and pipelines are up to date.",
                },
                "remote_status": {
                    "success": True,
                    "output": "Cache and remote 'localstorage' are in sync.",
                },
            },
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "list_dag_runs",
        mock_tool(
            "airflow",
            {
                "success": True,
                "dag_runs": [{"state": "success"}],
                "total_entries": 1,
            },
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "get_deployment_details",
        mock_tool(
            "deployment",
            {
                "success": True,
                "data": {
                    "desired_replicas": 2,
                    "available_replicas": 2,
                },
            },
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "list_pods",
        mock_tool(
            "pods",
            {
                "success": True,
                "count": 2,
                "data": [
                    {"name": "inference-pod-1", "ready": True},
                    {"name": "inference-pod-2", "ready": True},
                ],
                "continue_token": None,
            },
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "check_inference_api",
        lambda: ToolResult(
            status="success",
            data={"health": "alive", "readiness": "ready"},
            checked_at=checked_at,
        ),
    )

    caplog.set_level(logging.INFO, logger="agent.orchestrator")
    report = orchestrator.run_diagnostics(now=checked_at)

    log_records = [
        record
        for record in caplog.records
        if record.getMessage() == "Diagnostic subsystem completed"
    ]
    assert len(log_records) == 6
    request_ids = {record.request_id for record in log_records}
    assert len(request_ids) == 1
    assert next(iter(request_ids))
    assert {record.tool_name for record in log_records} == {
        "model_selection",
        "training_provenance",
        "dataset_state",
        "airflow",
        "kubernetes",
        "inference_api",
    }
    results = {
        "model_selection": report.model_selection,
        "training_provenance": report.training_provenance,
        "dataset_state": report.dataset_state,
        "airflow": report.airflow,
        "kubernetes": report.kubernetes,
        "inference_api": report.inference_api,
    }
    for record in log_records:
        result = results[record.tool_name]
        assert record.status == result.status
        assert record.duration_ms == result.duration_ms

    assert report.overall_status == "complete"
    assert report.model_selection.status == "success"
    assert report.training_provenance.status == "success"
    assert report.training_provenance.data["provenance"]["dataset_version"] == (
        "md5:abc123"
    )
    assert report.dataset_state.status == "success"
    assert report.airflow.status == "success"
    assert report.kubernetes.status == "success"
    assert report.kubernetes.data["ready_pod_count"] == 2
    assert report.inference_api.status == "success"

    assert calls == [
        ("provenance", {"run_id": "run-123"}),
        ("dvc", {}),
        ("airflow", {"dag_id": "mlops_a03_healthcheck", "limit": 10}),
        ("deployment", {"name": "inference-api", "namespace": "mlops-a03"}),
        ("pods", {"namespace": "mlops-a03", "limit": 50}),
    ]


def test_run_diagnostics_continues_when_no_model_is_selected(monkeypatch):
    checked_at = orchestrator.datetime(2026, 11, 15, tzinfo=orchestrator.UTC)
    calls = []

    monkeypatch.setattr(
        orchestrator,
        "select_best_recent_model",
        lambda now=None: ToolResult(
            status="unavailable",
            error="No qualifying recent model",
            checked_at=checked_at,
        ),
    )

    def provenance_must_be_skipped(**kwargs):
        raise AssertionError("Provenance must be skipped without a selected run")

    monkeypatch.setattr(
        orchestrator,
        "get_run_dataset_provenance",
        provenance_must_be_skipped,
    )

    def successful_tool(name, response):
        def execute(**kwargs):
            calls.append(name)
            return response

        return execute

    monkeypatch.setattr(
        orchestrator,
        "check_dataset_changes",
        successful_tool(
            "dvc",
            {
                "success": True,
                "local_status": {
                    "success": True,
                    "output": "Data and pipelines are up to date.",
                },
                "remote_status": {
                    "success": True,
                    "output": "Cache and remote 'localstorage' are in sync.",
                },
            },
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "list_dag_runs",
        successful_tool(
            "airflow",
            {"success": True, "dag_runs": [], "total_entries": 0},
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "get_deployment_details",
        successful_tool(
            "deployment",
            {
                "success": True,
                "data": {"desired_replicas": 2, "available_replicas": 2},
            },
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "list_pods",
        successful_tool(
            "pods",
            {
                "success": True,
                "count": 2,
                "data": [{"ready": True}, {"ready": True}],
                "continue_token": None,
            },
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "check_inference_api",
        lambda: ToolResult(
            status="success",
            data={"health": "alive", "readiness": "ready"},
            checked_at=checked_at,
        ),
    )

    report = orchestrator.run_diagnostics(now=checked_at)

    assert report.overall_status == "partial"
    assert report.model_selection.status == "unavailable"
    assert report.training_provenance.status == "skipped"
    assert report.dataset_state.status == "success"
    assert report.airflow.status == "unavailable"
    assert calls == ["dvc", "airflow", "deployment", "pods"]

    monkeypatch.setattr(
        orchestrator,
        "list_dag_runs",
        lambda **kwargs: {
            "success": True,
            "dag_runs": [{"state": "failed"}],
            "total_entries": 1,
        },
    )
    failed_report = orchestrator.run_diagnostics()
    assert failed_report.airflow.status == "partial"
    assert any(
        finding.source == "airflow"
        for finding in failed_report.findings
    )
    assert report.kubernetes.status == "success"
    assert report.inference_api.status == "success"

def test_run_diagnostics_warns_about_incomplete_provenance(monkeypatch):
    checked_at = orchestrator.datetime(2026, 11, 15, tzinfo=orchestrator.UTC)

    monkeypatch.setattr(
        orchestrator,
        "select_best_recent_model",
        lambda now=None: ToolResult(
            status="success",
            data={
                "selected_run": {
                    "run_id": "run-incomplete",
                    "metrics": {"f1": 0.93},
                }
            },
            checked_at=checked_at,
        ),
    )

    monkeypatch.setattr(
        orchestrator,
        "get_run_dataset_provenance",
        lambda **kwargs: {
            "success": True,
            "run_id": "run-incomplete",
            "complete": False,
            "missing_tags": ["dataset_dvc_hash"],
            "provenance": {
                "dataset_path": "data/raw/training_data.csv",
                "dataset_version": None,
            },
        },
    )

    monkeypatch.setattr(
        orchestrator,
        "check_dataset_changes",
        lambda: {
            "success": True,
            "local_status": "up to date",
            "remote_status": "up to date",
        },
    )

    monkeypatch.setattr(
        orchestrator,
        "list_dag_runs",
        lambda **kwargs: {
            "success": True,
            "dag_runs": [{"state": "success"}],
            "total_entries": 1,
        },
    )

    monkeypatch.setattr(
        orchestrator,
        "get_deployment_details",
        lambda **kwargs: {
            "success": True,
            "data": {
                "desired_replicas": 2,
                "available_replicas": 2,
            },
        },
    )

    monkeypatch.setattr(
        orchestrator,
        "list_pods",
        lambda **kwargs: {
            "success": True,
            "count": 2,
            "data": [{"ready": True}, {"ready": True}],
            "continue_token": None,
        },
    )

    monkeypatch.setattr(
        orchestrator,
        "check_inference_api",
        lambda: ToolResult(
            status="success",
            data={"health": "alive", "readiness": "ready"},
            checked_at=checked_at,
        ),
    )

    report = orchestrator.run_diagnostics(now=checked_at)

    assert report.training_provenance.status == "success"
    assert report.training_provenance.data["complete"] is False
    assert report.training_provenance.data["missing_tags"] == [
        "dataset_dvc_hash"
    ]

    provenance_warnings = [
        finding
        for finding in report.findings
        if finding.source == "training_provenance"
        and finding.severity == "warning"
    ]

    assert len(provenance_warnings) == 1

    assert any(
        "missing training provenance" in recommendation.lower()
        for recommendation in report.recommendations
    )


def test_run_diagnostics_warns_when_dvc_reports_local_changes(monkeypatch):
    checked_at = orchestrator.datetime(2026, 11, 15, tzinfo=orchestrator.UTC)

    monkeypatch.setattr(
        orchestrator,
        "select_best_recent_model",
        lambda now=None: ToolResult(
            status="unavailable",
            error="No qualifying recent model",
            checked_at=checked_at,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "check_dataset_changes",
        lambda: {
            "success": True,
            "local_status": {
                "success": True,
                "output": "data/raw/training_data.csv.dvc changed",
            },
            "remote_status": {
                "success": True,
                "output": "Cache and remote 'localstorage' are in sync.",
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "list_dag_runs",
        lambda **kwargs: {
            "success": True,
            "dag_runs": [{"state": "success"}],
            "total_entries": 1,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "get_deployment_details",
        lambda **kwargs: {
            "success": True,
            "data": {"desired_replicas": 2, "available_replicas": 2},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "list_pods",
        lambda **kwargs: {
            "success": True,
            "count": 2,
            "data": [{"ready": True}, {"ready": True}],
            "continue_token": None,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "check_inference_api",
        lambda: ToolResult(
            status="success",
            data={"health": "alive", "readiness": "ready"},
            checked_at=checked_at,
        ),
    )

    report = orchestrator.run_diagnostics(now=checked_at)

    assert report.dataset_state.status == "partial"
    assert any(
        finding.source == "dataset_state" and finding.severity == "warning"
        for finding in report.findings
    )
    assert any(
        "dataset" in recommendation.lower()
        and ("change" in recommendation.lower() or "dvc" in recommendation.lower())
        for recommendation in report.recommendations
    )


def test_run_diagnostics_accepts_clean_dvc_state(monkeypatch):
    checked_at = orchestrator.datetime(2026, 11, 15, tzinfo=orchestrator.UTC)

    monkeypatch.setattr(
        orchestrator,
        "select_best_recent_model",
        lambda now=None: ToolResult(
            status="unavailable",
            error="No qualifying recent model",
            checked_at=checked_at,
        ),
    )
    monkeypatch.setattr(
        orchestrator,
        "check_dataset_changes",
        lambda: {
            "success": True,
            "local_status": {
                "success": True,
                "output": "Data and pipelines are up to date.",
            },
            "remote_status": {
                "success": True,
                "output": "Cache and remote 'localstorage' are in sync.",
            },
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "list_dag_runs",
        lambda **kwargs: {
            "success": True,
            "dag_runs": [{"state": "success"}],
            "total_entries": 1,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "get_deployment_details",
        lambda **kwargs: {
            "success": True,
            "data": {"desired_replicas": 2, "available_replicas": 2},
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "list_pods",
        lambda **kwargs: {
            "success": True,
            "count": 2,
            "data": [{"ready": True}, {"ready": True}],
            "continue_token": None,
        },
    )
    monkeypatch.setattr(
        orchestrator,
        "check_inference_api",
        lambda: ToolResult(
            status="success",
            data={"health": "alive", "readiness": "ready"},
            checked_at=checked_at,
        ),
    )

    report = orchestrator.run_diagnostics(now=checked_at)

    assert report.dataset_state.status == "success"
    assert not any(
        finding.source == "dataset_state" and finding.severity == "warning"
        for finding in report.findings
    )
