"""Deterministic orchestration for cross-system MLOps diagnostics."""

import math
import os
import time
from datetime import UTC, datetime
from urllib.parse import urlsplit

import requests

from agent.schemas import DiagnosticFinding, DiagnosticReport, ToolResult
from agent.tools.airflow_tools import list_dag_runs
from agent.tools.dvc_tools import check_dataset_changes
from agent.tools.kubernetes_tools import (
    DEFAULT_NAMESPACE,
    get_deployment_details,
    list_pods,
)
from agent.tools.mlflow_tools import (
    get_run_dataset_provenance,
    list_runs,
)

INFERENCE_API_TIMEOUT_SECONDS = 3
HEALTHCHECK_DAG_ID = "mlops_a03_healthcheck"
INFERENCE_DEPLOYMENT_NAME = "inference-api"


def check_inference_api() -> ToolResult:
    """Check inference API liveness and model readiness independently."""
    checked_at = datetime.now(UTC)
    base_url = os.getenv("INFERENCE_API_URL", "").strip()

    if not base_url:
        return ToolResult(
            status="unavailable",
            error="INFERENCE_API_URL is not configured",
            checked_at=checked_at,
        )

    try:
        parsed_url = urlsplit(base_url)

        if (
            parsed_url.scheme not in {"http", "https"}
            or not parsed_url.netloc
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise ValueError("Invalid INFERENCE_API_URL")

        base_url = base_url.rstrip("/")
        endpoint_results = {}

        for endpoint, expected_status in (
            ("health", "alive"),
            ("ready", "ready"),
        ):
            started = time.perf_counter()

            try:
                response = requests.get(
                    f"{base_url}/{endpoint}",
                    timeout=INFERENCE_API_TIMEOUT_SECONDS,
                    allow_redirects=False,
                )
                duration_ms = (time.perf_counter() - started) * 1000

                try:
                    response_data = response.json()
                except ValueError:
                    response_data = None

                endpoint_ok = (
                    response.status_code == 200
                    and isinstance(response_data, dict)
                    and response_data.get("status") == expected_status
                )

                endpoint_results[endpoint] = {
                    "success": endpoint_ok,
                    "http_status": response.status_code,
                    "response": response_data,
                    "duration_ms": round(duration_ms, 2),
                }

                if not endpoint_ok:
                    endpoint_results[endpoint]["error"] = (
                        "Unexpected HTTP status or response body"
                    )

            except requests.RequestException as exc:
                endpoint_results[endpoint] = {
                    "success": False,
                    "error": str(exc),
                    "duration_ms": round(
                        (time.perf_counter() - started) * 1000,
                        2,
                    ),
                }

        successful_checks = sum(
            result["success"] for result in endpoint_results.values()
        )

        if successful_checks == 2:
            status = "success"
        elif successful_checks == 1:
            status = "partial"
        else:
            status = "failed"

        errors = [
            f"{endpoint}: {result['error']}"
            for endpoint, result in endpoint_results.items()
            if "error" in result
        ]

        return ToolResult(
            status=status,
            data={
                "base_url": base_url,
                "endpoints": endpoint_results,
            },
            error="; ".join(errors) if errors else None,
            checked_at=checked_at,
            duration_ms=round(
                sum(
                    result["duration_ms"]
                    for result in endpoint_results.values()
                ),
                2,
            ),
        )

    except ValueError as exc:
        return ToolResult(
            status="failed",
            error=str(exc),
            checked_at=checked_at,
        )


MODEL_SELECTION_EXPERIMENT = "student-pass-prediction"
MODEL_SELECTION_MAX_RUNS = 100
MODEL_SELECTION_RECENCY_DAYS = 30


def select_best_recent_model(
    now: datetime | None = None,
) -> ToolResult:
    """Select the highest-F1 finished run within the recent MLflow sample."""
    started = time.perf_counter()
    checked_at = datetime.now(UTC)
    reference_time = now or checked_at

    if reference_time.tzinfo is None or reference_time.utcoffset() is None:
        return ToolResult(
            status="failed",
            error="Reference time must be timezone-aware",
            checked_at=checked_at,
        )

    cutoff_ms = int(
        (
            reference_time.timestamp()
            - MODEL_SELECTION_RECENCY_DAYS * 24 * 60 * 60
        )
        * 1000
    )
    reference_ms = int(reference_time.timestamp() * 1000)

    listing = list_runs(
        experiment_name=MODEL_SELECTION_EXPERIMENT,
        max_results=MODEL_SELECTION_MAX_RUNS,
    )

    if not isinstance(listing, dict) or listing.get("success") is not True:
        error = (
            listing.get("error", "MLflow run listing failed")
            if isinstance(listing, dict)
            else "MLflow returned an invalid run-list response"
        )
        return ToolResult(
            status="failed",
            error=str(error),
            checked_at=checked_at,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    runs = listing.get("runs")
    if not isinstance(runs, list):
        return ToolResult(
            status="failed",
            error="MLflow run-list response has no valid runs list",
            checked_at=checked_at,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    candidates = []
    excluded = {
        "invalid_run": 0,
        "not_finished": 0,
        "outside_recency_window": 0,
        "missing_or_invalid_f1": 0,
    }

    for run in runs:
        if not isinstance(run, dict):
            excluded["invalid_run"] += 1
            continue

        if run.get("status") != "FINISHED":
            excluded["not_finished"] += 1
            continue

        start_time = run.get("start_time")
        if (
            isinstance(start_time, bool)
            or not isinstance(start_time, (int, float))
            or not math.isfinite(start_time)
            or not cutoff_ms <= start_time <= reference_ms
        ):
            excluded["outside_recency_window"] += 1
            continue

        metrics = run.get("metrics")
        f1_score = metrics.get("f1") if isinstance(metrics, dict) else None

        if (
            isinstance(f1_score, bool)
            or not isinstance(f1_score, (int, float))
            or not math.isfinite(f1_score)
            or not 0 <= f1_score <= 1
            or not isinstance(run.get("run_id"), str)
            or not run["run_id"]
        ):
            excluded["missing_or_invalid_f1"] += 1
            continue

        candidates.append(
            {
                "run_id": run["run_id"],
                "experiment_id": run.get("experiment_id"),
                "status": run["status"],
                "start_time": int(start_time),
                "f1": float(f1_score),
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["f1"],
            -candidate["start_time"],
            candidate["run_id"],
        )
    )

    selection_metadata = {
        "experiment_name": MODEL_SELECTION_EXPERIMENT,
        "metric": "f1",
        "recency_days": MODEL_SELECTION_RECENCY_DAYS,
        "max_runs_examined": MODEL_SELECTION_MAX_RUNS,
        "runs_examined": len(runs),
        "eligible_runs": len(candidates),
        "excluded_runs": excluded,
        "sample_limitation": (
            "Selection is limited to the runs returned by MLflow; "
            "the existing list_runs interface does not guarantee that "
            "the sample contains every recent run."
        ),
    }

    duration_ms = round((time.perf_counter() - started) * 1000, 2)

    if not candidates:
        return ToolResult(
            status="unavailable",
            data={
                **selection_metadata,
                "selected_run": None,
            },
            error="No qualifying FINISHED run with a valid F1 score "
            "was found in the returned sample within the last 30 days",
            checked_at=checked_at,
            duration_ms=duration_ms,
        )

    return ToolResult(
        status="success",
        data={
            **selection_metadata,
            "selected_run": candidates[0],
        },
        checked_at=checked_at,
        duration_ms=duration_ms,
    )



def _tool_response_result(response, source: str) -> ToolResult:
    """Convert a structured tool response into a validated ToolResult."""
    checked_at = datetime.now(UTC)

    if not isinstance(response, dict) or not isinstance(
        response.get("success"), bool
    ):
        return ToolResult(
            status="failed",
            error=f"{source} returned a malformed response",
            checked_at=checked_at,
        )

    if not response["success"]:
        error = response.get("error")
        return ToolResult(
            status="failed",
            data=response,
            error=error if isinstance(error, str) else f"{source} operation failed",
            checked_at=checked_at,
        )

    return ToolResult(
        status="success",
        data=response,
        checked_at=checked_at,
    )


def _run_read_only_tool(function, source: str, **kwargs) -> ToolResult:
    """Run one read-only tool without hiding failures from other checks."""
    started = time.perf_counter()

    try:
        response = function(**kwargs)
        result = _tool_response_result(response, source)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        result = ToolResult(
            status="unavailable",
            error=f"{source} check could not complete: {type(exc).__name__}",
            checked_at=datetime.now(UTC),
        )

    result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
    return result


def run_diagnostics(now: datetime | None = None) -> DiagnosticReport:
    """Collect read-only diagnostics across the MLOps platform."""
    generated_at = datetime.now(UTC)
    findings: list[DiagnosticFinding] = []
    recommendations: list[str] = []

    # 1. Select the best qualifying recent MLflow run.
    model_selection = select_best_recent_model(now=now)

    selected_run = None
    if isinstance(model_selection.data, dict):
        selected_run = model_selection.data.get("selected_run")

    run_id = selected_run.get("run_id") if isinstance(selected_run, dict) else None

    # 2. Retrieve the selected run's historical dataset provenance.
    if model_selection.status == "success" and isinstance(run_id, str) and run_id:
        training_provenance = _run_read_only_tool(
            get_run_dataset_provenance,
            "MLflow provenance",
            run_id=run_id,
        )
    else:
        training_provenance = ToolResult(
            status="skipped",
            error="Skipped because no qualifying model run was selected",
            checked_at=datetime.now(UTC),
        )

    # 3. Check current DVC state independently of model selection.
    dataset_state = _run_read_only_tool(
        check_dataset_changes,
        "DVC dataset state",
    )

    if dataset_state.status == "success":
        dvc_data = dataset_state.data
        local_status = (
            dvc_data.get("local_status") if isinstance(dvc_data, dict) else None
        )
        remote_status = (
            dvc_data.get("remote_status") if isinstance(dvc_data, dict) else None
        )

        local_output = (
            local_status.get("output")
            if isinstance(local_status, dict) and local_status.get("success") is True
            else None
        )
        remote_output = (
            remote_status.get("output")
            if isinstance(remote_status, dict) and remote_status.get("success") is True
            else None
        )

        local_clean = (
            isinstance(local_output, str)
            and local_output.strip() == "Data and pipelines are up to date."
        )
        remote_clean = (
            isinstance(remote_output, str)
            and remote_output.strip().startswith("Cache and remote '")
            and remote_output.strip().endswith("' are in sync.")
        )

        if not local_clean or not remote_clean:
            dataset_state = ToolResult(
                status="partial",
                data=dataset_state.data,
                error=(
                    "DVC checks executed successfully, but the dataset or "
                    "remote-cache state is not confirmed clean."
                ),
                checked_at=dataset_state.checked_at,
            )

    # 4. Inspect Airflow DAG history; do not trigger a DAG.
    airflow = _run_read_only_tool(
        list_dag_runs,
        "Airflow DAG history",
        dag_id=HEALTHCHECK_DAG_ID,
        limit=10,
    )
    if airflow.status == "success":
        airflow_data = airflow.data
        dag_runs = (
            airflow_data.get("dag_runs")
            if isinstance(airflow_data, dict)
            else None
        )
        airflow_error = None
        airflow_status = "partial"

        if not isinstance(dag_runs, list):
            airflow_error = "Airflow returned an invalid DAG-run history."
        elif not dag_runs:
            airflow_status = "unavailable"
            airflow_error = "No Airflow DAG runs are available as health evidence."
        else:
            states = [
                run.get("state", "").strip().lower()
                if isinstance(run, dict)
                and isinstance(run.get("state"), str)
                else ""
                for run in dag_runs
            ]
            if all(state == "success" for state in states):
                airflow_status = "success"
            else:
                airflow_error = (
                    "Airflow DAG history contains failed, unfinished, "
                    "or unrecognized run states."
                )

        if airflow_status != "success":
            airflow = ToolResult(
                status=airflow_status,
                data=airflow.data,
                error=airflow_error,
                checked_at=airflow.checked_at,
                duration_ms=airflow.duration_ms,
            )

    # 5. Inspect deployment and pod state independently.
    deployment_result = _run_read_only_tool(
        get_deployment_details,
        "Kubernetes deployment",
        name=INFERENCE_DEPLOYMENT_NAME,
        namespace=DEFAULT_NAMESPACE,
    )
    pods_result = _run_read_only_tool(
        list_pods,
        "Kubernetes pods",
        namespace=DEFAULT_NAMESPACE,
        limit=50,
    )

    kubernetes_data = {
        "deployment": deployment_result.model_dump(mode="json"),
        "pods": pods_result.model_dump(mode="json"),
    }

    if (
        deployment_result.status == "success"
        and pods_result.status == "success"
    ):
        deployment = deployment_result.data.get("data", {})
        pods = pods_result.data.get("data", [])

        desired = deployment.get("desired_replicas")
        available = deployment.get("available_replicas")
        ready_pods = sum(
            1
            for pod in pods
            if isinstance(pod, dict) and pod.get("ready") is True
        )

        replica_counts_valid = (
            isinstance(desired, int)
            and not isinstance(desired, bool)
            and isinstance(available, int)
            and not isinstance(available, bool)
        )

        healthy = (
            replica_counts_valid
            and desired > 0
            and available >= desired
            and len(pods) >= desired
            and ready_pods >= desired
        )

        kubernetes = ToolResult(
            status="success" if healthy else "partial",
            data={
                **kubernetes_data,
                "desired_replicas": desired,
                "available_replicas": available,
                "pod_count": len(pods),
                "ready_pod_count": ready_pods,
            },
            error=None if healthy else (
                "Deployment and pod readiness do not confirm all desired "
                "replicas are available"
            ),
            checked_at=datetime.now(UTC),
        )
    elif (
        deployment_result.status == "success"
        or pods_result.status == "success"
    ):
        kubernetes = ToolResult(
            status="partial",
            data=kubernetes_data,
            error="Only one of the deployment and pod checks succeeded",
            checked_at=datetime.now(UTC),
        )
    else:
        kubernetes = ToolResult(
            status="unavailable",
            data=kubernetes_data,
            error="Both Kubernetes checks failed",
            checked_at=datetime.now(UTC),
        )

    # 6. Check API liveness and readiness without assuming a port-forward.
    inference_api = check_inference_api()

    subsystem_results = {
        "model_selection": model_selection,
        "training_provenance": training_provenance,
        "dataset_state": dataset_state,
        "airflow": airflow,
        "kubernetes": kubernetes,
        "inference_api": inference_api,
    }

    for name, result in subsystem_results.items():
        if result.status not in {"success", "skipped"}:
            findings.append(
                DiagnosticFinding(
                    severity="warning",
                    source=name,
                    message=result.error or f"{name} check was {result.status}",
                )
            )

    if model_selection.status == "success" and isinstance(selected_run, dict):
        metrics = selected_run.get("metrics", {})
        f1_score = metrics.get("f1") if isinstance(metrics, dict) else None
        findings.append(
            DiagnosticFinding(
                severity="info",
                source="model_selection",
                message=(
                    f"Selected recent run {run_id}; its recorded F1 score is "
                    f"{f1_score}."
                ),
            )
        )

    if training_provenance.status == "success":
        provenance_data = training_provenance.data
        provenance = (
            provenance_data.get("provenance", {})
            if isinstance(provenance_data, dict)
            else {}
        )
        complete = (
            provenance_data.get("complete")
            if isinstance(provenance_data, dict)
            else None
        )
        if complete is not True:
            missing_tags = provenance_data.get("missing_tags", [])
            missing_tags = (
                [tag for tag in missing_tags if isinstance(tag, str)]
                if isinstance(missing_tags, list)
                else []
            )
            warning_message = (
                "Historical training dataset provenance is incomplete"
            )
            if missing_tags:
                warning_message += f"; missing tags: {', '.join(missing_tags)}"
            findings.append(
                DiagnosticFinding(
                    severity="warning",
                    source="training_provenance",
                    message=warning_message,
                )
            )
            recommendations.append(
                "Resolve missing training provenance before relying on the "
                "selected model's dataset lineage."
            )
        elif isinstance(provenance, dict):
            findings.append(
                DiagnosticFinding(
                    severity="info",
                    source="training_provenance",
                    message="Historical training dataset provenance was retrieved.",
                )
            )

    recommendations.append(
        "Review the DVC local and remote status before deciding whether the "
        "training dataset has changed."
    )
    recommendations.append(
        "Use dataset changes, provenance completeness, workflow outcomes, "
        "and current deployment health together when deciding whether to "
        "retrain; model F1 alone is insufficient."
    )

    statuses = [result.status for result in subsystem_results.values()]
    if all(status in {"success", "skipped"} for status in statuses):
        overall_status = "complete"
    elif not any(status in {"success", "partial"} for status in statuses):
        overall_status = "failed"
    else:
        overall_status = "partial"

    return DiagnosticReport(
        generated_at=generated_at,
        overall_status=overall_status,
        **subsystem_results,
        findings=findings,
        recommendations=recommendations,
    )
