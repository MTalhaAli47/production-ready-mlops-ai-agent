"""Safe MLflow tools for experiment tracking and model inspection."""

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRACKING_URI = (
    f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MAX_RESULTS = 1000

MAX_SCAN_RESULTS = 10000

MAX_ARTIFACT_DOWNLOAD_BYTES = 50 * 1024 * 1024
MAX_CACHED_DOWNLOADS = 5
ARTIFACT_DOWNLOAD_CACHE = (
    Path(tempfile.gettempdir()) / "mlops-a03-mlflow-downloads"
)


def _get_client():
    """Create an MLflow client using the configured tracking URI."""
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_TRACKING_URI,
    )
    return MlflowClient(tracking_uri=tracking_uri)


def _validate_limit(limit):
    """Validate a result limit."""
    return (
        isinstance(limit, int)
        and not isinstance(limit, bool)
        and 1 <= limit <= MAX_RESULTS
    )


def _validate_metric(metric):
    """Validate an MLflow metric name."""
    return (
        isinstance(metric, str)
        and bool(metric.strip())
        and bool(re.fullmatch(r"[A-Za-z0-9_.-]+", metric))
    )


def _serialize_run(run):
    """Convert an MLflow Run into a JSON-friendly dictionary."""
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def _find_experiment(client, experiment_name=None, experiment_id=None):
    """Resolve an experiment by its exact name or ID."""
    if experiment_id is not None:
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            return None
        return client.get_experiment(experiment_id)

    if not isinstance(experiment_name, str) or not experiment_name.strip():
        return None

    experiments = client.search_experiments(max_results=MAX_RESULTS)

    for experiment in experiments:
        if experiment.name == experiment_name:
            return experiment

    return None


def list_experiments(max_results=100):
    """List MLflow experiments."""
    if not _validate_limit(max_results):
        return {"success": False, "error": "Invalid max_results limit"}

    try:
        client = _get_client()
        experiments = client.search_experiments(
            max_results=max_results
        )

        return {
            "success": True,
            "experiments": [
                {
                    "experiment_id": experiment.experiment_id,
                    "name": experiment.name,
                    "artifact_location": experiment.artifact_location,
                    "lifecycle_stage": experiment.lifecycle_stage,
                }
                for experiment in experiments
            ],
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def list_runs(
    experiment_name=None,
    experiment_id=None,
    max_results=100,
):
    """List runs belonging to a named experiment or experiment ID."""
    if not _validate_limit(max_results):
        return {"success": False, "error": "Invalid max_results limit"}

    if experiment_name is not None and experiment_id is not None:
        return {
            "success": False,
            "error": "Provide experiment_name or experiment_id, not both",
        }

    try:
        client = _get_client()
        experiment = _find_experiment(
            client,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
        )

        if experiment is None:
            return {"success": False, "error": "Experiment not found"}

        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=max_results,
        )

        return {
            "success": True,
            "experiment_name": experiment.name,
            "experiment_id": experiment.experiment_id,
            "runs": [_serialize_run(run) for run in runs],
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def get_run_details(run_id):
    """Retrieve the recorded details of one MLflow run."""
    if not isinstance(run_id, str) or not run_id.strip():
        return {"success": False, "error": "Invalid run_id"}

    try:
        client = _get_client()
        run = client.get_run(run_id)
        return {"success": True, "run": _serialize_run(run)}

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def get_best_run(
    experiment_name,
    metric,
    ascending=False,
):
    """Find the best run for a metric in an experiment."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        return {"success": False, "error": "Invalid experiment_name"}

    if not _validate_metric(metric):
        return {"success": False, "error": "Invalid metric name"}

    if not isinstance(ascending, bool):
        return {"success": False, "error": "ascending must be a boolean"}

    try:
        client = _get_client()
        experiment = _find_experiment(
            client,
            experiment_name=experiment_name,
        )

        if experiment is None:
            return {"success": False, "error": "Experiment not found"}

        direction = "ASC" if ascending else "DESC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {direction}"],
            max_results=1,
        )

        if not runs or metric not in runs[0].data.metrics:
            return {
                "success": False,
                "error": f"No runs found with metric '{metric}'",
            }

        return {
            "success": True,
            "experiment_name": experiment.name,
            "metric": metric,
            "direction": direction,
            "best_run": _serialize_run(runs[0]),
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def compare_runs(
    experiment_name,
    metric,
    max_results=5,
    ascending=False,
):
    """Compare runs in an experiment, ordered by a selected metric."""
    if not isinstance(experiment_name, str) or not experiment_name.strip():
        return {"success": False, "error": "Invalid experiment_name"}

    if not _validate_metric(metric):
        return {"success": False, "error": "Invalid metric name"}

    if not _validate_limit(max_results):
        return {"success": False, "error": "Invalid max_results limit"}

    if not isinstance(ascending, bool):
        return {"success": False, "error": "ascending must be a boolean"}

    try:
        client = _get_client()
        experiment = _find_experiment(
            client,
            experiment_name=experiment_name,
        )

        if experiment is None:
            return {"success": False, "error": "Experiment not found"}

        direction = "ASC" if ascending else "DESC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {direction}"],
            max_results=max_results,
        )

        return {
            "success": True,
            "experiment_name": experiment.name,
            "metric": metric,
            "direction": direction,
            "runs": [_serialize_run(run) for run in runs],
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}



def compare_runs_across_experiments(
    metric,
    experiment_names=None,
    experiment_ids=None,
    max_results=5,
    ascending=False,
):
    """Compare runs globally across multiple MLflow experiments.

    Scans at most MAX_SCAN_RESULTS runs using MLflow pagination.
    Runs without the selected metric are placed after runs that have it.
    """
    if (experiment_names is None) == (experiment_ids is None):
        return {
            "success": False,
            "error": "Provide exactly one of experiment_names or experiment_ids",
        }

    if not _validate_metric(metric):
        return {"success": False, "error": "Invalid metric name"}

    if not _validate_limit(max_results):
        return {"success": False, "error": "Invalid max_results limit"}

    if not isinstance(ascending, bool):
        return {"success": False, "error": "ascending must be a boolean"}

    identifiers = experiment_names if experiment_names is not None else experiment_ids

    if (
        not isinstance(identifiers, (list, tuple))
        or not identifiers
        or any(not isinstance(item, str) or not item.strip() for item in identifiers)
    ):
        return {
            "success": False,
            "error": "Experiment identifiers must be a non-empty list of strings",
        }

    if len(set(identifiers)) != len(identifiers):
        return {
            "success": False,
            "error": "Duplicate experiment identifiers are not allowed",
        }

    try:
        client = _get_client()
        experiments = []

        for identifier in identifiers:
            if experiment_names is not None:
                experiment = _find_experiment(
                    client,
                    experiment_name=identifier,
                )
            else:
                experiment = _find_experiment(
                    client,
                    experiment_id=identifier,
                )

            if experiment is None:
                return {
                    "success": False,
                    "error": f"Experiment not found: {identifier}",
                }

            experiments.append(experiment)

        experiment_ids_resolved = [
            experiment.experiment_id for experiment in experiments
        ]

        runs = []
        page_token = None

        while len(runs) < MAX_SCAN_RESULTS:
            page_size = min(MAX_RESULTS, MAX_SCAN_RESULTS - len(runs))

            page = client.search_runs(
                experiment_ids=experiment_ids_resolved,
                max_results=page_size,
                page_token=page_token,
            )

            runs.extend(page)
            page_token = page.token

            if not page_token:
                break

        truncated = bool(page_token)
        direction = "ASC" if ascending else "DESC"

        def sort_key(run):
            value = run.data.metrics.get(metric)

            if value is None:
                return (1, 0, -(run.info.start_time or 0), run.info.run_id)

            metric_order = value if ascending else -value

            return (
                0,
                metric_order,
                -(run.info.start_time or 0),
                run.info.run_id,
            )

        runs.sort(key=sort_key)

        return {
            "success": True,
            "experiment_names": [experiment.name for experiment in experiments],
            "experiment_ids": experiment_ids_resolved,
            "metric": metric,
            "direction": direction,
            "runs": [_serialize_run(run) for run in runs[:max_results]],
            "scanned_runs": len(runs),
            "scan_limit": MAX_SCAN_RESULTS,
            "truncated": truncated,
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}

def list_registered_models(max_results=100):
    """List registered MLflow models and their available versions."""
    if not _validate_limit(max_results):
        return {"success": False, "error": "Invalid max_results limit"}

    try:
        client = _get_client()
        models = client.search_registered_models(
            max_results=max_results
        )

        return {
            "success": True,
            "models": [
                {
                    "name": model.name,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": version.version,
                            "stage": version.current_stage,
                            "status": version.status,
                            "run_id": version.run_id,
                            "source": version.source,
                        }
                        for version in (model.latest_versions or [])
                    ],
                }
                for model in models
            ],
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}



def _validate_registered_model_name(name):
    """Validate a registered model name for safe filter construction."""
    return (
        isinstance(name, str)
        and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name))
    )


def get_registered_model_details(name):
    """Retrieve registered model metadata and its available versions."""
    if not _validate_registered_model_name(name):
        return {"success": False, "error": "Invalid registered model name"}

    try:
        client = _get_client()
        model = client.get_registered_model(name)

        versions = []
        page_token = None

        while len(versions) < MAX_SCAN_RESULTS:
            page_size = min(
                MAX_RESULTS,
                MAX_SCAN_RESULTS - len(versions),
            )

            page = client.search_model_versions(
                filter_string=f"name='{name}'",
                max_results=page_size,
                page_token=page_token,
            )

            versions.extend(page)
            page_token = page.token

            if not page_token:
                break

        truncated = bool(page_token)
        versions.sort(key=lambda item: int(item.version))

        return {
            "success": True,
            "model": {
                "name": model.name,
                "description": model.description,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "tags": dict(model.tags or {}),
                "aliases": {
                    alias: str(version)
                    for alias, version in (model.aliases or {}).items()
                },
                "scanned_versions": len(versions),
                "scan_limit": MAX_SCAN_RESULTS,
                "truncated": truncated,
                "versions": [
                    {
                        "version": str(version.version),
                        "description": version.description,
                        "creation_timestamp": version.creation_timestamp,
                        "last_updated_timestamp": version.last_updated_timestamp,
                        "stage": version.current_stage,
                        "status": version.status,
                        "status_message": version.status_message,
                        "run_id": version.run_id,
                        "source": version.source,
                        "tags": dict(version.tags or {}),
                        "model_id": version.model_id,
                    }
                    for version in versions
                ],
            },
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def get_model_version_details(name, version):
    """Retrieve a registered model version and its source run status."""
    if not _validate_registered_model_name(name):
        return {"success": False, "error": "Invalid registered model name"}

    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
    ):
        return {"success": False, "error": "Invalid model version"}

    try:
        client = _get_client()
        model_version = client.get_model_version(name, str(version))
        run = (
            client.get_run(model_version.run_id)
            if model_version.run_id
            else None
        )

        return {
            "success": True,
            "version": {
                "name": model_version.name,
                "version": str(model_version.version),
                "description": model_version.description,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "stage": model_version.current_stage,
                "status": model_version.status,
                "status_message": model_version.status_message,
                "run_id": model_version.run_id,
                "run_status": run.info.status if run else None,
                "source": model_version.source,
                "tags": dict(model_version.tags or {}),
                "model_id": model_version.model_id,
                "user_id": model_version.user_id,
            },
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}

def list_run_artifacts(run_id, path=None):
    """List artifacts recorded for an MLflow run."""
    if not isinstance(run_id, str) or not run_id.strip():
        return {"success": False, "error": "Invalid run_id"}

    if path is not None and (
        not isinstance(path, str)
        or path.startswith("/")
        or "\\" in path
        or ".." in Path(path).parts
    ):
        return {"success": False, "error": "Invalid artifact path"}

    try:
        client = _get_client()
        artifacts = client.list_artifacts(run_id, path=path)

        return {
            "success": True,
            "run_id": run_id,
            "artifacts": [
                {
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size,
                }
                for artifact in artifacts
            ],
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def get_run_dataset_provenance(run_id):
    """Retrieve DVC dataset provenance recorded on an MLflow run."""
    if not isinstance(run_id, str) or not run_id.strip():
        return {"success": False, "error": "Invalid run_id"}

    required_tags = (
        "dataset_path",
        "dataset_git_revision",
        "dataset_dvc_hash",
        "dataset_dvc_metadata_path",
        "dataset_version",
    )

    try:
        client = _get_client()
        run = client.get_run(run_id)
        tags = run.data.tags

        provenance = {
            key: tags.get(key)
            for key in required_tags
        }
        missing_tags = [
            key for key, value in provenance.items()
            if value is None
        ]

        return {
            "success": True,
            "run_id": run_id,
            "provenance": provenance,
            "complete": not missing_tags,
            "missing_tags": missing_tags,
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}

def _validate_artifact_path(path):
    """Accept only safe, non-empty relative POSIX artifact paths."""
    return (
        isinstance(path, str)
        and bool(path)
        and "\x00" not in path
        and "\\" not in path
        and ":" not in path
        and not path.startswith("/")
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )


def _artifact_tree_size(client, identifier, path=None, logged_model=False):
    """Calculate an artifact subtree size using MLflow metadata."""
    total_size = 0
    pending = [path]
    visited = 0

    while pending:
        current_path = pending.pop()
        if logged_model:
            artifacts = client.list_logged_model_artifacts(
                identifier,
                path=current_path,
            )
        else:
            artifacts = client.list_artifacts(
                identifier,
                path=current_path,
            )

        for artifact in artifacts:
            visited += 1
            if visited > MAX_SCAN_RESULTS:
                raise ValueError("Artifact listing exceeds the scan limit")

            if not _validate_artifact_path(artifact.path):
                raise ValueError("Invalid artifact path")

            if artifact.is_dir:
                pending.append(artifact.path)
            else:
                if (
                    isinstance(artifact.file_size, bool)
                    or not isinstance(artifact.file_size, int)
                    or artifact.file_size < 0
                ):
                    raise ValueError("Artifact size is unavailable")
                total_size += artifact.file_size
                if total_size > MAX_ARTIFACT_DOWNLOAD_BYTES:
                    raise ValueError("Artifact exceeds the 50 MB download limit")

    return total_size


def _prepare_download_directory():
    """Create a unique directory inside the managed download cache."""
    cache = ARTIFACT_DOWNLOAD_CACHE
    cache.mkdir(parents=True, exist_ok=True)

    if cache.is_symlink():
        raise ValueError("Download cache must not be a symbolic link")

    destination = cache / uuid.uuid4().hex
    destination.mkdir()
    return destination


def _finish_download(destination, downloaded_path):
    """Validate a download and retain only a bounded number of directories."""
    downloaded_source = Path(downloaded_path)
    if downloaded_source.is_symlink():
        raise ValueError("Downloaded artifact contains a symbolic link")

    destination = destination.resolve()
    downloaded = downloaded_source.resolve()

    if (
        not downloaded.exists()
        or not downloaded.is_relative_to(destination)
    ):
        raise ValueError("Downloaded artifact path is invalid")

    if downloaded.is_file():
        total_size = downloaded.stat().st_size
    elif downloaded.is_dir():
        total_size = 0
        for item in downloaded.rglob("*"):
            if item.is_symlink():
                raise ValueError("Downloaded artifact contains a symbolic link")
            if item.is_file():
                total_size += item.stat().st_size
    else:
        raise ValueError("Downloaded artifact is not a regular file or directory")

    if total_size > MAX_ARTIFACT_DOWNLOAD_BYTES:
        raise ValueError("Downloaded artifact exceeds the 50 MB limit")

    cache = destination.parent
    retained = sorted(
        (
            item for item in cache.iterdir()
            if item.is_dir() and not item.is_symlink()
        ),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )

    while len(retained) > MAX_CACHED_DOWNLOADS:
        old_directory = next(
            (
                item for item in reversed(retained)
                if item.resolve() != destination
            ),
            None,
        )
        if old_directory is None:
            break
        shutil.rmtree(old_directory)
        retained.remove(old_directory)

    return {
        "success": True,
        "local_path": str(downloaded),
        "size_bytes": total_size,
    }


def list_logged_model_artifacts(model_id, path=None):
    """List artifacts belonging to an MLflow 3 logged model."""
    if not isinstance(model_id, str) or not model_id.strip():
        return {"success": False, "error": "Invalid model_id"}

    if path is not None and not _validate_artifact_path(path):
        return {"success": False, "error": "Invalid artifact path"}

    try:
        client = _get_client()
        client.get_logged_model(model_id)
        artifacts = client.list_logged_model_artifacts(
            model_id,
            path=path,
        )

        return {
            "success": True,
            "model_id": model_id,
            "artifacts": [
                {
                    "path": artifact.path,
                    "is_dir": artifact.is_dir,
                    "file_size": artifact.file_size,
                }
                for artifact in artifacts
            ],
        }

    except (MlflowException, SQLAlchemyError) as exc:
        return {"success": False, "error": str(exc)}


def download_run_artifact(run_id, path):
    """Download one run artifact into the managed local cache."""
    if not isinstance(run_id, str) or not run_id.strip():
        return {"success": False, "error": "Invalid run_id"}

    if not _validate_artifact_path(path):
        return {"success": False, "error": "Invalid artifact path"}

    destination = None

    try:
        client = _get_client()
        _artifact_tree_size(client, run_id, path=path)

        destination = _prepare_download_directory()
        downloaded_path = client.download_artifacts(
            run_id,
            path,
            dst_path=str(destination),
        )

        return _finish_download(destination, downloaded_path)

    except (MlflowException, SQLAlchemyError, OSError, ValueError) as exc:
        if destination is not None and destination.exists():
            shutil.rmtree(destination)
        return {"success": False, "error": str(exc)}


def download_logged_model_artifact(model_id):
    """Download all artifacts for an MLflow 3 logged model."""
    if not isinstance(model_id, str) or not model_id.strip():
        return {"success": False, "error": "Invalid model_id"}

    destination = None

    try:
        import mlflow

        client = _get_client()
        client.get_logged_model(model_id)
        _artifact_tree_size(
            client,
            model_id,
            logged_model=True,
        )

        destination = _prepare_download_directory()
        downloaded_path = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_id}",
            dst_path=str(destination),
            tracking_uri=client.tracking_uri,
        )

        return _finish_download(destination, downloaded_path)

    except (MlflowException, SQLAlchemyError, OSError, ValueError) as exc:
        if destination is not None and destination.exists():
            shutil.rmtree(destination)
        return {"success": False, "error": str(exc)}
