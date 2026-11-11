"""Safe DVC tools for dataset versioning and provenance."""

import json
import re
import subprocess
from pathlib import Path, PureWindowsPath

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT = 30


def _validate_dataset_file(dataset_file):
    """Validate a project-relative dataset path."""
    if not isinstance(dataset_file, str) or not dataset_file.strip():
        return None

    if "\\" in dataset_file:
        return None

    relative_path = Path(dataset_file)
    windows_path = PureWindowsPath(dataset_file)

    if (
        relative_path.is_absolute()
        or windows_path.is_absolute()
        or ".." in relative_path.parts
        or ".." in windows_path.parts
        or dataset_file.startswith("-")
    ):
        return None

    resolved = (PROJECT_ROOT / relative_path).resolve()

    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None

    return relative_path


def _get_metadata_path(dataset_file):
    """Return an existing DVC metadata path inside the project."""
    relative_path = _validate_dataset_file(dataset_file)

    if relative_path is None:
        return None

    metadata_path = (
        PROJECT_ROOT / f"{relative_path.as_posix()}.dvc"
    ).resolve()

    try:
        metadata_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None

    return metadata_path if metadata_path.is_file() else None


def _run(command):
    """Execute a fixed, argument-list command without using a shell."""
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )

        if result.returncode != 0:
            return {
                "success": False,
                "error": result.stderr.strip() or result.stdout.strip(),
            }

        return {"success": True, "output": result.stdout.strip()}

    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Command timed out"}

    except FileNotFoundError:
        return {"success": False, "error": "Required executable not found"}


def list_datasets():
    """Discover datasets tracked through DVC metadata files."""
    datasets = []
    excluded = {".git", ".venv", "dvc_storage", "__pycache__"}

    for metadata_file in PROJECT_ROOT.rglob("*.dvc"):
        if not metadata_file.is_file():
            continue

        if any(part in excluded for part in metadata_file.parts):
            continue

        relative_metadata = metadata_file.relative_to(PROJECT_ROOT).as_posix()

        try:
            metadata = yaml.safe_load(metadata_file.read_text()) or {}

            if not isinstance(metadata, dict):
                raise TypeError("DVC metadata must be a YAML mapping")

            outputs = metadata.get("outs", [])

            if not isinstance(outputs, list):
                raise TypeError("DVC metadata 'outs' must be a list")

            for output in outputs:
                if not isinstance(output, dict) or not isinstance(
                    output.get("path"), str
                ):
                    datasets.append({
                        "metadata_file": relative_metadata,
                        "error": "Invalid DVC output entry",
                    })
                    continue

                output_path = (
                    metadata_file.parent / output["path"]
                ).resolve()

                try:
                    output_path.relative_to(PROJECT_ROOT.resolve())
                except ValueError:
                    datasets.append({
                        "metadata_file": relative_metadata,
                        "error": "DVC output path escapes project directory",
                    })
                    continue

                datasets.append({
                    "metadata_file": relative_metadata,
                    "path": output_path.relative_to(
                        PROJECT_ROOT.resolve()
                    ).as_posix(),
                    "hash": output.get("md5"),
                })

        except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
            datasets.append({
                "metadata_file": relative_metadata,
                "error": str(exc),
            })

    return {"success": True, "datasets": datasets}


def get_dataset_provenance(dataset_file="data/raw/training_data.csv"):
    """Retrieve DVC metadata and the current Git commit."""
    metadata_path = _get_metadata_path(dataset_file)

    if metadata_path is None:
        return {
            "success": False,
            "error": "Invalid dataset path or DVC metadata file not found",
        }

    try:
        metadata = yaml.safe_load(metadata_path.read_text()) or {}

        if not isinstance(metadata, dict):
            raise TypeError("DVC metadata must be a YAML mapping")

    except (OSError, yaml.YAMLError, ValueError, TypeError) as exc:
        return {"success": False, "error": str(exc)}

    git_result = _run(["git", "rev-parse", "HEAD"])

    if not git_result["success"]:
        return git_result

    return {
        "success": True,
        "dataset": dataset_file,
        "git_commit": git_result["output"],
        "dvc_metadata": metadata_path.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "outputs": metadata.get("outs", []),
        "note": (
            "This reports the current Git commit, not necessarily "
            "the historical commit used by a training experiment."
        ),
    }


def check_dataset_changes():
    """Check local DVC changes and remote cache synchronization."""
    local = _run(["dvc", "status"])
    remote = _run(["dvc", "status", "-c"])

    return {
        "success": local["success"] and remote["success"],
        "local_status": local,
        "remote_status": remote,
    }


def compare_dataset_versions(old_revision, new_revision):
    """Compare DVC-tracked data between two Git revisions."""
    for revision in (old_revision, new_revision):
        if (
            not isinstance(revision, str)
            or not re.fullmatch(r"[A-Za-z0-9._/~^-]{1,100}", revision)
            or revision.startswith("-")
        ):
            return {"success": False, "error": "Invalid Git revision"}

    result = _run([
        "dvc", "diff", old_revision, new_revision, "--json"
    ])

    if not result["success"]:
        return result

    try:
        comparison = json.loads(result["output"])
    except json.JSONDecodeError:
        comparison = {"raw_output": result["output"]}

    return {
        "success": True,
        "old_revision": old_revision,
        "new_revision": new_revision,
        "comparison": comparison,
    }


def reproduce_dataset(dataset_file, git_revision):
    """Provide safe instructions for restoring a historical dataset."""
    relative_path = _validate_dataset_file(dataset_file)

    if relative_path is None:
        return {"success": False, "error": "Invalid dataset path"}

    if (
        not isinstance(git_revision, str)
        or not re.fullmatch(r"[A-Za-z0-9._/~^-]{1,100}", git_revision)
        or git_revision.startswith("-")
    ):
        return {"success": False, "error": "Invalid Git revision"}

    metadata_file = f"{relative_path.as_posix()}.dvc"

    return {
        "success": True,
        "dataset": relative_path.as_posix(),
        "git_revision": git_revision,
        "instructions": [
            f"git checkout {git_revision} -- {metadata_file}",
            f"dvc checkout {metadata_file}",
        ],
        "warning": (
            "These commands modify the working dataset. Run them in a "
            "clean working tree and inspect the target revision first."
        ),
    }
