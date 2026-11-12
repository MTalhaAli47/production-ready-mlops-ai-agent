"""Automated tests for the DVC tools."""

import subprocess

import yaml

from agent.tools import dvc_tools


def test_list_datasets_finds_dvc_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)
    metadata = tmp_path / "data" / "sample.csv.dvc"
    metadata.parent.mkdir()
    metadata.write_text(
        yaml.safe_dump({
            "outs": [{"path": "sample.csv", "md5": "abc123"}]
        })
    )

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert result["datasets"][0]["path"] == "data/sample.csv"
    assert result["datasets"][0]["hash"] == "abc123"


def test_list_datasets_handles_invalid_yaml(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)
    metadata = tmp_path / "broken.dvc"
    metadata.write_text("outs: [invalid")

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert "error" in result["datasets"][0]


def test_list_datasets_rejects_output_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)
    metadata = tmp_path / "outside.dvc"
    metadata.write_text(
        yaml.safe_dump({"outs": [{"path": "../../outside.csv"}]})
    )

    result = dvc_tools.list_datasets()

    assert "error" in result["datasets"][0]


def test_get_dataset_provenance_returns_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)
    metadata = tmp_path / "data" / "sample.csv.dvc"
    metadata.parent.mkdir()
    metadata.write_text(
        yaml.safe_dump({"outs": [{"path": "sample.csv", "md5": "abc"}]})
    )
    monkeypatch.setattr(
        dvc_tools,
        "_run",
        lambda command: {"success": True, "output": "abc123commit"},
    )

    result = dvc_tools.get_dataset_provenance("data/sample.csv")

    assert result["success"] is True
    assert result["git_commit"] == "abc123commit"


def test_get_dataset_provenance_rejects_invalid_path():
    result = dvc_tools.get_dataset_provenance("../../etc/passwd")

    assert result["success"] is False


def test_check_dataset_changes_reports_both_statuses(monkeypatch):
    monkeypatch.setattr(
        dvc_tools,
        "_run",
        lambda command: {"success": True, "output": "up to date"},
    )

    result = dvc_tools.check_dataset_changes()

    assert result["success"] is True
    assert result["local_status"]["success"] is True
    assert result["remote_status"]["success"] is True


def test_check_dataset_changes_reports_remote_failure(monkeypatch):
    def fake_run(command):
        if command == ["dvc", "status", "-c"]:
            return {"success": False, "error": "Remote unavailable"}
        return {"success": True, "output": "up to date"}

    monkeypatch.setattr(dvc_tools, "_run", fake_run)

    result = dvc_tools.check_dataset_changes()

    assert result["success"] is False
    assert result["remote_status"]["error"] == "Remote unavailable"


def test_compare_dataset_versions_parses_json(monkeypatch):
    monkeypatch.setattr(
        dvc_tools,
        "_run",
        lambda command: {
            "success": True,
            "output": '{"added": ["new.csv"], "deleted": [], "modified": []}',
        },
    )

    result = dvc_tools.compare_dataset_versions("HEAD~1", "HEAD")

    assert result["success"] is True
    assert result["comparison"]["added"] == ["new.csv"]


def test_compare_dataset_versions_rejects_invalid_revision():
    result = dvc_tools.compare_dataset_versions("--help", "HEAD")

    assert result["success"] is False
    assert result["error"] == "Invalid Git revision"


def test_compare_dataset_versions_propagates_dvc_failure(monkeypatch):
    monkeypatch.setattr(
        dvc_tools,
        "_run",
        lambda command: {"success": False, "error": "DVC failed"},
    )

    result = dvc_tools.compare_dataset_versions("HEAD~1", "HEAD")

    assert result["success"] is False
    assert result["error"] == "DVC failed"


def test_reproduce_dataset_builds_historical_instructions():
    result = dvc_tools.reproduce_dataset(
        "data/old_dataset.csv", "HEAD~1"
    )

    assert result["success"] is True
    assert result["instructions"][0] == (
        "git checkout HEAD~1 -- data/old_dataset.csv.dvc"
    )
    assert result["instructions"][1] == (
        "dvc checkout data/old_dataset.csv.dvc"
    )


def test_reproduce_dataset_rejects_path_traversal():
    result = dvc_tools.reproduce_dataset("../../etc/passwd", "HEAD")

    assert result["success"] is False


def test_reproduce_dataset_rejects_invalid_revision():
    result = dvc_tools.reproduce_dataset(
        "data/sample.csv", "--exec"
    )

    assert result["success"] is False


def test_run_handles_timeout(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="dvc status", timeout=30)

    monkeypatch.setattr(dvc_tools.subprocess, "run", timeout)

    result = dvc_tools._run(["dvc", "status"])

    assert result == {"success": False, "error": "Command timed out"}


def test_run_handles_missing_executable(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(dvc_tools.subprocess, "run", missing)

    result = dvc_tools._run(["dvc", "status"])

    assert result == {
        "success": False,
        "error": "Required executable not found",
    }


def test_run_handles_nonzero_exit_code(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        dvc_tools.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="DVC operation failed",
            stdout="",
        ),
    )

    result = dvc_tools._run(["dvc", "status"])

    assert result == {
        "success": False,
        "error": "DVC operation failed",
    }


def test_list_datasets_handles_invalid_metadata_structure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    metadata = tmp_path / "invalid.dvc"
    metadata.write_text("outs: invalid\n")

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert len(result["datasets"]) == 1
    assert "error" in result["datasets"][0]
    assert "outs" in result["datasets"][0]["error"]


def test_get_dataset_provenance_handles_git_failure(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    metadata = tmp_path / "data" / "sample.csv.dvc"
    metadata.parent.mkdir()
    metadata.write_text(
        yaml.safe_dump({"outs": [{"path": "sample.csv"}]})
    )

    monkeypatch.setattr(
        dvc_tools,
        "_run",
        lambda command: {
            "success": False,
            "error": "Git repository unavailable",
        },
    )

    result = dvc_tools.get_dataset_provenance("data/sample.csv")

    assert result["success"] is False
    assert result["error"] == "Git repository unavailable"


def test_compare_dataset_versions_handles_invalid_json(monkeypatch):
    monkeypatch.setattr(
        dvc_tools,
        "_run",
        lambda command: {
            "success": True,
            "output": "Unexpected DVC output",
        },
    )

    result = dvc_tools.compare_dataset_versions("HEAD~1", "HEAD")

    assert result["success"] is True
    assert result["comparison"] == {
        "raw_output": "Unexpected DVC output"
    }


# Additional tests targeting previously uncovered execution paths.


def test_validate_dataset_file_rejects_empty_and_non_string_paths():
    assert dvc_tools._validate_dataset_file("") is None
    assert dvc_tools._validate_dataset_file("   ") is None
    assert dvc_tools._validate_dataset_file(None) is None
    assert dvc_tools._validate_dataset_file(123) is None


def test_validate_dataset_file_rejects_windows_separators():
    assert dvc_tools._validate_dataset_file(
        r"data\training_data.csv"
    ) is None


def test_validate_dataset_file_rejects_symlink_escape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    outside = tmp_path.parent / "outside_dataset.csv"
    outside.write_text("id,value\n1,100\n")

    link = tmp_path / "dataset.csv"
    link.symlink_to(outside)

    assert dvc_tools._validate_dataset_file("dataset.csv") is None


def test_get_metadata_path_rejects_symlink_escape(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    outside = tmp_path.parent / "outside.csv.dvc"
    outside.write_text("outs: []\n")

    metadata = tmp_path / "data" / "sample.csv.dvc"
    metadata.parent.mkdir()
    metadata.symlink_to(outside)

    assert dvc_tools._get_metadata_path("data/sample.csv") is None


def test_run_handles_successful_command(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        dvc_tools.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="DVC operation successful\n",
            stderr="",
        ),
    )

    result = dvc_tools._run(["dvc", "status"])

    assert result == {
        "success": True,
        "output": "DVC operation successful",
    }


def test_list_datasets_skips_non_file_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    broken_link = tmp_path / "broken.dvc"
    broken_link.symlink_to(tmp_path / "missing.dvc")

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert result["datasets"] == []


def test_list_datasets_skips_excluded_directories(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    excluded = tmp_path / ".venv"
    excluded.mkdir()

    (excluded / "ignored.dvc").write_text(
        "outs:\n  - path: ignored.csv\n"
    )

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert result["datasets"] == []


def test_list_datasets_rejects_non_mapping_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    metadata = tmp_path / "invalid.dvc"
    metadata.write_text("- invalid\n- metadata\n")

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert len(result["datasets"]) == 1
    assert "error" in result["datasets"][0]
    assert "mapping" in result["datasets"][0]["error"]


def test_list_datasets_handles_invalid_output_entries(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    metadata = tmp_path / "invalid_output.dvc"
    metadata.write_text(
        yaml.safe_dump({
            "outs": [
                "invalid-entry",
                {"md5": "missing-path"},
                {"path": 123},
            ]
        })
    )

    result = dvc_tools.list_datasets()

    assert result["success"] is True
    assert len(result["datasets"]) == 3

    for dataset in result["datasets"]:
        assert dataset["error"] == "Invalid DVC output entry"


def test_get_dataset_provenance_rejects_non_mapping_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dvc_tools, "PROJECT_ROOT", tmp_path)

    metadata = tmp_path / "data" / "sample.csv.dvc"
    metadata.parent.mkdir()
    metadata.write_text("- invalid\n- metadata\n")

    result = dvc_tools.get_dataset_provenance("data/sample.csv")

    assert result["success"] is False
    assert "mapping" in result["error"]
