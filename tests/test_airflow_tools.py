"""Automated tests for the Airflow tools."""

import json
import urllib.error

import pytest

from agent.tools import airflow_tools


@pytest.mark.parametrize(
    "value",
    ["", "../admin", "bad/id", "bad id", None, 123],
)
def test_validate_dag_id_rejects_invalid_values(value):
    result = airflow_tools._validate_identifier(
        value,
        airflow_tools.DAG_ID_PATTERN,
        "DAG ID",
    )
    assert result["success"] is False


@pytest.mark.parametrize(
    "limit,offset",
    [
        (0, 0),
        (101, 0),
        (True, 0),
        (10, -1),
        (10, True),
        ("10", 0),
        (10, "0"),
    ],
)
def test_validate_pagination_rejects_invalid_values(limit, offset):
    result = airflow_tools._validate_pagination(limit, offset)
    assert result["success"] is False


def test_validate_pagination_accepts_valid_values():
    assert airflow_tools._validate_pagination(50, 0) is None
    assert airflow_tools._validate_pagination(100, 100000) is None


def test_list_dags_returns_dags(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {
            "success": True,
            "data": {"dags": [{"dag_id": "example"}], "total_entries": 1},
        },
    )

    result = airflow_tools.list_dags()

    assert result["success"] is True
    assert result["dags"][0]["dag_id"] == "example"
    assert result["total_entries"] == 1


def test_list_dags_rejects_invalid_pagination():
    result = airflow_tools.list_dags(limit=True)
    assert result == {"success": False, "error": "Invalid limit"}


def test_list_dags_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {"success": True, "data": []},
    )

    result = airflow_tools.list_dags()
    assert result["success"] is False


def test_get_dag_details_encodes_identifier(monkeypatch):
    captured = {}

    def fake_request(path, **kwargs):
        captured["path"] = path
        return {"success": True, "data": {"dag_id": "dag.with-dash"}}

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.get_dag_details("dag.with-dash")

    assert result["success"] is True
    assert captured["path"] == "/dags/dag.with-dash"


def test_get_dag_details_rejects_invalid_identifier():
    result = airflow_tools.get_dag_details("../other")
    assert result["success"] is False


def test_list_dag_runs_uses_offset_pagination(monkeypatch):
    captured = {}

    def fake_request(path, query=None, **kwargs):
        captured["path"] = path
        captured["query"] = query
        return {
            "success": True,
            "data": {
                "dag_runs": [{"dag_run_id": "run-1"}],
                "total_entries": 3,
                "next_cursor": "next-page",
                "previous_cursor": None,
            },
        }

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.list_dag_runs("example", limit=10, offset=20)

    assert result["success"] is True
    assert captured["query"] == {"limit": 10, "offset": 20}
    assert result["next_cursor"] == "next-page"
    assert result["previous_cursor"] is None


def test_list_dag_runs_supports_cursor_pagination(monkeypatch):
    captured = {}

    def fake_request(path, query=None, **kwargs):
        captured["query"] = query
        return {
            "success": True,
            "data": {
                "dag_runs": [],
                "total_entries": 0,
                "next_cursor": None,
                "previous_cursor": "previous-page",
            },
        }

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.list_dag_runs("example", cursor="next-page")

    assert result["success"] is True
    assert captured["query"] == {
        "limit": 50,
        "offset": 0,
        "cursor": "next-page",
    }
    assert result["previous_cursor"] == "previous-page"


def test_list_dag_runs_allows_empty_initial_cursor(monkeypatch):
    captured = {}

    def fake_request(path, query=None, **kwargs):
        captured["query"] = query
        return {"success": True, "data": {"dag_runs": []}}

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.list_dag_runs("example", cursor="")

    assert result["success"] is True
    assert captured["query"]["cursor"] == ""


@pytest.mark.parametrize("cursor", [123, True, "x" * 4097])
def test_list_dag_runs_rejects_invalid_cursor(cursor):
    result = airflow_tools.list_dag_runs("example", cursor=cursor)
    assert result == {"success": False, "error": "Invalid cursor"}


def test_list_dag_runs_propagates_api_failure(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {
            "success": False,
            "error": "Airflow unavailable",
        },
    )

    result = airflow_tools.list_dag_runs("example")

    assert result == {
        "success": False,
        "error": "Airflow unavailable",
    }


def test_get_dag_run_details_returns_run(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {
            "success": True,
            "data": {"dag_run_id": "manual__test", "state": "success"},
        },
    )

    result = airflow_tools.get_dag_run_details("example", "manual__test")

    assert result["success"] is True
    assert result["dag_run"]["state"] == "success"


def test_get_dag_run_details_rejects_invalid_run_id():
    result = airflow_tools.get_dag_run_details("example", "../run")
    assert result["success"] is False


def test_list_task_instances_returns_instances(monkeypatch):
    captured = {}

    def fake_request(path, query=None, **kwargs):
        captured["query"] = query
        return {
            "success": True,
            "data": {
                "task_instances": [{"task_id": "task_a"}],
                "total_entries": 1,
            },
        }

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.list_task_instances(
        "example",
        "manual__test",
        limit=10,
        offset=5,
        task_id="task_a",
    )

    assert result["success"] is True
    assert result["task_instances"][0]["task_id"] == "task_a"
    assert captured["query"] == {
        "limit": 10,
        "offset": 5,
        "task_id": "task_a",
    }


def test_list_task_instances_rejects_invalid_task_id():
    result = airflow_tools.list_task_instances(
        "example",
        "manual__test",
        task_id="../bad",
    )
    assert result["success"] is False


def test_get_task_logs_returns_structured_logs(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {
            "success": True,
            "data": {
                "content": [{"event": "task completed"}],
                "continuation_token": None,
            },
        },
    )

    result = airflow_tools.get_task_logs(
        "example",
        "manual__test",
        "verify_environment",
    )

    assert result["success"] is True
    assert result["logs"][0]["event"] == "task completed"
    assert result["continuation_token"] is None


def test_get_task_logs_passes_continuation_token_and_map_index(monkeypatch):
    captured = {}

    def fake_request(path, query=None, **kwargs):
        captured["path"] = path
        captured["query"] = query
        return {
            "success": True,
            "data": {
                "content": ["continued log line"],
                "continuation_token": "next-token",
            },
        }

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.get_task_logs(
        "example",
        "manual__test",
        "mapped_task",
        continuation_token="previous-token",
        map_index=2,
        full_content=False,
    )

    assert result["success"] is True
    assert captured["query"] == {
        "full_content": "false",
        "map_index": 2,
        "token": "previous-token",
    }
    assert result["continuation_token"] == "next-token"


def test_get_task_logs_accepts_plain_text_content(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {
            "success": True,
            "data": {"content": ["line one", "line two"]},
        },
    )

    result = airflow_tools.get_task_logs(
        "example",
        "manual__test",
        "verify_environment",
    )

    assert result["success"] is True
    assert result["logs"] == ["line one", "line two"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"try_number": True},
        {"try_number": -1},
        {"try_number": "1"},
        {"full_content": "true"},
        {"map_index": True},
        {"map_index": -2},
        {"map_index": "0"},
        {"continuation_token": 123},
        {"continuation_token": "x" * 4097},
    ],
)
def test_get_task_logs_rejects_invalid_arguments(kwargs):
    result = airflow_tools.get_task_logs(
        "example",
        "manual__test",
        "verify_environment",
        **kwargs,
    )
    assert result["success"] is False


def test_get_task_logs_rejects_malformed_response(monkeypatch):
    monkeypatch.setattr(
        airflow_tools,
        "_api_request",
        lambda *args, **kwargs: {
            "success": True,
            "data": {"content": "not-a-list"},
        },
    )

    result = airflow_tools.get_task_logs(
        "example",
        "manual__test",
        "verify_environment",
    )

    assert result["success"] is False


def test_trigger_dag_requires_explicit_approval():
    result = airflow_tools.trigger_dag(
        "mlops_a03_healthcheck",
        "2026-11-14T12:00:00+00:00",
    )

    assert result["success"] is False
    assert "approved=True" in result["error"]


def test_trigger_dag_rejects_non_allowlisted_dag():
    result = airflow_tools.trigger_dag(
        "untrusted_dag",
        "2026-11-14T12:00:00+00:00",
        approved=True,
    )

    assert result["success"] is False
    assert "allowlist" in result["error"]


@pytest.mark.parametrize(
    "logical_date",
    [
        "2026-11-14T12:00:00",
        "not-a-date",
        "",
        None,
    ],
)
def test_trigger_dag_rejects_invalid_logical_date(logical_date):
    result = airflow_tools.trigger_dag(
        "mlops_a03_healthcheck",
        logical_date,
        approved=True,
    )

    assert result["success"] is False


def test_trigger_dag_sends_safe_approved_payload(monkeypatch):
    captured = {}

    def fake_request(path, method="GET", payload=None, **kwargs):
        captured["path"] = path
        captured["method"] = method
        captured["payload"] = payload
        return {
            "success": True,
            "data": {"dag_run_id": "manual__approved", "state": "queued"},
        }

    monkeypatch.setattr(airflow_tools, "_api_request", fake_request)

    result = airflow_tools.trigger_dag(
        "mlops_a03_healthcheck",
        "2026-11-14T12:00:00Z",
        approved=True,
        dag_run_id="manual__approved",
        note="Unit test",
    )

    assert result["success"] is True
    assert captured["method"] == "POST"
    assert captured["path"] == (
        "/dags/mlops_a03_healthcheck/dagRuns"
    )
    assert captured["payload"]["logical_date"] == (
        "2026-11-14T12:00:00+00:00"
    )
    assert captured["payload"]["dag_run_id"] == "manual__approved"
    assert captured["payload"]["note"] == "Unit test"
    assert "drain_dag" not in captured["payload"]


def test_trigger_dag_rejects_invalid_note():
    result = airflow_tools.trigger_dag(
        "mlops_a03_healthcheck",
        "2026-11-14T12:00:00Z",
        approved=True,
        note="x" * 1001,
    )

    assert result["success"] is False
    assert result["error"] == "Invalid trigger note"


def test_request_handles_http_error(monkeypatch):
    def raise_http_error(*args, **kwargs):
        raise urllib.error.HTTPError(
            "http://localhost",
            500,
            "Server Error",
            {},
            None,
        )

    monkeypatch.setattr(
        airflow_tools.urllib.request,
        "urlopen",
        raise_http_error,
    )

    result = airflow_tools._request("http://localhost")

    assert result["success"] is False
    assert result["status_code"] == 500


def test_request_handles_connection_error(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(
        airflow_tools.urllib.request,
        "urlopen",
        raise_connection_error,
    )

    result = airflow_tools._request("http://localhost")

    assert result["success"] is False
    assert "Airflow connection failed" in result["error"]


def test_request_rejects_invalid_json(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"not-json"

    monkeypatch.setattr(
        airflow_tools.urllib.request,
        "urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )

    result = airflow_tools._request("http://localhost")

    assert result["success"] is False
    assert result["error"] == "Airflow returned invalid JSON"


def test_request_posts_json_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"access_token":"test-token"}'

    def fake_urlopen(request, timeout):
        captured["method"] = request.method
        captured["headers"] = request.headers
        captured["body"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        airflow_tools.urllib.request,
        "urlopen",
        fake_urlopen,
    )

    result = airflow_tools._request(
        "http://localhost/auth/token",
        method="POST",
        payload={"username": "airflow", "password": "test-password"},
    )

    assert result["success"] is True
    assert result["status_code"] == 201
    assert result["data"]["access_token"] == "test-token"
    assert json.loads(captured["body"]) == {
        "username": "airflow",
        "password": "test-password",
    }
    assert captured["timeout"] == airflow_tools.REQUEST_TIMEOUT
