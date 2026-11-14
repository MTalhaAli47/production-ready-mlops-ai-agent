"""Safe Airflow 3 API tools for DAG and task inspection."""

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_USERNAME = "airflow"
DEFAULT_ALLOWED_TRIGGER_DAGS = ("mlops_a03_healthcheck",)

REQUEST_TIMEOUT = 30
MAX_RESULTS = 100
MAX_OFFSET = 100000

DAG_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,250}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.:+-]{1,250}$")
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,250}$")


def _load_configuration():
    """Load Airflow credentials without exposing them."""
    load_dotenv(ENV_FILE)

    base_url = os.getenv("AIRFLOW_API_URL", DEFAULT_API_URL).strip()
    username = os.getenv(
        "_AIRFLOW_WWW_USER_USERNAME",
        DEFAULT_USERNAME,
    ).strip()
    password = os.getenv("_AIRFLOW_WWW_USER_PASSWORD", "")

    if not base_url.startswith(("http://", "https://")):
        raise ValueError("AIRFLOW_API_URL must use HTTP or HTTPS")

    if not username or not password:
        raise ValueError(
            "Airflow credentials are missing from the environment"
        )

    return base_url.rstrip("/"), username, password


def _validate_identifier(value, pattern, label):
    """Validate a path identifier before making an API request."""
    if not isinstance(value, str) or not pattern.fullmatch(value):
        return {
            "success": False,
            "error": f"Invalid {label}",
        }

    return None


def _validate_pagination(limit, offset):
    """Validate bounded offset-based pagination."""
    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or not 1 <= limit <= MAX_RESULTS
    ):
        return {"success": False, "error": "Invalid limit"}

    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or not 0 <= offset <= MAX_OFFSET
    ):
        return {"success": False, "error": "Invalid offset"}

    return None


def _request(url, method="GET", token=None, payload=None):
    """Make an HTTP request and return a structured result."""
    headers = {"Accept": "application/json"}

    data = None

    if token:
        headers["Authorization"] = f"Bearer {token}"

    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=REQUEST_TIMEOUT,
        ) as response:
            raw = response.read()

            if not raw:
                return {
                    "success": True,
                    "data": None,
                    "status_code": response.status,
                }

            try:
                result = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {
                    "success": False,
                    "error": "Airflow returned invalid JSON",
                    "status_code": response.status,
                }

            return {
                "success": True,
                "data": result,
                "status_code": response.status,
            }

    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            detail = parsed.get("detail", parsed)
        except (ValueError, AttributeError):
            detail = "Airflow returned an HTTP error"

        return {
            "success": False,
            "error": f"Airflow HTTP {exc.code}: {detail}",
            "status_code": exc.code,
        }

    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "success": False,
            "error": f"Airflow connection failed: {type(exc).__name__}",
        }


def _get_token():
    """Obtain a short-lived Airflow API access token."""
    try:
        base_url, username, password = _load_configuration()
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    result = _request(
        f"{base_url}/auth/token",
        method="POST",
        payload={
            "username": username,
            "password": password,
        },
    )

    if not result["success"]:
        return result

    data = result.get("data")

    if not isinstance(data, dict):
        return {
            "success": False,
            "error": "Invalid Airflow authentication response",
        }

    token = data.get("access_token")

    if not isinstance(token, str) or not token:
        return {
            "success": False,
            "error": "Airflow authentication did not return an access token",
        }

    return {
        "success": True,
        "base_url": base_url,
        "token": token,
    }


def _api_request(path, method="GET", payload=None, query=None):
    """Authenticate and call an Airflow API endpoint."""
    auth = _get_token()

    if not auth["success"]:
        return auth

    base_url = auth["base_url"]
    token = auth["token"]

    url = f"{base_url}/api/v2{path}"

    if query:
        url += "?" + urllib.parse.urlencode(query)

    result = _request(
        url,
        method=method,
        token=token,
        payload=payload,
    )

    # Do not return or retain authentication material in results.
    token = None
    auth["token"] = None

    return result


def list_dags(limit=50, offset=0):
    """List Airflow DAGs."""
    invalid = _validate_pagination(limit, offset)
    if invalid:
        return invalid

    result = _api_request(
        "/dags",
        query={"limit": limit, "offset": offset},
    )

    if not result["success"]:
        return result

    data = result.get("data")
    if not isinstance(data, dict):
        return {"success": False, "error": "Invalid DAG list response"}

    dags = data.get("dags")
    if not isinstance(dags, list):
        return {"success": False, "error": "DAG list missing from response"}

    return {
        "success": True,
        "dags": dags,
        "total_entries": data.get("total_entries"),
    }


def get_dag_details(dag_id):
    """Retrieve metadata for one DAG."""
    invalid = _validate_identifier(dag_id, DAG_ID_PATTERN, "DAG ID")
    if invalid:
        return invalid

    path = f"/dags/{urllib.parse.quote(dag_id, safe='')}"
    result = _api_request(path)

    if not result["success"]:
        return result

    if not isinstance(result.get("data"), dict):
        return {"success": False, "error": "Invalid DAG detail response"}

    return {"success": True, "dag": result["data"]}


def list_dag_runs(dag_id, limit=50, offset=0, cursor=None):
    """List runs for a DAG using offset or cursor pagination."""
    invalid = _validate_identifier(dag_id, DAG_ID_PATTERN, "DAG ID")
    if invalid:
        return invalid

    invalid = _validate_pagination(limit, offset)
    if invalid:
        return invalid

    if (
        cursor is not None
        and (
            not isinstance(cursor, str)
            or len(cursor) > 4096
        )
    ):
        return {"success": False, "error": "Invalid cursor"}

    path = (
        f"/dags/{urllib.parse.quote(dag_id, safe='')}"
        "/dagRuns"
    )

    query = {"limit": limit, "offset": offset}
    if cursor is not None:
        query["cursor"] = cursor

    result = _api_request(path, query=query)

    if not result["success"]:
        return result

    data = result.get("data")
    if not isinstance(data, dict):
        return {"success": False, "error": "Invalid DAG run list response"}

    runs = data.get("dag_runs")
    if not isinstance(runs, list):
        return {"success": False, "error": "DAG run list missing from response"}

    return {
        "success": True,
        "dag_runs": runs,
        "total_entries": data.get("total_entries"),
        "next_cursor": data.get("next_cursor"),
        "previous_cursor": data.get("previous_cursor"),
    }


def get_dag_run_details(dag_id, dag_run_id):
    """Retrieve metadata for one DAG run."""
    invalid = _validate_identifier(dag_id, DAG_ID_PATTERN, "DAG ID")
    if invalid:
        return invalid

    invalid = _validate_identifier(
        dag_run_id,
        RUN_ID_PATTERN,
        "DAG run ID",
    )
    if invalid:
        return invalid

    path = (
        f"/dags/{urllib.parse.quote(dag_id, safe='')}"
        f"/dagRuns/{urllib.parse.quote(dag_run_id, safe='')}"
    )

    result = _api_request(path)

    if not result["success"]:
        return result

    if not isinstance(result.get("data"), dict):
        return {"success": False, "error": "Invalid DAG run detail response"}

    return {"success": True, "dag_run": result["data"]}


def list_task_instances(
    dag_id,
    dag_run_id,
    limit=50,
    offset=0,
    task_id=None,
):
    """List task instances belonging to a DAG run."""
    invalid = _validate_identifier(dag_id, DAG_ID_PATTERN, "DAG ID")
    if invalid:
        return invalid

    invalid = _validate_identifier(
        dag_run_id,
        RUN_ID_PATTERN,
        "DAG run ID",
    )
    if invalid:
        return invalid

    invalid = _validate_pagination(limit, offset)
    if invalid:
        return invalid

    if task_id is not None:
        invalid = _validate_identifier(
            task_id,
            TASK_ID_PATTERN,
            "task ID",
        )
        if invalid:
            return invalid

    path = (
        f"/dags/{urllib.parse.quote(dag_id, safe='')}"
        f"/dagRuns/{urllib.parse.quote(dag_run_id, safe='')}"
        "/taskInstances"
    )

    query = {"limit": limit, "offset": offset}
    if task_id is not None:
        query["task_id"] = task_id

    result = _api_request(path, query=query)

    if not result["success"]:
        return result

    data = result.get("data")
    if not isinstance(data, dict):
        return {"success": False, "error": "Invalid task instance response"}

    instances = data.get("task_instances")
    if not isinstance(instances, list):
        return {
            "success": False,
            "error": "Task instance list missing from response",
        }

    return {
        "success": True,
        "task_instances": instances,
        "total_entries": data.get("total_entries"),
    }


def get_task_logs(
    dag_id,
    dag_run_id,
    task_id,
    try_number=1,
    full_content=True,
    continuation_token=None,
    map_index=-1,
):
    """Retrieve a page of logs for a specific task attempt."""
    invalid = _validate_identifier(dag_id, DAG_ID_PATTERN, "DAG ID")
    if invalid:
        return invalid

    invalid = _validate_identifier(
        dag_run_id,
        RUN_ID_PATTERN,
        "DAG run ID",
    )
    if invalid:
        return invalid

    invalid = _validate_identifier(task_id, TASK_ID_PATTERN, "task ID")
    if invalid:
        return invalid

    if (
        not isinstance(try_number, int)
        or isinstance(try_number, bool)
        or try_number < 0
    ):
        return {"success": False, "error": "Invalid try number"}

    if not isinstance(full_content, bool):
        return {"success": False, "error": "Invalid full_content value"}

    if (
        not isinstance(map_index, int)
        or isinstance(map_index, bool)
        or map_index < -1
    ):
        return {"success": False, "error": "Invalid map_index"}

    if (
        continuation_token is not None
        and (
            not isinstance(continuation_token, str)
            or len(continuation_token) > 4096
        )
    ):
        return {"success": False, "error": "Invalid continuation token"}

    path = (
        f"/dags/{urllib.parse.quote(dag_id, safe='')}"
        f"/dagRuns/{urllib.parse.quote(dag_run_id, safe='')}"
        f"/taskInstances/{urllib.parse.quote(task_id, safe='')}"
        f"/logs/{try_number}"
    )

    query = {
        "full_content": str(full_content).lower(),
        "map_index": map_index,
    }
    if continuation_token is not None:
        query["token"] = continuation_token

    result = _api_request(path, query=query)

    if not result["success"]:
        return result

    data = result.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("content"), list):
        return {"success": False, "error": "Invalid task log response"}

    return {
        "success": True,
        "logs": data["content"],
        "continuation_token": data.get("continuation_token"),
    }


def trigger_dag(
    dag_id,
    logical_date,
    approved=False,
    dag_run_id=None,
    note=None,
):
    """Trigger an explicitly approved DAG run.

    This operation is disabled unless approved=True and the DAG is
    included in DEFAULT_ALLOWED_TRIGGER_DAGS.
    """
    if approved is not True:
        return {
            "success": False,
            "error": "DAG triggering requires explicit approved=True",
        }

    invalid = _validate_identifier(dag_id, DAG_ID_PATTERN, "DAG ID")
    if invalid:
        return invalid

    if dag_id not in DEFAULT_ALLOWED_TRIGGER_DAGS:
        return {
            "success": False,
            "error": "DAG is not in the trigger allowlist",
        }

    if not isinstance(logical_date, str) or not logical_date.strip():
        return {
            "success": False,
            "error": "A timezone-aware logical_date is required",
        }

    try:
        parsed_date = datetime.fromisoformat(
            logical_date
        )
        if parsed_date.tzinfo is None or parsed_date.utcoffset() is None:
            raise ValueError
    except ValueError:
        return {
            "success": False,
            "error": "logical_date must be a valid timezone-aware ISO 8601 date",
        }

    payload = {"logical_date": parsed_date.isoformat()}

    if dag_run_id is not None:
        invalid = _validate_identifier(
            dag_run_id,
            RUN_ID_PATTERN,
            "DAG run ID",
        )
        if invalid:
            return invalid
        payload["dag_run_id"] = dag_run_id

    if note is not None:
        if not isinstance(note, str) or len(note) > 1000:
            return {"success": False, "error": "Invalid trigger note"}
        payload["note"] = note

    path = (
        f"/dags/{urllib.parse.quote(dag_id, safe='')}"
        "/dagRuns"
    )

    result = _api_request(path, method="POST", payload=payload)

    if not result["success"]:
        return result

    if not isinstance(result.get("data"), dict):
        return {
            "success": False,
            "error": "Invalid DAG trigger response",
        }

    return {
        "success": True,
        "dag_run": result["data"],
    }
