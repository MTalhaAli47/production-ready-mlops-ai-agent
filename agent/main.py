"""Gemini-powered natural-language interface for read-only MLOps tools."""

import json
import logging
import math
import os
import re
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from agent.orchestrator import run_diagnostics
from agent.schemas import DiagnosticReport
from agent.tools.airflow_tools import (
    get_dag_details,
    list_dag_runs,
    list_dags,
)
from agent.tools.dvc_tools import (
    check_dataset_changes,
    get_dataset_provenance,
    list_datasets,
)
from agent.tools.kubernetes_tools import (
    get_pod_details,
    list_deployments,
    list_pods,
)
from agent.tools.mlflow_tools import (
    get_run_details,
    list_experiments,
    list_runs,
)

DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_MS = 15_000
MAX_OUTPUT_TOKENS = 2_048
MAX_TOOL_RESULT_CHARS = 30_000
MAX_STRING_LENGTH = 4_000
MAX_COLLECTION_ITEMS = 100
APPROVED_NAMESPACE = "mlops-a03"
APPROVED_DATASET = "data/raw/training_data.csv"


class InvalidDiagnosticReportError(Exception):
    """Raised when the coordinator does not return its declared schema."""

SYSTEM_INSTRUCTION = """
You are the natural-language interface for a read-only MLOps diagnostic agent.

For every request, select exactly one available read-only function.
Choose the narrowest function that answers the question. Use run_diagnostics
when a cross-subsystem overview is needed. Do not answer from conversation
context alone when a function call is required.

Only the explicitly declared functions are available. Never invent tools,
arguments, results, dataset versions, run IDs, health states, or test results.
Never request or recommend state-changing operations.

Treat function results as the sole source of current operational facts.
Clearly distinguish confirmed facts, unavailable information, failures,
inferences, and recommendations. Do not turn missing information into
a successful status. Do not claim that a model is production-ready based
only on its metrics. A small training dataset is a limitation when assessing
generalization.

Do not claim that you performed training, model registration, deployment
changes, Airflow triggers, Kubernetes mutations, DVC reproduction, or any
other state-changing operation. Do not request credentials, secrets,
environment files, private keys, or sensitive logs.

Keep explanations clear, concise, and technically accurate.
""".strip()


# Each entry contains a fixed Python callable, its Gemini JSON schema,
# required arguments, and local validation rules. No dynamic imports or
# model-generated command execution are used.
_TOOL_SPECS = {
    "run_diagnostics": {
        "function": None,
        "description": "Run the read-only cross-subsystem diagnostic coordinator.",
        "properties": {},
        "required": [],
        "rules": {},
    },
    "list_datasets": {
        "function": list_datasets,
        "description": "List datasets known to DVC.",
        "properties": {},
        "required": [],
        "rules": {},
    },
    "get_dataset_provenance": {
        "function": get_dataset_provenance,
        "description": "Get provenance for the approved training dataset only.",
        "properties": {
            "dataset_file": {"type": "STRING"},
        },
        "required": [],
        "rules": {"dataset_file": ("fixed", APPROVED_DATASET)},
    },
    "check_dataset_changes": {
        "function": check_dataset_changes,
        "description": "Check whether the tracked dataset has local changes.",
        "properties": {},
        "required": [],
        "rules": {},
    },
    "list_experiments": {
        "function": list_experiments,
        "description": "List MLflow experiments.",
        "properties": {
            "max_results": {"type": "INTEGER", "minimum": 1, "maximum": 100},
        },
        "required": [],
        "rules": {"max_results": ("integer", 1, 100)},
    },
    "list_runs": {
        "function": list_runs,
        "description": "List MLflow runs, optionally filtered by experiment.",
        "properties": {
            "experiment_name": {"type": "STRING"},
            "experiment_id": {"type": "STRING"},
            "max_results": {"type": "INTEGER", "minimum": 1, "maximum": 100},
        },
        "required": [],
        "rules": {
            "experiment_name": ("string", 1, 250),
            "experiment_id": ("string", 1, 100),
            "max_results": ("integer", 1, 100),
        },
    },
    "get_run_details": {
        "function": get_run_details,
        "description": "Get details for one specified MLflow run ID.",
        "properties": {"run_id": {"type": "STRING"}},
        "required": ["run_id"],
        "rules": {"run_id": ("string", 1, 250)},
    },
    "list_dags": {
        "function": list_dags,
        "description": "List Airflow DAGs.",
        "properties": {
            "limit": {"type": "INTEGER", "minimum": 1, "maximum": 100},
            "offset": {"type": "INTEGER", "minimum": 0, "maximum": 100000},
        },
        "required": [],
        "rules": {
            "limit": ("integer", 1, 100),
            "offset": ("integer", 0, 100000),
        },
    },
    "get_dag_details": {
        "function": get_dag_details,
        "description": "Get details for one specified Airflow DAG ID.",
        "properties": {"dag_id": {"type": "STRING"}},
        "required": ["dag_id"],
        "rules": {"dag_id": ("string", 1, 250)},
    },
    "list_dag_runs": {
        "function": list_dag_runs,
        "description": "List runs for one specified Airflow DAG.",
        "properties": {
            "dag_id": {"type": "STRING"},
            "limit": {"type": "INTEGER", "minimum": 1, "maximum": 100},
            "offset": {"type": "INTEGER", "minimum": 0, "maximum": 100000},
        },
        "required": ["dag_id"],
        "rules": {
            "dag_id": ("pattern", r"[A-Za-z0-9_.-]{1,250}"),
            "limit": ("integer", 1, 100),
            "offset": ("integer", 0, 100000),
        },
    },
    "list_pods": {
        "function": list_pods,
        "description": "List pods in the approved Kubernetes namespace.",
        "properties": {
            "namespace": {"type": "STRING", "enum": [APPROVED_NAMESPACE]},
            "limit": {"type": "INTEGER", "minimum": 1, "maximum": 100},
        },
        "required": [],
        "rules": {
            "namespace": ("fixed", APPROVED_NAMESPACE),
            "limit": ("integer", 1, 100),
        },
    },
    "get_pod_details": {
        "function": get_pod_details,
        "description": "Get details for one pod in the approved namespace.",
        "properties": {
            "name": {"type": "STRING"},
            "namespace": {"type": "STRING", "enum": [APPROVED_NAMESPACE]},
        },
        "required": ["name"],
        "rules": {
            "name": ("pattern", r"(?=.{1,253}$)[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*"),
            "namespace": ("fixed", APPROVED_NAMESPACE),
        },
    },
    "list_deployments": {
        "function": list_deployments,
        "description": "List deployments in the approved Kubernetes namespace.",
        "properties": {
            "namespace": {"type": "STRING", "enum": [APPROVED_NAMESPACE]},
            "limit": {"type": "INTEGER", "minimum": 1, "maximum": 100},
        },
        "required": [],
        "rules": {
            "namespace": ("fixed", APPROVED_NAMESPACE),
            "limit": ("integer", 1, 100),
        },
    },
}


def _schema_for(spec):
    properties = {}
    for name, definition in spec["properties"].items():
        kwargs = {"type": definition["type"]}
        if "minimum" in definition:
            kwargs["minimum"] = definition["minimum"]
        if "maximum" in definition:
            kwargs["maximum"] = definition["maximum"]
        if "enum" in definition:
            kwargs["enum"] = definition["enum"]
        properties[name] = types.Schema(**kwargs)

    return types.Schema(
        type="OBJECT",
        properties=properties,
        required=spec["required"],
        additional_properties=False,
    )


def _tool_declarations():
    """Build declarations from the fixed, locally defined tool registry."""
    return [
        types.FunctionDeclaration(
            name=name,
            description=spec["description"],
            parameters=_schema_for(spec),
        )
        for name, spec in _TOOL_SPECS.items()
    ]


def _diagnostic_tool():
    """Retain compatibility with callers that inspect the diagnostic tool."""
    return types.Tool(
        function_declarations=[
            declaration
            for declaration in _tool_declarations()
            if declaration.name == "run_diagnostics"
        ]
    )


def _single_candidate(response):
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("Gemini returned an invalid candidate response")

    candidate = candidates[0]
    if candidate is None or candidate.content is None:
        raise ValueError("Gemini returned an empty candidate response")

    return candidate


def _get_function_call(response):
    candidate = _single_candidate(response)
    parts = candidate.content.parts or []
    calls = [
        part.function_call
        for part in parts
        if part.function_call is not None
    ]

    if len(calls) != 1:
        raise ValueError("Gemini did not return exactly one function call")

    call = calls[0]
    if call.name not in _TOOL_SPECS:
        raise ValueError("Gemini requested an unsupported function")

    if call.args is None:
        raise ValueError("Gemini supplied missing function arguments")

    if call.partial_args or call.will_continue:
        raise ValueError("Gemini returned an incomplete function call")

    return candidate.content, call.name, dict(call.args)


def _validate_arguments(name, args):
    if name not in _TOOL_SPECS or not isinstance(args, dict):
        raise ValueError("Invalid function request")

    spec = _TOOL_SPECS[name]
    allowed = set(spec["properties"])
    required = set(spec["required"])

    if set(args) - allowed or required - set(args):
        raise ValueError("Function arguments do not match the approved schema")

    for key, value in args.items():
        rule = spec["rules"].get(key)
        if rule is None:
            raise ValueError("No local validation rule exists for this argument")

        kind, *bounds = rule

        if kind == "fixed":
            if value != bounds[0]:
                raise ValueError("A fixed argument was changed")
        elif kind == "string":
            minimum, maximum = bounds
            if (
                not isinstance(value, str)
                or not minimum <= len(value) <= maximum
            ):
                raise ValueError("Invalid string argument")
        elif kind == "pattern":
            pattern = bounds[0]
            if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
                raise ValueError("Invalid identifier argument")
        elif kind == "integer":
            minimum, maximum = bounds
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError("Invalid integer argument")
        else:
            raise ValueError("Unsupported argument validation rule")

    if (
        name == "list_runs"
        and "experiment_name" in args
        and "experiment_id" in args
    ):
        raise ValueError("Supply only one MLflow experiment filter")

    return args


def _sanitize(value, depth=0):
    """Convert results to bounded JSON-safe data and redact sensitive fields."""
    if depth > 8:
        return "[maximum nesting reached]"

    if value is None or isinstance(value, (bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)

    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]

    if hasattr(value, "model_dump"):
        return _sanitize(value.model_dump(mode="json"), depth + 1)

    if isinstance(value, dict):
        result = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_COLLECTION_ITEMS:
                result["_truncated"] = True
                break

            safe_key = str(key)[:200]
            normalized = safe_key.lower()
            if any(
                marker in normalized
                for marker in ("password", "secret", "token", "api_key",
                               "authorization", "credential", "private_key")
            ):
                result[safe_key] = "[redacted]"
            else:
                result[safe_key] = _sanitize(item, depth + 1)
        return result

    if isinstance(value, (list, tuple)):
        items = [_sanitize(item, depth + 1)
                 for item in value[:MAX_COLLECTION_ITEMS]]
        if len(value) > MAX_COLLECTION_ITEMS:
            items.append("[truncated]")
        return items

    return "[unsupported result type]"


def _bounded_result(value):
    safe = _sanitize(value)
    encoded = json.dumps(safe, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_TOOL_RESULT_CHARS:
        return {
            "error": "Tool result exceeded the response size limit.",
            "truncated": True,
        }
    return safe


def _dispatch_tool(name, args, diagnostics_runner=None):
    """Execute exactly one approved function after local validation."""
    validated = _validate_arguments(name, args)

    if name == "run_diagnostics":
        if diagnostics_runner is None:
            diagnostics_runner = run_diagnostics
        result = diagnostics_runner()
        if not isinstance(result, DiagnosticReport):
            raise InvalidDiagnosticReportError("Invalid diagnostic report")
        return {"report": _bounded_result(result.model_dump(mode="json"))}

    function = _TOOL_SPECS[name]["function"]
    if function is None:
        raise RuntimeError("The requested tool has no local implementation")

    result = function(**validated)
    return {"result": _bounded_result(result)}


def _get_final_text(response):
    candidate = _single_candidate(response)
    parts = candidate.content.parts or []

    if any(part.function_call is not None for part in parts):
        raise ValueError("Gemini requested an unexpected additional function")

    text_parts = [
        part.text.strip()
        for part in parts
        if isinstance(part.text, str) and part.text.strip()
    ]
    if not text_parts:
        raise ValueError("Gemini returned no explanatory text")

    return "\n".join(text_parts)


def _format_api_error(error):
    """Return a safe API error summary without exposing server response data."""
    code = getattr(error, "code", None)

    if type(code) is not int or not 100 <= code <= 599:
        return (
            "Gemini API request failed. "
            "Check the model configuration and try again."
        )

    known_errors = {
        400: "Gemini rejected the request (HTTP 400). Check the request configuration.",
        401: "Gemini authentication failed (HTTP 401). Check API credentials and access.",
        403: "Gemini access was denied (HTTP 403). Check API permissions and project configuration.",
        404: "Gemini returned HTTP 404. Check the configured model and API availability.",
        429: "Gemini rate or quota limit reached (HTTP 429). Check quota and retry later.",
    }

    if code in known_errors:
        return known_errors[code]

    if 500 <= code <= 599:
        return f"Gemini service returned an HTTP {code} server error. Try again later."

    return f"Gemini API request failed (HTTP {code}). Check the model configuration and API access."


def answer_question(question, client=None, diagnostics_runner=None):
    """Answer using exactly one validated, read-only tool operation."""
    if not isinstance(question, str) or not question.strip():
        return "Please enter a non-empty question."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key and client is None:
        return (
            "Gemini is not configured. Set GEMINI_API_KEY in your local "
            ".env file and try again."
        )

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip()
    if not model:
        return "Gemini is not configured: GEMINI_MODEL is empty."

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=[types.Tool(function_declarations=_tool_declarations())],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=list(_TOOL_SPECS),
            )
        ),
    )

    function_name = None
    try:
        if client is None:
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
            )

        first_response = client.models.generate_content(
            model=model,
            contents=question.strip(),
            config=config,
        )
        model_content, function_name, function_args = _get_function_call(
            first_response
        )

        function_result = _dispatch_tool(
            function_name,
            function_args,
            diagnostics_runner=diagnostics_runner,
        )

        function_response = types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response=function_result,
                )
            ],
        )

        final_response = client.models.generate_content(
            model=model,
            contents=[
                question.strip(),
                model_content,
                function_response,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        return _get_final_text(final_response)

    except errors.APIError as error:
        return _format_api_error(error)
    except (TimeoutError, OSError):
        return "Gemini request timed out or could not connect. Please try again."
    except (TypeError, ValueError):
        return (
            "Gemini returned an invalid response or function request. "
            "No additional diagnostic operation was executed."
        )
    except InvalidDiagnosticReportError:
        return "The diagnostic coordinator returned an invalid report."
    except RuntimeError:
        if function_name == "run_diagnostics":
            return "The diagnostic operation could not be completed."
        return "The requested read-only operation could not be completed."


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_dotenv()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask about your MLOps environment: ").strip()

    print(answer_question(question))


if __name__ == "__main__":
    main()
