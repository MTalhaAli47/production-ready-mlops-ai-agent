from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import main

EXPECTED_TOOLS = {
    "run_diagnostics",
    "list_datasets",
    "get_dataset_provenance",
    "check_dataset_changes",
    "list_experiments",
    "list_runs",
    "get_run_details",
    "list_dags",
    "get_dag_details",
    "list_dag_runs",
    "list_pods",
    "get_pod_details",
    "list_deployments",
}


def make_response(*calls):
    parts = [
        SimpleNamespace(function_call=call, text=None)
        for call in calls
    ]
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def make_call(name, args=None):
    return SimpleNamespace(
        name=name,
        args={} if args is None else args,
        partial_args=None,
        will_continue=False,
    )


def test_registry_contains_only_approved_read_only_tools():
    assert set(main._TOOL_SPECS) == EXPECTED_TOOLS

    forbidden = {
        "reproduce_dataset",
        "trigger_dag",
        "register_model",
        "delete_model",
        "restart_pod",
        "deploy_model",
    }
    assert not forbidden.intersection(main._TOOL_SPECS)


def test_gemini_can_select_an_independent_tool():
    content, name, args = main._get_function_call(
        make_response(make_call("list_datasets"))
    )

    assert name == "list_datasets"
    assert args == {}
    assert content.parts


def test_gemini_can_select_tool_with_arguments():
    _, name, args = main._get_function_call(
        make_response(
            make_call(
                "get_dataset_provenance",
                {"dataset_file": "data/raw/training_data.csv"},
            )
        )
    )

    assert name == "get_dataset_provenance"
    assert args == {"dataset_file": "data/raw/training_data.csv"}


def test_unknown_function_is_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        main._get_function_call(
            make_response(make_call("trigger_dag", {"dag_id": "example"}))
        )


def test_multiple_function_calls_are_rejected():
    with pytest.raises(ValueError, match="exactly one"):
        main._get_function_call(
            make_response(
                make_call("list_datasets"),
                make_call("list_dags"),
            )
        )


def test_incomplete_function_call_is_rejected():
    call = make_call("list_datasets")
    call.will_continue = True

    with pytest.raises(ValueError, match="incomplete"):
        main._get_function_call(make_response(call))


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("list_datasets", {"unexpected": True}),
        ("get_dataset_provenance", {"dataset_file": "../../.env"}),
        ("list_pods", {"namespace": "default"}),
        ("list_dags", {"limit": True}),
        ("list_dags", {"limit": 101}),
        ("get_run_details", {}),
        ("get_pod_details", {"name": "../secret"}),
        ("run_diagnostics", {"unexpected": "argument"}),
        (
            "list_runs",
            {"experiment_name": "one", "experiment_id": "two"},
        ),
    ],
)
def test_invalid_arguments_are_rejected(name, args):
    with pytest.raises(ValueError):
        main._validate_arguments(name, args)


def test_valid_arguments_are_preserved():
    args = {
        "dag_id": "mlops_a03_training_pipeline",
        "limit": 10,
        "offset": 0,
    }

    assert main._validate_arguments("list_dag_runs", args) == args


def test_dispatch_invokes_only_the_selected_function(monkeypatch):
    selected = Mock(return_value={"datasets": ["training_data.csv"]})
    unused = Mock()

    monkeypatch.setitem(
        main._TOOL_SPECS,
        "list_datasets",
        {**main._TOOL_SPECS["list_datasets"], "function": selected},
    )
    monkeypatch.setitem(
        main._TOOL_SPECS,
        "list_dags",
        {**main._TOOL_SPECS["list_dags"], "function": unused},
    )

    result = main._dispatch_tool("list_datasets", {})

    assert result == {"result": {"datasets": ["training_data.csv"]}}
    selected.assert_called_once_with()
    unused.assert_not_called()


def test_dispatch_rejects_invalid_arguments_before_execution(monkeypatch):
    selected = Mock()

    monkeypatch.setitem(
        main._TOOL_SPECS,
        "list_pods",
        {**main._TOOL_SPECS["list_pods"], "function": selected},
    )

    with pytest.raises(ValueError):
        main._dispatch_tool("list_pods", {"namespace": "default"})

    selected.assert_not_called()


def test_dispatch_preserves_tool_runtime_errors(monkeypatch):
    selected = Mock(side_effect=RuntimeError("tool unavailable"))

    monkeypatch.setitem(
        main._TOOL_SPECS,
        "list_datasets",
        {**main._TOOL_SPECS["list_datasets"], "function": selected},
    )

    with pytest.raises(RuntimeError, match="tool unavailable"):
        main._dispatch_tool("list_datasets", {})


def test_result_sanitization_redacts_sensitive_fields():
    result = main._sanitize(
        {
            "status": "ok",
            "api_key": "secret-value",
            "nested": {"password": "hidden", "count": 2},
        }
    )

    assert result["status"] == "ok"
    assert result["api_key"] == "[redacted]"
    assert result["nested"]["password"] == "[redacted]"
    assert result["nested"]["count"] == 2


def test_result_sanitization_handles_non_finite_numbers():
    result = main._sanitize({"metric": float("nan")})

    assert result == {"metric": "nan"}


def test_oversized_tool_result_is_rejected_as_bounded_response():
    result = main._bounded_result({f"field_{i}": "x" * 4_000 for i in range(10)})

    assert result == {
        "error": "Tool result exceeded the response size limit.",
        "truncated": True,
    }


def test_diagnostic_dispatch_requires_diagnostic_report(monkeypatch):
    runner = Mock(return_value={"not": "a diagnostic report"})

    with pytest.raises(main.InvalidDiagnosticReportError):
        main._dispatch_tool("run_diagnostics", {}, diagnostics_runner=runner)

    runner.assert_called_once_with()
