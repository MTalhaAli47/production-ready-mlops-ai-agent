"""Focused tests for the Gemini natural-language diagnostic interface."""

from datetime import UTC, datetime
from unittest.mock import Mock

from google.genai import types

from agent.main import _format_api_error, answer_question
from agent.schemas import DiagnosticReport, ToolResult


def make_report():
    """Return a valid, minimal diagnostic report."""
    result = ToolResult(status="success", data={})
    return DiagnosticReport(
        generated_at=datetime.now(UTC),
        overall_status="complete",
        model_selection=result,
        training_provenance=result,
        dataset_state=result,
        airflow=result,
        kubernetes=result,
        inference_api=result,
        findings=[],
        recommendations=[],
    )


def make_function_response(name="run_diagnostics", args=None):
    """Build an SDK response containing one function call."""
    call = types.FunctionCall(name=name, args={} if args is None else args)
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[types.Part(function_call=call)],
        )
    )
    return types.GenerateContentResponse(candidates=[candidate])


def make_text_response(text="Diagnostics are complete."):
    """Build an SDK response containing explanatory text."""
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[types.Part(text=text)],
        )
    )
    return types.GenerateContentResponse(candidates=[candidate])


def make_client(*responses):
    client = Mock()
    client.models.generate_content.side_effect = responses
    return client


def test_answer_question_runs_diagnostics_once_and_returns_explanation(monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    client = make_client(
        make_function_response(),
        make_text_response("The deployment is healthy."),
    )
    runner = Mock(return_value=make_report())

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == "The deployment is healthy."
    runner.assert_called_once_with()
    assert client.models.generate_content.call_count == 2
    first_call = client.models.generate_content.call_args_list[0]
    assert first_call.kwargs["model"] == "gemini-3.5-flash-lite"

    config = first_call.kwargs["config"]
    calling_config = config.tool_config.function_calling_config
    assert calling_config.mode == "ANY"
    assert set(calling_config.allowed_function_names) == {
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

    declared_names = {
        declaration.name
        for tool in config.tools
        for declaration in tool.function_declarations
    }
    assert declared_names == set(calling_config.allowed_function_names)
    assert not declared_names.intersection({
        "trigger_dag",
        "reproduce_dataset",
        "get_pod_logs",
        "get_task_logs",
    })

    second_call = client.models.generate_content.call_args_list[1]
    assert second_call.kwargs["config"].tools is None


def test_answer_question_without_api_key_does_not_run_diagnostics(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        diagnostics_runner=runner,
    )

    assert "Gemini is not configured" in answer
    runner.assert_not_called()


def test_unsupported_function_call_does_not_run_diagnostics():
    client = make_client(
        make_function_response(name="delete_deployment"),
    )
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_not_called()
    assert client.models.generate_content.call_count == 1


def test_non_empty_function_arguments_are_rejected():
    client = make_client(make_function_response(args={"namespace": "default"}))
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_not_called()


def test_response_without_function_call_is_rejected():
    client = make_client(make_text_response("No function call."))
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_not_called()


def test_multiple_function_calls_are_rejected():
    candidate = types.Candidate(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="run_diagnostics",
                        args={},
                    )
                ),
                types.Part(
                    function_call=types.FunctionCall(
                        name="run_diagnostics",
                        args={},
                    )
                ),
            ],
        )
    )
    client = make_client(
        types.GenerateContentResponse(candidates=[candidate])
    )
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_not_called()


def test_empty_candidate_response_is_rejected():
    client = make_client(types.GenerateContentResponse(candidates=[]))
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_not_called()


def test_invalid_coordinator_report_is_rejected():
    client = make_client(make_function_response())
    runner = Mock(return_value={"overall_status": "complete"})

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid report" in answer
    runner.assert_called_once_with()
    assert client.models.generate_content.call_count == 1


def test_final_response_with_another_function_call_is_rejected():
    client = make_client(
        make_function_response(),
        make_function_response(),
    )
    runner = Mock(return_value=make_report())

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_called_once_with()
    assert client.models.generate_content.call_count == 2


def test_diagnostic_runtime_error_is_handled():
    client = make_client(make_function_response())
    runner = Mock(side_effect=RuntimeError("diagnostic failed"))

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == "The diagnostic operation could not be completed."
    runner.assert_called_once_with()
    assert client.models.generate_content.call_count == 1


def test_gemini_api_error_is_handled():
    from google.genai import errors

    class SimulatedAPIError(errors.APIError):
        def __init__(self):
            Exception.__init__(self, "simulated API failure")

    client = make_client()
    client.models.generate_content.side_effect = SimulatedAPIError()
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == (
        "Gemini API request failed. Check the model configuration and try again."
    )
    runner.assert_not_called()


def test_gemini_timeout_is_handled():
    client = make_client()
    client.models.generate_content.side_effect = TimeoutError("simulated timeout")
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == (
        "Gemini request timed out or could not connect. Please try again."
    )
    runner.assert_not_called()


def test_connection_error_is_handled():
    client = make_client()
    client.models.generate_content.side_effect = OSError("simulated connection error")
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == (
        "Gemini request timed out or could not connect. Please try again."
    )
    runner.assert_not_called()


def test_client_initialization_error_is_handled(monkeypatch):
    from unittest.mock import patch

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    runner = Mock()

    with patch(
        "agent.main.genai.Client",
        side_effect=OSError("simulated initialization failure"),
    ) as client_constructor:
        answer = answer_question(
            "Check my MLOps environment.",
            diagnostics_runner=runner,
        )

    assert answer == (
        "Gemini request timed out or could not connect. Please try again."
    )
    client_constructor.assert_called_once()
    runner.assert_not_called()


def test_empty_final_explanation_is_rejected():
    client = make_client(
        make_function_response(),
        make_text_response(""),
    )
    runner = Mock(return_value=make_report())

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert "invalid response or function request" in answer
    runner.assert_called_once_with()


def test_empty_question_does_not_call_gemini_or_diagnostics():
    client = make_client()
    runner = Mock()

    answer = answer_question(
        "   ",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == "Please enter a non-empty question."
    client.models.generate_content.assert_not_called()
    runner.assert_not_called()


def test_candidate_with_missing_content_is_rejected():
    from types import SimpleNamespace

    from agent.main import _single_candidate

    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=None)]
    )

    try:
        _single_candidate(response)
    except ValueError as exc:
        assert "empty candidate response" in str(exc)
    else:
        raise AssertionError("Expected invalid candidate response")


def test_incomplete_function_call_is_rejected():
    from types import SimpleNamespace

    from agent.main import _get_function_call

    call = SimpleNamespace(
        name="run_diagnostics",
        args={},
        partial_args={"unfinished": True},
        will_continue=False,
    )
    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[SimpleNamespace(function_call=call)]
                )
            )
        ]
    )

    try:
        _get_function_call(response)
    except ValueError as exc:
        assert "incomplete function call" in str(exc)
    else:
        raise AssertionError("Expected incomplete function call rejection")


def test_empty_gemini_model_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "   ")
    client = make_client()
    runner = Mock()

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
        diagnostics_runner=runner,
    )

    assert answer == "Gemini is not configured: GEMINI_MODEL is empty."
    client.models.generate_content.assert_not_called()
    runner.assert_not_called()


def test_default_diagnostics_runner_is_used(monkeypatch):
    import importlib

    gemini_main = importlib.import_module("agent.main")
    runner = Mock(return_value=make_report())
    monkeypatch.setattr(gemini_main, "run_diagnostics", runner)

    client = make_client(
        make_function_response(),
        make_text_response("Diagnostics completed."),
    )

    answer = answer_question(
        "Check my MLOps environment.",
        client=client,
    )

    assert answer == "Diagnostics completed."
    runner.assert_called_once_with()


def test_main_uses_command_line_question(monkeypatch, capsys):
    import importlib

    gemini_main = importlib.import_module("agent.main")
    load_dotenv = Mock()
    answer = Mock(return_value="CLI response")
    configure_logging = Mock()

    monkeypatch.setattr(
        gemini_main.logging,
        "basicConfig",
        configure_logging,
    )
    monkeypatch.setattr(gemini_main, "load_dotenv", load_dotenv)
    monkeypatch.setattr(gemini_main, "answer_question", answer)
    monkeypatch.setattr(
        gemini_main.sys,
        "argv",
        ["agent", "Check", "deployment", "health"],
    )

    gemini_main.main()

    configure_logging.assert_called_once_with(
        level=gemini_main.logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_dotenv.assert_called_once_with()
    answer.assert_called_once_with("Check deployment health")
    assert capsys.readouterr().out == "CLI response\n"


def test_main_prompts_when_no_command_line_question(monkeypatch, capsys):
    import importlib

    gemini_main = importlib.import_module("agent.main")
    load_dotenv = Mock()
    answer = Mock(return_value="Interactive response")

    monkeypatch.setattr(gemini_main, "load_dotenv", load_dotenv)
    monkeypatch.setattr(gemini_main, "answer_question", answer)
    monkeypatch.setattr(gemini_main.sys, "argv", ["agent"])
    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "  Check deployment health  ",
    )

    gemini_main.main()

    load_dotenv.assert_called_once_with()
    answer.assert_called_once_with("Check deployment health")
    assert capsys.readouterr().out == "Interactive response\n"

def test_module_entry_point_executes_main(monkeypatch):
    import runpy
    import sys
    from pathlib import Path

    gemini_main = __import__("agent.main", fromlist=["main"])
    source_path = Path(gemini_main.__file__).resolve()

    monkeypatch.setattr(sys, "argv", ["agent/main.py"])
    monkeypatch.setattr("builtins.input", lambda _: "Summarize diagnostics")
    monkeypatch.setattr("dotenv.load_dotenv", lambda: False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    runpy.run_path(str(source_path), run_name="__main__")


def test_api_error_formatter_classifies_safe_http_statuses():
    from google.genai import errors

    cases = {
        400: "HTTP 400",
        401: "HTTP 401",
        403: "HTTP 403",
        404: "HTTP 404",
        429: "HTTP 429",
        503: "HTTP 503",
    }

    for status_code, expected in cases.items():
        error = errors.APIError(
            status_code,
            {"error": {"message": "sensitive simulated server response"}},
        )
        answer = _format_api_error(error)

        assert expected in answer
        assert "sensitive simulated server response" not in answer


def test_api_error_formatter_never_exposes_server_supplied_text():
    from google.genai import errors

    error = errors.APIError(
        403,
        {
            "error": {
                "status": "SECRET_STATUS_VALUE",
                "message": "SECRET_MESSAGE_VALUE",
                "details": "SECRET_DETAILS_VALUE",
            }
        },
    )

    answer = _format_api_error(error)

    assert "HTTP 403" in answer
    assert "SECRET_STATUS_VALUE" not in answer
    assert "SECRET_MESSAGE_VALUE" not in answer
    assert "SECRET_DETAILS_VALUE" not in answer


def test_api_error_formatter_rejects_invalid_status_codes():
    class InvalidStatusError(Exception):
        def __init__(self, code):
            self.code = code

    fallback = (
        "Gemini API request failed. Check the model configuration and try again."
    )

    for code in (None, True, False, 99, 600, "403", 403.0):
        assert _format_api_error(InvalidStatusError(code)) == fallback
