"""Gemini-powered natural-language interface for read-only MLOps diagnostics."""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from agent.orchestrator import run_diagnostics
from agent.schemas import DiagnosticReport

DEFAULT_MODEL = "gemini-3.5-flash-lite"
REQUEST_TIMEOUT_MS = 15_000
MAX_OUTPUT_TOKENS = 2_048

SYSTEM_INSTRUCTION = """
You are the natural-language interface for a read-only MLOps diagnostic agent.

You have exactly one available operation: run_diagnostics.
Call it once to obtain the current diagnostic report before answering.
Do not claim to have checked anything that is absent from that report.
Treat the report as the sole source of operational facts.
Clearly distinguish confirmed facts, unavailable information, failures,
inferences, and recommendations.
Never claim that a model is production-ready based only on its metrics.
A small training dataset is a limitation when assessing generalization.
Do not recommend or claim that you performed training, model registration,
deployment changes, Airflow triggers, or other state-changing operations.
Never invent dataset versions, run IDs, health states, or test results.
Keep the explanation clear and concise.
""".strip()


def _diagnostic_tool():
    """Declare the only function Gemini is permitted to request."""
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="run_diagnostics",
                description=(
                    "Run the deterministic, read-only MLOps diagnostic "
                    "coordinator. Takes no arguments."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                    required=[],
                ),
            )
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
    if call.name != "run_diagnostics":
        raise ValueError("Gemini requested an unsupported function")

    if call.args is None or dict(call.args) != {}:
        raise ValueError("Gemini supplied invalid diagnostic arguments")

    if call.partial_args or call.will_continue:
        raise ValueError("Gemini returned an incomplete function call")

    return candidate.content


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


def answer_question(question, client=None, diagnostics_runner=None):
    """Answer a question using one validated, read-only diagnostic call."""
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

    if diagnostics_runner is None:
        diagnostics_runner = run_diagnostics

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        tools=[_diagnostic_tool()],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY",
                allowed_function_names=["run_diagnostics"],
            )
        ),
    )

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
        model_content = _get_function_call(first_response)

        report = diagnostics_runner()
        if not isinstance(report, DiagnosticReport):
            return "The diagnostic coordinator returned an invalid report."

        function_response = types.Content(
            role="user",
            parts=[
                types.Part.from_function_response(
                    name="run_diagnostics",
                    response={"report": report.model_dump(mode="json")},
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

    except errors.APIError:
        return "Gemini API request failed. Check the model configuration and try again."
    except (TimeoutError, OSError):
        return "Gemini request timed out or could not connect. Please try again."
    except (TypeError, ValueError):
        return (
            "Gemini returned an invalid response or function request. "
            "No additional diagnostic operation was executed."
        )
    except RuntimeError:
        return "The diagnostic operation could not be completed."


def main():
    load_dotenv()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Ask about your MLOps environment: ").strip()

    print(answer_question(question))


if __name__ == "__main__":
    main()
