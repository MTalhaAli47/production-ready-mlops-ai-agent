# Agent Design and Safety

## Overview

The MLOps AI Agent combines a deterministic diagnostic coordinator with a
Gemini-powered natural-language interface.

The coordinator is the source of operational facts. Gemini interprets and
explains the structured diagnostic report; it does not independently inspect
infrastructure or execute operational commands.

## Gemini interaction flow

The application exposes one operational function to Gemini:

`run_diagnostics`

The interaction follows a controlled two-stage process:

1. Gemini receives the user's diagnostic question and the function definition.
2. The application validates the requested function call.
3. The diagnostic coordinator executes once.
4. The application serializes the structured report using
   `model_dump(mode="json")`.
5. Gemini receives the report and prepares a natural-language explanation.
6. The final Gemini request has no tools available.

This prevents the final explanation stage from initiating another diagnostic
execution.

The system instruction requires Gemini to base operational claims on the
provided diagnostic report. The configured generation temperature is zero.

## Coordinator boundary

`agent/orchestrator.py` coordinates the diagnostic subsystems:

- MLflow model selection and historical training provenance.
- DVC dataset and remote-state checks.
- Airflow health-check DAG history.
- Kubernetes deployment and pod inspection.
- FastAPI health and readiness checks.

The coordinator aggregates subsystem results, findings, and recommendations
into the structured diagnostic report.

It does not retrain models, register models, trigger Airflow DAGs, or modify
Kubernetes resources.

## Read-only tool policy

The agent is designed for inspection, not infrastructure administration.

- Gemini is not given arbitrary shell access.
- Only the `run_diagnostics` function is exposed.
- Subsystem tools perform explicit diagnostic operations.
- Kubernetes inspection is restricted to the configured namespace and uses
  bounded result retrieval.
- External service checks use bounded timeouts where configured.
- No diagnostic operation is intended to mutate infrastructure or model state.

Any separately authorized administrative operation must be performed outside
the diagnostic coordinator.

## Structured results and trust boundaries

Pydantic schemas in `agent/schemas.py` constrain the structure of diagnostic
results.

Subsystem statuses distinguish successful checks from incomplete,
failed, unavailable, and skipped checks. Missing evidence must not be
presented as a confirmed failure or success.

The report is the evidence supplied to Gemini. Natural-language explanations
must preserve distinctions between observed facts, missing information,
and recommendations.

## Error handling and sanitization

The coordinator handles expected operational exceptions without exposing
uncontrolled exception messages.

- Failed or malformed tool responses are converted into controlled results.
- Raw exception messages are not included in diagnostic output.
- MLflow provenance is restricted to an explicit set of recognized fields.
- Provenance fields must contain nonempty strings.
- Provenance completeness is recalculated from accepted fields.
- Malformed Airflow history produces a controlled partial result.
- Nested DVC and Kubernetes failure details are not exposed indiscriminately.
- Subsystem failures are represented independently so one failure does not
  automatically erase evidence from other checks.

The implementation deliberately avoids a blanket `Exception` handler.

## Structured logging

Each diagnostic run receives a UUID correlation identifier.

The coordinator logs each of its six normalized subsystem results once,
after collection, using the message:

`Diagnostic subsystem completed.`

Structured log fields include:

- `request_id`
- `tool_name`
- `status`
- `duration_ms`

Diagnostic logs do not include report contents, model data, raw exception
messages, URLs, credentials, or API keys.

CLI INFO logging is configured inside the application entry point rather
than as an import-time side effect.

## Credential handling

The Gemini API key is read from local environment configuration.

- Keep the actual `.env` file local and ignored by Git.
- Keep `.env.example` limited to placeholders.
- Never hard-code or commit API keys.
- Do not expose credentials in terminal output, logs, screenshots, or
  support messages.
- Automated tests use mocked Gemini interactions and do not require a
  live API key.

## Operational interpretation

A successful diagnostic response describes the evidence available at the
time of inspection. It does not guarantee future service availability.

A missing or unavailable subsystem is missing evidence, not proof that
the subsystem is permanently broken.

Likewise, a high model metric on the nine-row demonstration dataset does
not establish generalization or production readiness. Gemini must not
interpret that metric alone as evidence that retraining is unnecessary.

## Current Gemini dispatcher: independent read-only tools

The current Gemini integration exposes a fixed allowlist of 13 read-only
functions. It supports both focused inspection and the cross-subsystem
coordinator:

- Coordinator: `run_diagnostics`.
- DVC: `list_datasets`, `get_dataset_provenance`,
  `check_dataset_changes`.
- MLflow: `list_experiments`, `list_runs`, `get_run_details`.
- Airflow: `list_dags`, `get_dag_details`, `list_dag_runs`.
- Kubernetes: `list_pods`, `get_pod_details`, `list_deployments`.

The earlier coordinator-only description documents the original implementation
and is superseded by this dispatcher design.

### Function selection and execution boundary

Gemini is configured with function-calling mode `ANY` and the explicit
13-function allowlist. Each request must produce exactly one approved function
call. The application rejects zero, multiple, unknown, malformed, or incomplete
calls.

The application independently validates arguments before dispatch. Validation
covers required and unknown fields, types, bounds, identifier formats,
approved fixed values, and conflicting selectors. SDK schemas are an
additional interface constraint, not a security boundary.

Only the selected, locally registered callable is executed. The dispatcher
does not evaluate model-generated code, import model-selected modules, execute
shell commands, or accept arbitrary operations. The coordinator remains
available for questions requiring a cross-subsystem diagnostic overview; it
is no longer the only way to retrieve operational facts.

Tool results are sanitized before being returned to Gemini. Sensitive
dictionary keys are redacted, nested collections and strings are bounded, and
serialized tool results are limited to 30,000 characters. The final
natural-language explanation is generated after the tool response, without
exposing another tool-selection opportunity.

### Scope exclusions

The allowlist deliberately excludes state-changing operations, including
Airflow DAG triggering, DVC reproduction, Kubernetes mutations, and MLflow
registry or deletion operations. Sensitive task and pod log retrieval is also
excluded. Kubernetes access is restricted to the approved `mlops-a03`
namespace.

Existing credential-handling and logging restrictions remain in force:
credentials and API keys must remain local and must not appear in logs,
screenshots, reports, or source control.
