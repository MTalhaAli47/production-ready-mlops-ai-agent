# System Architecture

## Overview

The Production-Ready MLOps AI Agent is a read-only diagnostic application.
It combines deterministic Python diagnostic tools with a Gemini-powered
natural-language interface.

The system separates operational inspection from natural-language
interpretation:

- The diagnostic coordinator gathers evidence from the configured services.
- Each subsystem tool performs a bounded, read-only inspection.
- Pydantic models define the structure of subsystem results and the final report.
- Gemini receives the report and explains its findings to the user.
- Gemini is not given arbitrary shell access or infrastructure mutation tools.

## High-level component flow

    User
      |
      v
    Gemini interface (agent/main.py)
      |
      | validated run_diagnostics function request
      v
    Diagnostic coordinator (agent/orchestrator.py)
      |
      +--> MLflow tools
      |      +--> recent experiment runs
      |      +--> model selection and training provenance
      |
      +--> DVC tools
      |      +--> dataset state
      |      +--> local cache and remote status
      |
      +--> Airflow tools
      |      +--> health-check DAG history
      |
      +--> Kubernetes tools
      |      +--> deployment details
      |      +--> bounded pod inspection
      |
      +--> FastAPI
             +--> /health
             +--> /ready
      |
      v
    Structured diagnostic report (agent/schemas.py)
      |
      v
    Gemini explanation
      |
      v
    User

## Main components

### Natural-language interface

`agent/main.py` provides the command-line interface and Gemini integration.
The agent exposes one operational function: `run_diagnostics`.

The application validates the function request before invoking the
coordinator. It then serializes the structured diagnostic report and supplies
that report to Gemini for explanation.

The interface supports a command-line question and interactive use. A live
Gemini session requires the configured API key and network access.

### Diagnostic coordinator

`agent/orchestrator.py` coordinates the subsystem checks and aggregates
their results.

Its responsibilities include:

- Selecting an eligible recent MLflow run deterministically.
- Retrieving historical training provenance for the selected run.
- Checking DVC dataset and remote state.
- Inspecting Airflow health-check DAG history.
- Inspecting the configured Kubernetes deployment and its pods.
- Checking inference API health and readiness.
- Aggregating subsystem statuses, findings, and recommendations.

Subsystem failures are represented in the report rather than being treated
as proof that every other subsystem failed.

### Structured schemas

`agent/schemas.py` defines the Pydantic models used for diagnostic results.

The report contains an overall status, generated timestamp, subsystem
results, findings, and recommendations. Individual tool results carry a
status, structured data, sanitized error information, check timestamp, and
duration.

The schema constrains the shape of the report. It does not independently
verify that an external service is healthy; that evidence comes from the
corresponding diagnostic tool.

## Diagnostic data sources

| Subsystem | Source | Inspection |
|---|---|---|
| Model selection | MLflow tracking store | Recent experiment runs and eligible metrics |
| Training provenance | MLflow run metadata | Dataset path and recorded DVC/Git metadata |
| Dataset state | DVC | Working-tree and remote-cache status |
| Workflow | Airflow 3 | Health-check DAG run history |
| Deployment | Kubernetes | Deployment status and pod readiness |
| Inference | FastAPI | Health and model readiness endpoints |

### Model selection and provenance

The coordinator examines a bounded sample of recent experiment runs and
applies deterministic eligibility and ordering rules. The selection is based
on recorded run metadata and metrics; it is not a new model evaluation.

After selecting a run, the coordinator retrieves its historical training
provenance. This is important because the current dataset state is not
necessarily the same as the dataset used by a historical training run.

### Dataset checks

DVC checks provide evidence about the current dataset working state and
configured remote cache. They are separate from historical MLflow provenance.

A clean DVC status does not by itself establish that a model generalizes
well or that its training dataset was representative.

### Workflow checks

The Airflow integration inspects the configured health-check DAG history.
It does not trigger DAG runs. Historical success is evidence about a recorded
run, not proof that Airflow is currently available or that every production
workflow is healthy.

### Kubernetes checks

The Kubernetes integration inspects the configured namespace, deployment,
and bounded pod results. It does not create, patch, delete, or scale
Kubernetes resources.

Deployment replica information and pod readiness are reported as observed
state. A successful inspection is not a guarantee of future availability.

### Inference checks

The coordinator checks the FastAPI health and readiness endpoints.
Process health and model readiness are distinct signals: a running process
does not necessarily mean that the model is loaded and ready to serve
predictions.

## Diagnostic report interpretation

The report distinguishes subsystem status from the overall status.

- `success`: the subsystem check completed successfully.
- `partial`: the check returned incomplete or degraded information.
- `failed`: the check encountered a failure.
- `unavailable`: the required source could not provide a usable result.
- `skipped`: the check was intentionally not performed.

The overall report aggregates these results. A partial or unavailable
subsystem should be treated as missing evidence, not as proof that the
underlying service is broken.

Findings and recommendations summarize observed conditions. They should be
read alongside the source subsystem result and its timestamp.

## Trust and safety boundaries

The design intentionally limits the agent's operational authority:

1. Gemini is exposed to the diagnostic function, not arbitrary shell commands.
2. The coordinator invokes explicit diagnostic tools.
3. Diagnostic tools perform read-only inspection.
4. Kubernetes inspection is namespace-restricted and bounded.
5. Subsystem errors are sanitized before inclusion in diagnostic output.
6. Structured provenance is restricted to an explicit set of fields.
7. Retraining, model registration, Airflow triggers, and Kubernetes changes
   are outside the coordinator's permissions.

The Gemini explanation is a natural-language interpretation of the report.
The structured subsystem results remain the source of operational evidence.

## Important interpretation caveat

The demonstration training dataset contains only nine rows. Metrics recorded
for that dataset cannot establish model generalization or production
readiness. In particular, a high F1 score on this small dataset is not enough
to conclude that retraining is unnecessary.

Recommendations about retraining must consider dataset quality and size,
historical provenance, data changes, workflow history, and deployment health.
The diagnostic agent reports evidence; it does not make an automatic
retraining decision.
