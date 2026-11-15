# Known Limitations

## Dataset size and model evaluation

The demonstration training dataset contains only nine rows.

Recorded evaluation metrics demonstrate experiment tracking and diagnostic
reporting. They are not sufficient evidence of model generalization,
robustness, fairness, or production readiness.

A high F1 score on this dataset must not be interpreted as proof that the
model will perform well on unseen or real-world data.

A larger, representative dataset and a more rigorous evaluation process are
required before making production model-quality claims.

## MLflow model selection

The coordinator examines a bounded set of recent MLflow runs rather than
performing an exhaustive search across the entire tracking history.

Model selection uses deterministic eligibility and ordering rules. It
depends on the metrics and metadata recorded in MLflow.

The selected run is a diagnostic candidate, not a newly trained or
independently validated model.

Historical training provenance may be incomplete. When required metadata
is missing or malformed, the agent cannot reliably reconstruct the
historical dataset version.

## DVC and dataset provenance

DVC checks report the dataset and remote-cache state visible to the
configured environment at diagnostic time.

Current DVC state and historical MLflow provenance answer different
questions. A clean working tree or remote cache does not independently
prove that a historical model was trained using the current dataset.

The quality and completeness of provenance depend on the metadata recorded
with the original training run.

## Airflow environment

The included Airflow Docker Compose environment is intended for local
development and demonstration. It is not a production-hardened Airflow
deployment.

The diagnostic integration inspects health-check DAG history. It does not
trigger workflows or prove that every scheduled or production workflow is
operating correctly.

Historical DAG success does not guarantee current Airflow availability.

## Kubernetes environment

Kubernetes diagnostics depend on access to the configured cluster,
credentials, namespace, and resources.

The integration performs bounded, read-only inspection. It does not
provision infrastructure, repair workloads, scale deployments, or change
cluster resources.

A diagnostic result describes the observed deployment and pod state at
the time of the check. It cannot guarantee future availability.

## Inference API checks

The agent checks the configured FastAPI health and readiness endpoints.
These checks provide operational signals, not a comprehensive performance,
load, security, or end-to-end prediction-quality assessment.

Process health and model readiness are separate conditions. A healthy
process does not necessarily mean the model is ready to serve predictions.

## Gemini dependency

Live natural-language interaction requires:

- A valid Gemini API key configured locally.
- Network access to the Gemini service.
- Availability of the configured Gemini model and API.

Without these prerequisites, live Gemini interaction cannot be completed.
Automated tests use mocked interactions and do not establish live API
availability or validate real-world model responses.

Gemini explanations are constrained to the structured diagnostic report,
but natural-language output should still be reviewed against the underlying
diagnostic evidence.

## Diagnostic availability and timing

Subsystem checks depend on external services and local infrastructure.
A service may be unavailable, unreachable, misconfigured, or slow when a
diagnostic is run.

Timeouts, bounded result retrieval, and sanitized error handling improve
predictability but can limit the amount of evidence collected.

An unavailable result means the diagnostic could not obtain usable evidence.
It does not necessarily mean the underlying service is permanently broken.

## Read-only scope

The agent is an inspection and explanation tool, not an automated
infrastructure remediation platform.

It does not:

- Retrain models.
- Register or promote models.
- Trigger Airflow DAG runs.
- Create, patch, delete, or scale Kubernetes resources.
- Execute arbitrary shell commands through Gemini.
- Automatically decide that retraining is required or unnecessary.

Any administrative or remediation action requires a separate, explicitly
authorized workflow.

## Logging and observability

Structured diagnostic logs provide request correlation, subsystem identity,
status, and duration.

They intentionally exclude report contents, model data, raw exception
messages, URLs, credentials, and API keys. This protects sensitive
information but means detailed diagnosis may require reviewing the
corresponding service through its own authorized logging and monitoring
facilities.

## Development and deployment maturity

The project integrates several MLOps components for demonstration and
diagnostic purposes. Successful local tests and service checks do not
constitute a production deployment certification.

Production adoption would require environment-specific security review,
credential management, infrastructure hardening, monitoring, access
control, operational procedures, representative model evaluation, and
reliability testing.

## Gemini independent dispatcher limitations

The current Gemini dispatcher exposes a deliberately limited set of 13
read-only functions. It supports focused DVC, MLflow, Airflow, and Kubernetes
inspection, as well as the cross-subsystem `run_diagnostics` coordinator.
It does not provide arbitrary tool discovery or execution.

The independent dispatcher has been tested locally with mocked Gemini
responses and tool implementations. Its function allowlist, configuration,
argument validation, rejection paths, and selected-only dispatch are covered
by automated tests. **Independent tool selection has not yet been
live-verified against the Gemini API.** Existing coordinator-based live demos
do not establish that the new independent selections work end to end.

Live operation requires a valid locally configured API key, network access,
and availability of the configured Gemini model. Gemini's natural-language
explanation should be checked against the underlying tool evidence; it is not
a substitute for the returned operational facts.

The exposed tool set intentionally excludes state-changing operations,
sensitive task/pod log retrieval, and arbitrary commands. Kubernetes
inspection is restricted to the approved `mlops-a03` namespace. Local
argument validation, result sanitization, and output-size bounds reduce risk
but do not make the model's explanation infallible.

The training dataset still contains only nine rows. Metrics from this dataset
are limited evidence and do not establish reliable generalization or
production readiness.
