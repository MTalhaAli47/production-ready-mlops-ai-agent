# Agent Demo Scenarios

## Purpose

These scenarios demonstrate the MLOps AI Agent's ability to collect and
explain operational evidence across model tracking, dataset versioning,
workflow orchestration, deployment, and inference health.

The scenarios are prompts for the agent, not claims that every service is
currently running. Results depend on the state of the configured services
when diagnostics execute.

## Scenario 1 — End-to-end MLOps health assessment

**Prompt**

> Find the best recent model, identify the dataset version used to train it,
> check deployment status, and explain whether retraining is necessary.

**Evidence to examine**

- MLflow: eligible recent runs and selected model metrics.
- MLflow: historical dataset and training provenance for the selected run.
- DVC: current dataset working-tree and remote-cache state.
- Airflow: `mlops_a03_healthcheck` DAG history.
- Kubernetes: `inference-api` deployment and pod readiness in `mlops-a03`.
- FastAPI: `/health` and `/ready` responses.

**Expected explanation**

The agent should connect the selected model to its historical training
provenance, distinguish that historical dataset from the current DVC state,
and report workflow, deployment, and inference evidence separately.

It should distinguish observed facts, unavailable information, and
recommendations. It must not conclude that retraining is unnecessary solely
because a recorded metric is high.

## Scenario 2 — Model selection and training provenance

**Prompt**

> Which recent MLflow run is the best eligible candidate, what metrics support
> that selection, and which dataset version was recorded for its training?

**Evidence to examine**

- Recent runs in the configured MLflow experiment.
- Run status and metric eligibility.
- Deterministic model-selection ordering.
- Historical provenance fields recorded for the selected run.

**Expected explanation**

The agent should identify the selected run and explain the recorded metric
and provenance fields. It should disclose missing or incomplete provenance
rather than infer a dataset version from the current working tree.

Model selection is based on recorded experiment data; it is not a fresh
evaluation or proof of generalization.

## Scenario 3 — Dataset state and training-data consistency

**Prompt**

> Check whether the current DVC dataset state is consistent with the dataset
> recorded for the selected MLflow run. Explain any differences.

**Evidence to examine**

- Historical MLflow dataset path, Git revision, DVC hash, metadata path,
  and dataset version, where available.
- Current DVC working-tree status.
- Current DVC remote-cache status.

**Expected explanation**

The agent should compare only available, relevant identifiers and state
clearly when a comparison cannot be established. A clean DVC status does
not prove that the current dataset is identical to a historical training
dataset unless the recorded identifiers support that conclusion.

## Scenario 4 — Workflow and deployment readiness

**Prompt**

> Review the latest available Airflow health-check history and the Kubernetes
> inference deployment. Are there signs of an operational problem?

**Evidence to examine**

- Airflow health-check DAG run history and recorded states.
- Kubernetes deployment replica and condition information.
- Bounded pod phase and readiness information.
- The status and timestamp of each subsystem check.

**Expected explanation**

The agent should separate historical workflow results from current
deployment observations. It should identify partial or unavailable evidence
and avoid treating an old successful DAG run as proof of current health.

The diagnostic operation must not trigger a DAG run or modify Kubernetes
resources.

## Scenario 5 — Inference API health and retraining considerations

**Prompt**

> Check inference API health and model readiness. Considering the available
> model, dataset, and workflow evidence, what should be investigated before
> deciding whether to retrain?

**Evidence to examine**

- FastAPI `/health` process-health result.
- FastAPI `/ready` model-readiness result.
- MLflow metrics and historical provenance.
- DVC dataset state.
- Airflow workflow history.
- Kubernetes deployment and pod observations.

**Expected explanation**

The agent should distinguish process health from model readiness and
deployment availability. Its recommendations should be tied to observed
evidence and explicitly identify missing evidence.

The demonstration dataset contains only nine rows. Its recorded metrics
cannot establish generalization or production readiness. The agent must
not make an automatic retraining decision based on those metrics alone.

## Demonstration procedure

1. Confirm which required services are intentionally available for the demo.
2. Start the application using the CLI instructions in the root README.
3. Submit the scenario prompts individually.
4. Preserve the returned diagnostic evidence and the agent's explanation.
5. Record unavailable services as unavailable; do not alter infrastructure
   just to force a successful result.
6. Review whether the explanation distinguishes facts, missing evidence,
   and recommendations.
7. Capture screenshots only after checking that they contain no credentials,
   API keys, private configuration, or other sensitive information.

## Evidence and evaluation checklist

- [ ] The agent selects a recent eligible model deterministically.
- [ ] The selected model is connected to its recorded training provenance.
- [ ] Historical provenance is distinguished from current DVC state.
- [ ] Airflow history is reported as historical workflow evidence.
- [ ] Kubernetes deployment and pod observations are reported separately.
- [ ] Inference process health is distinguished from model readiness.
- [ ] Missing or unavailable evidence is not invented.
- [ ] Recommendations are supported by the diagnostic report.
- [ ] No mutation or retraining action is initiated by the diagnostic agent.
- [ ] The nine-row dataset limitation is disclosed when discussing model quality.

## Independent Gemini tool-selection scenarios

The dispatcher adds focused read-only scenarios alongside the existing
coordinator-based Gemini demonstrations.

| Area | Example request | Expected selection |
| --- | --- | --- |
| DVC | List available datasets | `list_datasets` |
| DVC | Show provenance for the training dataset | `get_dataset_provenance` |
| DVC | Check whether the dataset has local or remote changes | `check_dataset_changes` |
| MLflow | List experiments | `list_experiments` |
| MLflow | List runs for an experiment | `list_runs` |
| MLflow | Inspect a specific run | `get_run_details` |
| Airflow | List DAGs | `list_dags` |
| Airflow | Inspect a DAG or its runs | `get_dag_details` or `list_dag_runs` |
| Kubernetes | List pods in the approved namespace | `list_pods` |
| Kubernetes | Inspect a pod or list deployments | `get_pod_details` or `list_deployments` |
| Cross-subsystem | Summarize the overall MLOps environment | `run_diagnostics` |

Requests to trigger DAGs, reproduce datasets, mutate Kubernetes resources,
retrieve sensitive logs, or perform other state-changing operations must not
be dispatched.

### Verification status

The independent dispatcher and its configuration are covered by local
automated tests using mocks. These tests verify the fixed allowlist, function
selection and argument validation, selected-only dispatch, rejection of
invalid calls, and exclusion of representative mutating or sensitive tools.

These scenarios are not yet evidence of live independent tool selection
through the Gemini API. Mark them as **automated/local test coverage** until a
live demonstration has been explicitly approved and completed.

The existing five coordinator-based Gemini demonstrations remain the
previously captured live evidence. They do not, by themselves, demonstrate
independent selection of the newly exposed functions.
