# Production-Ready MLOps AI Agent

A read-only, natural-language MLOps diagnostic agent that brings together
dataset versioning, experiment tracking, workflow orchestration, Kubernetes
monitoring, and model-serving health checks.

The project combines deterministic diagnostic tools with a Gemini-powered
interface. The coordinator gathers operational evidence; Gemini explains the
result without independently executing infrastructure operations.

## Project objectives

- Track datasets and historical training provenance with DVC and MLflow.
- Inspect experiment runs and select an eligible recent model deterministically.
- Monitor an Airflow health-check DAG.
- Inspect Kubernetes deployment and pod status.
- Check FastAPI inference health and readiness.
- Present consolidated diagnostics through a constrained Gemini agent.
- Keep diagnostic operations read-only and validate behavior with automated tests
  and GitHub Actions.

## Architecture

    User
      |
      v
    Gemini natural-language interface
      |
      | validated run_diagnostics tool request
      v
    Read-only diagnostic coordinator
      |
      +--> MLflow: recent model and training provenance
      +--> DVC: dataset and remote-cache state
      +--> Airflow: health-check DAG history
      +--> Kubernetes: deployment and pod status
      +--> FastAPI: health and readiness
      |
      v
    Structured diagnostic report
      |
      v
    Gemini explanation

The coordinator is the source of operational facts. Gemini explains the
structured report; it is not given arbitrary shell access or infrastructure
mutation tools.

## Technology stack

| Area | Technology |
|---|---|
| Language | Python 3.12 |
| Dataset versioning | DVC |
| Experiment tracking | MLflow |
| Workflow orchestration | Apache Airflow 3 |
| Workflow database | PostgreSQL |
| Containerization | Docker |
| Orchestration | Kubernetes |
| Local Kubernetes environment | Minikube |
| Inference API | FastAPI |
| Agent interface | Google Gemini SDK |
| Validation | pytest, pytest-cov, Ruff |
| Continuous integration | GitHub Actions |

## Repository structure

    agent/
      main.py                 Gemini agent and CLI
      orchestrator.py         Read-only diagnostic coordinator
      schemas.py              Structured diagnostic models
      tools/
        dvc_tools.py
        mlflow_tools.py
        airflow_tools.py
        kubernetes_tools.py
    api/
      main.py                 Student prediction API
    airflow/
      dags/
        mlops_a03_healthcheck.py
    kubernetes/
      namespace.yaml
      deployment.yaml
      service.yaml
    tests/
    .github/
      workflows/
        ci.yml
    Dockerfile
    docker-compose.yml
    .env.example
    pyproject.toml
    requirements.txt

## Prerequisites

- WSL 2 with Ubuntu.
- Python 3.12.
- Git.
- Docker Engine and Docker Compose for the Airflow environment.
- A local Kubernetes environment such as Minikube for Kubernetes checks.
- A Gemini API key for live natural-language demonstrations.

The automated test suite is designed to use mocks for external services; a
Gemini API key and live infrastructure are not required for CI tests.

## Installation

From the project directory, create and activate a virtual environment if one
does not already exist:

    if [ ! -d .venv ]; then
        python3.12 -m venv .venv
    fi
    source .venv/bin/activate

Install the project and development dependencies:

    uv pip install --python .venv/bin/python -e '.[dev]'

Run static checks and tests:

    .venv/bin/ruff check .
    .venv/bin/pytest -q

Use the existing environment where available. Do not recreate an established
virtual environment merely to repeat installation.

## Configuration

The repository includes `.env.example` as a configuration template.
For a live Gemini session, create a local `.env` file and configure:

    GEMINI_API_KEY=your_gemini_api_key
    GEMINI_MODEL=gemini-3.5-flash-lite

Keep real credentials in the local ignored `.env` file. Never commit API keys,
passwords, tokens, kubeconfig files, or other secrets. Do not put real secrets
in `.env.example`.

Infrastructure-specific configuration is described in the project setup
documentation as it is completed.

## Running the agent

The command-line interface accepts a question as an argument:

    .venv/bin/python -m agent.main "Run the MLOps diagnostics and summarize the result."

It can also be started interactively:

    .venv/bin/python -m agent.main

The agent requests the diagnostic tool, validates the request, runs the
coordinator, and provides the resulting structured report to Gemini for
explanation. A live Gemini API key is required for this interaction.

## Inference API

The FastAPI application exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Process health |
| GET | `/ready` | Model and service readiness |
| POST | `/predict` | Student pass prediction |

The prediction request uses integer fields `hours_studied`, `attendance`,
and `previous_score`. Additional fields are rejected.

The API expects its model artifact and metadata under `api/model/`. Follow the
deployment configuration and setup instructions before starting the service.

## Diagnostic subsystems

The coordinator reports results for six areas:

1. Model selection.
2. Historical training provenance.
3. DVC dataset and remote state.
4. Airflow DAG history.
5. Kubernetes deployment and pod health.
6. FastAPI inference health and readiness.

Each subsystem result is represented using a structured status, data, error
information, check timestamp, and duration. The overall report aggregates
findings and recommendations.

## Safety boundaries

- Diagnostic operations are read-only.
- Kubernetes inspection is namespace-restricted and bounded.
- No arbitrary shell execution is exposed to Gemini.
- The agent exposes only the `run_diagnostics` function.
- Gemini receives the diagnostic report as the source of operational facts.
- Credentials and raw exception messages are not included in diagnostic logs.
- Model registration, retraining, Airflow triggers, and Kubernetes updates are
  outside the coordinator's permissions.

## Testing and continuous integration

The test suite covers diagnostic tools, coordinator behavior, Gemini request
handling, failure paths, and safety controls.

GitHub Actions runs on pushes to `main` and pull requests targeting `main`.
The CI workflow installs project dependencies, runs Ruff, and executes pytest.

The workflow has completed a successful GitHub Actions run:

[View the successful CI run](https://github.com/MTalhaAli47/production-ready-mlops-ai-agent/actions/runs/43703355197)

## Important limitations

- The demonstration dataset contains only nine rows. It is not sufficient to
  establish model generalization or production quality.
- A high F1 score on this small dataset is not evidence that retraining is
  unnecessary or that the model is production-ready.
- Model selection is limited to a sample of at most 100 recent experiment runs.
- Live diagnostics depend on the relevant services being available and correctly
  configured.
- A successful historical infrastructure check does not guarantee that the
  services are currently running.
- The Gemini interface requires a configured API key and network access.
- The Airflow Docker Compose environment is intended for development and
  demonstration, not production-hardened deployment.

## Project status

Phases 1–7 have been implemented and verified, including automated tests,
infrastructure checks where applicable, and a successful GitHub Actions run.
Documentation and demonstration evidence are being completed in Phase 8.

## License

No license has been specified yet. All rights reserved unless a license is
added to this repository.

## Additional documentation

- [System architecture](docs/architecture.md)
- [Setup guide](docs/setup.md)
- [Agent design and safety](docs/agent-and-safety.md)
- [Demo scenarios](docs/demo-scenarios.md)
- [Known limitations](docs/known-limitations.md)
