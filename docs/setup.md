# Setup Guide

## Prerequisites

The project is developed for Linux and WSL 2 with Ubuntu.

Required or optional tools:

- Python 3.12 (`>=3.12,<3.13`).
- `uv` for Python environment and dependency management.
- Git.
- Docker Engine and Docker Compose for the optional Airflow environment.
- A Kubernetes environment such as Minikube for Kubernetes diagnostics.
- A Gemini API key and network access for live Gemini requests only.

## Obtain the repository

For a fresh checkout:

    git clone https://github.com/MTalhaAli47/production-ready-mlops-ai-agent.git
    cd production-ready-mlops-ai-agent

If you already have the repository, use its existing project directory.

## Create the Python environment

Create the environment only if it does not already exist:

    if [ ! -x .venv/bin/python ]; then
        uv venv --python 3.12 .venv
    fi

Install the project and development dependencies:

    uv pip install --python .venv/bin/python -e '.[dev]'

Activate the environment if desired:

    source .venv/bin/activate

Do not recreate an existing environment simply to reinstall dependencies.

## Validate the installation

Run Ruff:

    .venv/bin/ruff check .

Run the test suite:

    .venv/bin/pytest -q

The automated tests use controlled test doubles where appropriate and do
not require a live Gemini API request.

## Environment configuration

The repository includes `.env.example` with these Gemini placeholders:

- `GEMINI_API_KEY`
- `GEMINI_MODEL`

For live Gemini requests, configure the values in a local, ignored `.env`
file. Check whether `.env` already exists before creating or changing it.
Do not overwrite an existing file or symlink.

Never commit `.env`, paste credentials into source code, or expose API keys
in screenshots, logs, terminal output, or support messages. Keep
`.env.example` limited to placeholders.

The project can be tested without a live Gemini API request.

## MLflow and DVC

MLflow provides experiment tracking and historical run provenance.
DVC provides dataset versioning and remote-state inspection.

The configured MLflow tracking store and DVC remote must be available for
live diagnostics. Check their availability in the environment where the
application is running.

The demonstration dataset is small. Its metrics demonstrate experiment
tracking; they do not establish production model quality.

## Optional Airflow environment

The repository includes a Docker Compose development environment for
Airflow 3 and PostgreSQL.

Airflow requires database and authentication-related configuration.
For a fresh environment, review the Compose configuration and provide the
required values through a local, protected environment file.

Do not overwrite an existing `.env` file or symlink. Do not remove Docker
volumes during troubleshooting: persistent volumes may contain database
state.

The configured health-check DAG is `mlops_a03_healthcheck`.

The diagnostic agent inspects DAG history; it does not trigger DAG runs.
The Compose environment is intended for development and demonstration,
not production-hardened deployment.

## Optional Kubernetes environment

Kubernetes diagnostics require access to a configured cluster and the
expected namespace: `mlops-a03`.

The diagnostic integration inspects deployment and pod state. It does not
apply manifests or mutate cluster resources.

Do not apply, delete, scale, or otherwise modify cluster resources solely
to make a diagnostic check pass.

## Inference API

The FastAPI application exposes:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Process health |
| GET | `/ready` | Model readiness |
| POST | `/predict` | Student pass prediction |

The prediction input contains integer values for `hours_studied`,
`attendance`, and `previous_score`. Additional fields are rejected.

The API expects its model artifact and metadata in `api/model/`. Refer to
the repository deployment configuration before starting the API or
building its container image.

## Running diagnostics

Use the CLI instructions in the root README to submit a diagnostic
question or start interactive mode.

A useful end-to-end question is:

> Find the best recent model, identify the dataset version used to train
> it, check deployment status, and explain whether retraining is necessary.

The answer depends on which services are available at the time of the
check. Historical success does not guarantee current availability.

## Troubleshooting principles

- Confirm the active Python interpreter is `.venv/bin/python`.
- Confirm required services are running before interpreting an unavailable
  result as an application defect.
- Check service configuration locally without printing credentials.
- Distinguish process health from model readiness.
- Treat missing provenance as missing evidence; do not invent a dataset
  version.
- Preserve existing Docker volumes and Kubernetes resources during
  diagnosis.
- Consult the root README and architecture guide for component behavior
  and safety boundaries.
