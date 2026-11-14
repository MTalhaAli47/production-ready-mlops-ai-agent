import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_readiness_endpoint(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_prediction_returns_expected_schema(client):
    response = client.post(
        "/predict",
        json={
            "hours_studied": 4,
            "attendance": 70,
            "previous_score": 55,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in (0, 1)
    assert set(body["probabilities"]) == {"0", "1"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["model_id"] == (
        "m-c826a253fa7342ed9a813ef6745be891"
    )
    assert body["source_run_id"] == (
        "f813a10421fc4052902b876891e4b44a"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "hours_studied": 4,
            "attendance": 70,
        },
        {
            "hours_studied": 4,
            "attendance": 70,
            "previous_score": 55,
            "unexpected": 1,
        },
        {
            "hours_studied": True,
            "attendance": 70,
            "previous_score": 55,
        },
        {
            "hours_studied": 4.5,
            "attendance": 70,
            "previous_score": 55,
        },
        {
            "hours_studied": "4",
            "attendance": 70,
            "previous_score": 55,
        },
    ],
)
def test_prediction_rejects_invalid_payloads(client, payload):
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_prediction_rejects_null_values(client):
    response = client.post(
        "/predict",
        json={
            "hours_studied": 4,
            "attendance": None,
            "previous_score": 55,
        },
    )

    assert response.status_code == 422
