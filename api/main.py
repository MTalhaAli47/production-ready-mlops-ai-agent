import json
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import skops.io as sio
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, StrictInt

API_DIR = Path(__file__).resolve().parent
MODEL_DIR = API_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.skops"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURES = [
    "hours_studied",
    "attendance",
    "previous_score",
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hours_studied: StrictInt
    attendance: StrictInt
    previous_score: StrictInt


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: dict[str, float]
    model_id: str
    source_run_id: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact not found: {MODEL_PATH}")

    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Model metadata not found: {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if metadata.get("features") != FEATURES:
        raise RuntimeError("Model metadata feature order does not match API schema")

    # Load only explicitly trusted types. The inspected artifact has none.
    model = sio.load(str(MODEL_PATH), trusted=[])

    if list(model.feature_names_in_) != FEATURES:
        raise RuntimeError("Loaded model feature names do not match API schema")

    if list(model.classes_) != [0, 1]:
        raise RuntimeError("Expected classifier classes [0, 1]")

    app.state.model = model
    app.state.model_metadata = metadata

    yield

    app.state.model = None
    app.state.model_metadata = None


app = FastAPI(
    title="Student Pass Prediction API",
    description="Inference API for the MLOps Assignment 03 trained model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def readiness(request: Request):
    if getattr(request.app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    model = getattr(request.app.state, "model", None)
    metadata = getattr(request.app.state, "model_metadata", None)

    if model is None or metadata is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = pd.DataFrame(
        [[
            payload.hours_studied,
            payload.attendance,
            payload.previous_score,
        ]],
        columns=FEATURES,
    )

    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]

    return PredictionResponse(
        prediction=prediction,
        probabilities={
            "0": float(probabilities[0]),
            "1": float(probabilities[1]),
        },
        model_id=metadata["model_id"],
        source_run_id=metadata["source_run_id"],
    )
