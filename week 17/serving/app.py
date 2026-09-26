from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).parents[1]
MODEL_URI = os.getenv("MODEL_URI", "models:/TelcoChurnModel/Production")
mlflow.set_tracking_uri((ROOT / "artifacts" / "mlruns").as_uri())
app = FastAPI(title="Telco Churn Model API", version="1.0.0")
model: Any = None


def get_model() -> Any:
    global model
    if model is None:
        model = mlflow.pyfunc.load_model(MODEL_URI)
    return model


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": MODEL_URI}


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        prediction = get_model().predict(pd.DataFrame([payload]))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid feature payload: {exc}") from exc
    return {"churn": int(prediction[0]), "label": "Yes" if int(prediction[0]) else "No"}

