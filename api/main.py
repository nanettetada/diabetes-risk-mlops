"""FastAPI service exposing /predict and /health.

Run with:  uvicorn api.main:app --reload
Docs at:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from diabetes_mlops.predict import load_artifact, predict_one  # noqa: E402

app = FastAPI(
    title="Diabetes Risk API",
    version="0.1.0",
    description="Screening-triage probability for type-2 diabetes from routine clinical measurements.",
)


class PatientPayload(BaseModel):
    Pregnancies: int = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=400, description="Plasma glucose, 2-h OGTT (mg/dL)")
    BloodPressure: float = Field(..., ge=0, le=200, description="Diastolic BP (mm Hg)")
    SkinThickness: float = Field(..., ge=0, le=100, description="Triceps skin fold (mm)")
    Insulin: float = Field(..., ge=0, le=1500, description="2-h serum insulin (µU/mL)")
    BMI: float = Field(..., ge=0, le=80)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3.0)
    Age: int = Field(..., ge=0, le=120)


class PredictionResponse(BaseModel):
    probability: float
    label: int
    threshold: float
    model: str


@app.get("/health")
def health() -> dict:
    try:
        art = load_artifact()
        return {"status": "ok", "model": art["model_name"]}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientPayload) -> dict:
    try:
        return predict_one(payload.model_dump())
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
