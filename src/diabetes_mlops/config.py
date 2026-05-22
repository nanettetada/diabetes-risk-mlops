"""Project-wide configuration. Pure constants, no side effects on import."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

RAW_CSV = RAW_DIR / "diabetes.csv"
PROCESSED_CSV = PROCESSED_DIR / "diabetes_clean.csv"
MODEL_PATH = MODELS_DIR / "diabetes_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Public mirror of the Pima Indians Diabetes dataset (CSV with header row).
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
)
COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
FEATURES = COLUMNS[:-1]
TARGET = "Outcome"

# Columns where a value of 0 is physiologically impossible and means "missing".
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Clinician-specified minimum recall for the positive class. Threshold is
# tuned downward from 0.5 until this is met on cross-validated predictions.
TARGET_RECALL = 0.85

MLFLOW_EXPERIMENT = "diabetes-risk"
