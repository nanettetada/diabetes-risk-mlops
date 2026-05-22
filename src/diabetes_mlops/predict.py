"""Inference helpers — loaded by both the Streamlit app and the FastAPI service."""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from . import config as C


@lru_cache(maxsize=1)
def load_artifact(path: Path = C.MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Run `python -m diabetes_mlops.train` first."
        )
    return joblib.load(path)


def predict_one(payload: Mapping[str, float]) -> dict:
    """Score a single patient.

    Args:
        payload: mapping with the eight feature keys (case-sensitive).
    Returns:
        dict with ``probability`` (float in [0,1]), ``label`` (0/1) and the
        threshold used.
    """
    art = load_artifact()
    row = pd.DataFrame([[payload[f] for f in art["feature_order"]]],
                       columns=art["feature_order"])
    proba = float(art["pipeline"].predict_proba(row)[0, 1])
    label = int(proba >= art["threshold"])
    return {
        "probability": proba,
        "label": label,
        "threshold": float(art["threshold"]),
        "model": art["model_name"],
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    art = load_artifact()
    X = df[art["feature_order"]]
    proba = art["pipeline"].predict_proba(X)[:, 1]
    out = df.copy()
    out["probability"] = proba
    out["label"] = (proba >= art["threshold"]).astype(int)
    return out
