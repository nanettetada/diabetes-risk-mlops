"""Train, evaluate, and register the diabetes risk model.

Logs every candidate model to MLflow, then promotes the best (by ROC-AUC) to
``models/diabetes_pipeline.joblib`` together with the chosen probability
threshold and the headline metrics in ``models/metrics.json``.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    import mlflow  # noqa: F401
    _HAS_MLFLOW = True
except ModuleNotFoundError:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from . import config as C
from .data import build_processed, clean
from .features import make_preprocessor

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    name: str
    pipeline: Pipeline
    metrics: dict
    threshold: float


def _candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=C.RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=C.RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.05, max_iter=400, random_state=C.RANDOM_STATE
        ),
    }


def _tune_threshold(y_true: np.ndarray, y_score: np.ndarray, min_recall: float) -> float:
    """Pick the highest threshold that still achieves the required recall."""
    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns thresholds of length n-1 relative to recall.
    ok = recall[:-1] >= min_recall
    if not ok.any():
        log.warning("No threshold meets recall >= %.2f, defaulting to 0.5", min_recall)
        return 0.5
    candidate_thresholds = thresh[ok]
    return float(candidate_thresholds.max())


def _fit_and_score(name: str, estimator, X_train, y_train, X_test, y_test) -> RunResult:
    pipe = Pipeline([("pre", make_preprocessor()), ("clf", estimator)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    oof_proba = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = _tune_threshold(y_train.to_numpy(), oof_proba, C.TARGET_RECALL)

    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_proba)),
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "recall": float(recall_score(y_test, test_pred)),
        "f1": float(f1_score(y_test, test_pred)),
        "threshold": float(threshold),
        "cv_oof_roc_auc": float(roc_auc_score(y_train, oof_proba)),
    }
    cm = confusion_matrix(y_test, test_pred).tolist()
    metrics["confusion_matrix"] = cm

    return RunResult(name=name, pipeline=pipe, metrics=metrics, threshold=threshold)


def train(processed_csv: Path = C.PROCESSED_CSV) -> RunResult:
    if not processed_csv.exists():
        build_processed()
    df = pd.read_csv(processed_csv)
    # Belt-and-braces: clean again in case the processed file was hand-edited.
    df = clean(df)

    X = df[C.FEATURES]
    y = df[C.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )

    if _HAS_MLFLOW:
        mlflow.set_tracking_uri((C.ROOT / "mlruns").as_uri())
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
    else:
        log.warning("mlflow not installed — skipping experiment tracking")

    results: list[RunResult] = []
    for name, est in _candidates().items():
        if _HAS_MLFLOW:
            with mlflow.start_run(run_name=name):
                res = _fit_and_score(name, est, X_train, y_train, X_test, y_test)
                mlflow.log_param("model", name)
                mlflow.log_param("target_recall", C.TARGET_RECALL)
                for k, v in res.metrics.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                log.info("%s -> %s", name, {k: v for k, v in res.metrics.items() if k != "confusion_matrix"})
                results.append(res)
        else:
            res = _fit_and_score(name, est, X_train, y_train, X_test, y_test)
            log.info("%s -> %s", name, {k: v for k, v in res.metrics.items() if k != "confusion_matrix"})
            results.append(res)

    best = max(results, key=lambda r: r.metrics["roc_auc"])
    log.info("Best model: %s (ROC-AUC=%.3f)", best.name, best.metrics["roc_auc"])

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": best.pipeline,
        "threshold": best.threshold,
        "feature_order": C.FEATURES,
        "model_name": best.name,
        "metrics": best.metrics,
    }
    joblib.dump(artifact, C.MODEL_PATH)
    C.METRICS_PATH.write_text(
        json.dumps({"selected": best.name, **best.metrics}, indent=2)
    )
    log.info("Wrote %s and %s", C.MODEL_PATH, C.METRICS_PATH)
    return best


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
