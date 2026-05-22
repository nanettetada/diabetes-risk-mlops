"""End-to-end smoke test: train a tiny model, predict, assert shape."""
from __future__ import annotations

import importlib

import pytest

from diabetes_mlops import config as C


@pytest.fixture(scope="module")
def trained_model():
    # Build processed data + train. This hits the live dataset URL.
    data_mod = importlib.import_module("diabetes_mlops.data")
    train_mod = importlib.import_module("diabetes_mlops.train")
    data_mod.build_processed()
    train_mod.train()
    return C.MODEL_PATH


def test_predict_one_returns_valid_probability(trained_model):
    from diabetes_mlops.predict import predict_one

    payload = {
        "Pregnancies": 2,
        "Glucose": 138,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 45,
    }
    result = predict_one(payload)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in (0, 1)
    assert "threshold" in result and 0.0 < result["threshold"] < 1.0
