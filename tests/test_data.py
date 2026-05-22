"""Data layer tests — no network required for the cleaning logic."""
from __future__ import annotations

import numpy as np
import pandas as pd

from diabetes_mlops import config as C
from diabetes_mlops.data import clean


def test_clean_replaces_impossible_zeros_with_nan():
    df = pd.DataFrame(
        {
            "Pregnancies": [0, 2],  # zero IS valid here
            "Glucose": [0, 130],
            "BloodPressure": [0, 70],
            "SkinThickness": [0, 25],
            "Insulin": [0, 80],
            "BMI": [0.0, 28.5],
            "DiabetesPedigreeFunction": [0.3, 0.5],
            "Age": [22, 45],
            "Outcome": [0, 1],
        }
    )
    out = clean(df)
    for col in C.ZERO_IS_MISSING:
        assert np.isnan(out.loc[0, col]), f"{col} zero should be NaN"
    # Pregnancies==0 must be preserved (a valid value, not "missing")
    assert out.loc[0, "Pregnancies"] == 0
    # Non-zero values are untouched
    assert out.loc[1, "Glucose"] == 130
