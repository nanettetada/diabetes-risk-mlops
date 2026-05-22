"""Feature pipeline: imputation + scaling, wrapped in a sklearn Pipeline.

Keeping this in a Pipeline (not done as a one-off pandas transform) means the
same code path runs at fit time and at predict time — no train/serve skew.
"""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
