# Methodology

This document expands on the high-level methodology section of the README, covering the decisions a hiring manager would want to interrogate.

## 1. Framing

This is a **screening-triage** problem: identify patients who warrant further testing. The downstream action is "order HbA1c", not "diagnose diabetes". The cost asymmetry — a missed case is worse than an unnecessary follow-up — drives every modelling decision below.

## 2. Data integrity

The Pima Indians Diabetes dataset encodes biological zeros (e.g. `Glucose = 0`, `BMI = 0`) for many records that are actually missing values. A domain-naïve pipeline would carry these zeros into training and produce a model that has secretly learnt "zero means high risk" (because the upstream coders happened to fill in non-zero values for healthier patients more often).

`src/diabetes_mlops/data.py::clean()` replaces these with `NaN`, and the sklearn `SimpleImputer(strategy="median")` step inside the pipeline does the actual imputation — **inside cross-validation folds**, so no test-fold information leaks into the median.

## 3. Validation strategy

Stratified 5-fold CV on the training partition, stratified test holdout (20%). Stratification matters: the class balance is roughly 65/35, and a vanilla random split can drift the test prevalence by several percentage points, which materially changes the reported recall.

All hyperparameter selection and threshold tuning happen on cross-validated out-of-fold predictions on the training set. The held-out test set is touched **once**, after model selection, to produce the headline numbers.

## 4. Threshold tuning

The default 0.5 cutoff is arbitrary for a screening tool. Instead, the threshold is chosen as the **largest value that still achieves recall ≥ 0.85** on out-of-fold predictions (configurable in `config.py::TARGET_RECALL`). This frames the precision/recall trade-off as a clinician-set policy parameter — change the constant, retrain, the artifact's stored threshold updates automatically and both deployment surfaces pick it up.

## 5. Model comparison

| Model | Why it's in the lineup |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients map to clinical intuition (a one-SD bump in `Glucose` shifts log-odds by X). |
| Random Forest | Captures non-linear interactions (e.g. BMI × Age) without manual feature engineering. |
| HistGradientBoosting | The current state-of-the-art on small/medium tabular data; usually wins on this benchmark. |

Selection rule: highest CV ROC-AUC, ties broken by recall-at-target-threshold. ROC-AUC is the right ranking metric because it's threshold-free and invariant to class prevalence shifts.

## 6. Experiment tracking

MLflow logs every candidate. Open the UI to compare:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Each run logs hyperparameters, all metrics (including the confusion matrix as JSON), and the serialised pipeline. Run names match the model family for easy filtering.

## 7. What's NOT done (yet)

- **Calibration**. Probabilities from gradient-boosted trees aren't always well-calibrated; a `CalibratedClassifierCV` pass with isotonic regression would be the next addition.
- **Fairness analysis**. The dataset is single-population (female, Pima heritage) — fairness across protected attributes can't be evaluated within it.
- **Drift monitoring**. Production drift detection (Evidently, NannyML) would be the next MLOps step beyond tracking.
- **Model registry promotion gates**. MLflow Model Registry transitions (Staging → Production) are stubbed out; a real deployment would wire CI/CD to these transitions.

These are deliberate omissions, not oversights — keeping scope honest for a portfolio project.
