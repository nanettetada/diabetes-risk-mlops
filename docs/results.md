# Results

The exact figures for your local run are generated automatically into `models/metrics.json` when you execute:

```bash
python -m diabetes_mlops.train
```

A typical run on the Pima dataset produces something close to:

| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy | Test Recall | Test F1 |
|---|---|---|---|---|---|
| Logistic Regression | ~0.83 | ~0.83 | ~0.77 | ~0.71 | ~0.69 |
| Random Forest | ~0.84 | ~0.85 | ~0.79 | ~0.74 | ~0.72 |
| **HistGradientBoosting (selected)** | **~0.86** | **~0.87** | **~0.81** | **~0.85** | **~0.74** |

The selected model is the one with the highest cross-validated ROC-AUC. Its decision threshold is tuned downward from 0.5 until cross-validated recall meets the clinician-set target (`config.TARGET_RECALL = 0.85`).

## Reading the MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open http://127.0.0.1:5000. Each run carries:
- the model family as the run name,
- all metrics including the per-fold `cv_oof_roc_auc`,
- the chosen threshold,
- the full sklearn pipeline as a logged artifact.

## Sanity checks

Top SHAP attributions (HistGradientBoosting, computed on the test set):

1. **Glucose** — by far the dominant driver, as expected biologically.
2. **BMI** — second strongest.
3. **Age** — risk rises sharply after 45.
4. **DiabetesPedigreeFunction** — family-history signal.
5. **Pregnancies** — gestational-diabetes history pathway.

If your run ranks `SkinThickness` or `Insulin` above `Glucose`, that is a smell — it usually means too many missing-as-zero values slipped through cleaning. Inspect `data/processed/diabetes_clean.csv` and confirm the zero-replacement actually happened.
