"""Build docs/demo.png — a 3-panel composite of the dashboard's Model Insights
tab so the README has a real screenshot rather than a placeholder.

Run from project root:
    python scripts/build_demo_image.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from diabetes_mlops import config as C
from diabetes_mlops.data import clean, load_raw
from diabetes_mlops.predict import load_artifact

OUT = ROOT / "docs" / "demo.png"


def main() -> None:
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid", context="talk")
    # Dark-theme palette matching the live Streamlit dashboard
    BG     = "#07091a"
    PANEL  = "#0f1330"
    TEXT   = "#e6e9f5"
    PURPLE = "#7c3aed"
    CYAN   = "#22d3ee"
    PINK   = "#f472b6"
    AMBER  = "#fbbf24"

    df = clean(load_raw())
    art = load_artifact()
    X, y = df[C.FEATURES], df[C.TARGET]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE
    )
    proba = art["pipeline"].predict_proba(X_test)[:, 1]
    pred = (proba >= art["threshold"]).astype(int)

    fig = plt.figure(figsize=(18, 6), constrained_layout=True, facecolor=BG)
    fig.suptitle(
        "Diabetes Risk Dashboard — Model Insights",
        fontsize=20, fontweight="bold", color=TEXT,
    )
    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0], facecolor=PANEL)
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["No diabetes", "Diabetes"]
    ).plot(ax=ax1, cmap="Purples", colorbar=False, values_format="d")
    ax1.set_title("Confusion matrix (test)", color=TEXT)
    for txt in ax1.texts:
        txt.set_color("white")
    ax1.tick_params(colors=TEXT); [s.set_color("#2a2f4f") for s in ax1.spines.values()]

    ax2 = fig.add_subplot(gs[0, 1], facecolor=PANEL)
    fpr, tpr, _ = roc_curve(y_test, proba)
    precision, recall, _ = precision_recall_curve(y_test, proba)
    ax2.plot(fpr, tpr, color=PURPLE, lw=3, label=f"ROC (AUC = {art['metrics']['roc_auc']:.3f})")
    ax2.fill_between(fpr, 0, tpr, color=PURPLE, alpha=0.15)
    ax2.plot(recall, precision, color=PINK, lw=3, label="Precision-Recall")
    ax2.plot([0, 1], [0, 1], "--", color="#444a6e", lw=1)
    ax2.axvline(C.TARGET_RECALL, color=AMBER, ls=":", lw=2, label=f"Recall target {C.TARGET_RECALL:.0%}")
    ax2.set_xlabel("FPR  /  Recall", color=TEXT)
    ax2.set_ylabel("TPR  /  Precision", color=TEXT)
    ax2.set_title("ROC + PR curves", color=TEXT)
    ax2.legend(loc="lower right", fontsize=10, facecolor=PANEL, edgecolor="#2a2f4f", labelcolor=TEXT)
    ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(colors=TEXT); [s.set_color("#2a2f4f") for s in ax2.spines.values()]

    ax3 = fig.add_subplot(gs[0, 2], facecolor=PANEL)
    result = permutation_importance(
        art["pipeline"], X_test, y_test,
        n_repeats=10, random_state=C.RANDOM_STATE, n_jobs=-1, scoring="roc_auc",
    )
    imp = pd.DataFrame({
        "feature": art["feature_order"],
        "importance": result.importances_mean,
        "std": result.importances_std,
    }).sort_values("importance")
    # gradient pink -> cyan
    n = len(imp); colors = [(PINK if v < 0.33 else CYAN if v > 0.66 else PURPLE)
                            for v in [(i / max(1, n - 1)) for i in range(n)]]
    ax3.barh(imp["feature"], imp["importance"], xerr=imp["std"], color=colors, ecolor="#444a6e")
    ax3.set_xlabel("Δ ROC-AUC when shuffled", color=TEXT)
    ax3.set_title("Feature importance", color=TEXT)
    ax3.tick_params(colors=TEXT); [s.set_color("#2a2f4f") for s in ax3.spines.values()]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor=BG)
    print(f"Wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
