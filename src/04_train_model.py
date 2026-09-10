"""
04_train_model.py
-------------------
Trains and evaluates baseline classifiers (Random Forest + XGBoost) on the
extracted URL features, and saves the better-performing model plus
evaluation artifacts (confusion matrix, feature importance, metrics).

This is the "Phase 1 centralized baseline" referenced in the dissertation's
motivation for federated learning: it shows what a single organization can
achieve with full access to its own labeled data, as a reference point for
the federated setup in later work.

Reads:  data/processed/features.csv
Writes: models/best_model.joblib
        results/metrics.json
        results/confusion_matrix.png
        results/feature_importance.png
"""

import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

BASE = os.path.dirname(__file__)
FEATURES_FILE = os.path.join(BASE, "..", "data", "processed", "features.csv")
MODEL_OUT = os.path.join(BASE, "..", "models", "best_model.joblib")
METRICS_OUT = os.path.join(BASE, "..", "results", "metrics.json")
CM_PLOT_OUT = os.path.join(BASE, "..", "results", "confusion_matrix.png")
FI_PLOT_OUT = os.path.join(BASE, "..", "results", "feature_importance.png")

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = [c for c in df.columns if c not in ("url", "label")]
    X = df[feature_cols]
    y = df["label"]
    return X, y, feature_cols


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }
    return metrics, preds


def main():
    X, y, feature_cols = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # --- Model 1: Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_preds = evaluate(rf, X_test, y_test, "RandomForest")

    # --- Model 2: XGBoost ---
    xgb = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    xgb.fit(X_train, y_train)
    xgb_metrics, xgb_preds = evaluate(xgb, X_test, y_test, "XGBoost")

    print("\n=== Random Forest ===")
    for k, v in rf_metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    print("\n=== XGBoost ===")
    for k, v in xgb_metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    # Pick the better model by F1 (balances precision/recall - both matter:
    # false positives annoy users, false negatives let quishing through)
    if xgb_metrics["f1"] >= rf_metrics["f1"]:
        best_model, best_metrics, best_preds, best_name = xgb, xgb_metrics, xgb_preds, "XGBoost"
        importances = xgb.feature_importances_
    else:
        best_model, best_metrics, best_preds, best_name = rf, rf_metrics, rf_preds, "RandomForest"
        importances = rf.feature_importances_

    print(f"\n>>> Best model: {best_name} (by F1 score)")

    # --- Save model ---
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(best_model, MODEL_OUT)

    # --- Save metrics (both models, for transparency) ---
    os.makedirs(os.path.dirname(METRICS_OUT), exist_ok=True)
    all_metrics = {
        "random_forest": rf_metrics,
        "xgboost": xgb_metrics,
        "best_model": best_name,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- Confusion matrix plot ---
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(CM_PLOT_OUT, dpi=150)
    plt.close()

    # --- Feature importance plot ---
    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances
    }).sort_values("importance", ascending=True)

    plt.figure(figsize=(8, 7))
    plt.barh(fi_df["feature"], fi_df["importance"], color="#4C72B0")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(FI_PLOT_OUT, dpi=150)
    plt.close()

    print("\nTop 5 most important features:")
    print(fi_df.sort_values("importance", ascending=False).head(5).to_string(index=False))

    print("\nFull classification report (best model):")
    print(classification_report(y_test, best_preds, target_names=["Legitimate", "Phishing"]))

    # --- Robustness ablation ---
    # The top feature (suspicious_tld) dominated importance in early runs.
    # Check how much the model relies on it alone vs. genuine signal spread
    # across the other features - an attacker registering a normal TLD
    # should not trivially defeat the whole model.
    top_feature = fi_df.sort_values("importance", ascending=False).iloc[0]["feature"]
    X_train_ablated = X_train.drop(columns=[top_feature])
    X_test_ablated = X_test.drop(columns=[top_feature])

    ablated_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss"
    )
    ablated_model.fit(X_train_ablated, y_train)
    ablated_metrics, _ = evaluate(ablated_model, X_test_ablated, y_test, f"XGBoost (no {top_feature})")

    print(f"\n=== Robustness check: performance WITHOUT top feature '{top_feature}' ===")
    for k, v in ablated_metrics.items():
        if k != "model":
            print(f"  {k}: {v:.4f}")

    all_metrics["ablation_top_feature_removed"] = top_feature
    all_metrics["ablation_metrics"] = ablated_metrics
    with open(METRICS_OUT, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved metrics (incl. ablation) to {METRICS_OUT}")
    print(f"Saved plots to {CM_PLOT_OUT} and {FI_PLOT_OUT}")


if __name__ == "__main__":
    main()
