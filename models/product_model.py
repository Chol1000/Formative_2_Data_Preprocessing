"""
Product Recommendation Model — Pipeline Script

Trains a Random Forest classifier to recommend a product category to a customer
from behavioural and demographic features in the merged dataset
(data/merged_dataset.csv).

Target   : product_category  (e.g. Electronics | Books | Clothing | ...)
Input    : data/merged_dataset.csv
Output   : outputs/trained_models/product_model.pkl + product_encoder.pkl
           outputs/trained_models/model_metrics.json  (updated)

Features : numeric columns remaining after dropping product_category,
           customer_id, transaction_id, and purchase_date;
           categorical columns are label-encoded on the fly.

GridSearchCV is used to tune n_estimators, max_depth, min_samples_leaf, and
max_features so the final model is the best estimator found.

All functions are importable for use from the command-line app (app/auth_simulation.py).

Usage:
  python models/product_model.py
  python -m models.product_model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

ROOT      = Path(__file__).resolve().parent.parent
FEAT_FILE = ROOT / "data" / "merged_dataset.csv"
OUT_DIR   = ROOT / "outputs" / "trained_models"

DROP_COLS = {"product_category", "customer_id", "transaction_id", "purchase_date"}


# ── Importable functions ──────────────────────────────────────────────────────

def prepare(df: pd.DataFrame, target: str = "product_category"):
    """
    Encode categorical columns, drop the target and ID columns,
    label-encode the target.
    Returns (X: ndarray, y: ndarray, encoder: LabelEncoder).
    """
    d     = df.copy()
    y_raw = d.pop(target)
    d     = d.drop(columns=[c for c in DROP_COLS - {target} if c in d.columns],
                   errors="ignore")
    for col in d.select_dtypes(include=["object", "string"]).columns:
        d[col] = LabelEncoder().fit_transform(d[col].astype(str))
    d  = d.select_dtypes(include="number").fillna(d.median(numeric_only=True))
    le = LabelEncoder()
    y  = le.fit_transform(y_raw.astype(str))
    return d.values.astype(float), y, le


def train(X: np.ndarray, y: np.ndarray):
    """
    GridSearchCV over RandomForest hyperparameters with Stratified K-Fold CV.
    Prints best params, CV accuracy, F1-weighted, and log-loss.
    Fits the best estimator on all data and returns (classifier, metrics_dict).
    """
    min_class = int(min(np.bincount(y)))
    n_splits  = max(2, min(5, min_class))
    cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    param_grid = {
        "n_estimators":    [200, 300, 500, 1000],
        "max_depth":       [None, 5, 10, 15],
        "min_samples_leaf":[1, 2, 4],
        "max_features":    ["sqrt", "log2", None],
        "class_weight":    ["balanced", "balanced_subsample"],
    }
    base = RandomForestClassifier(random_state=42)
    gs   = GridSearchCV(base, param_grid, cv=cv, scoring="f1_weighted",
                        refit=True, n_jobs=-1)
    print("  Running GridSearchCV — this may take 1–2 minutes ...")
    gs.fit(X, y)
    clf = gs.best_estimator_

    print(f"  CV folds     : {n_splits}")
    print(f"  Best params  : {gs.best_params_}")

    res = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "f1_weighted"])

    acc_mean = float(np.mean(res["test_accuracy"]))
    acc_std  = float(np.std(res["test_accuracy"]))
    f1_mean  = float(np.mean(res["test_f1_weighted"]))
    f1_std   = float(np.std(res["test_f1_weighted"]))

    print(f"  Accuracy     : {acc_mean:.4f}  (+/- {acc_std:.4f})")
    print(f"  F1 weighted  : {f1_mean:.4f}  (+/- {f1_std:.4f})")

    ll = None
    if min_class >= 2:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        clf_tmp = RandomForestClassifier(
            **gs.best_params_, random_state=42
        ).fit(Xtr, ytr)
        ll = round(log_loss(yte, clf_tmp.predict_proba(Xte)), 4)
        print(f"  Log loss     : {ll}")

    metrics = {
        "model"            : "Product Recommendation",
        "cv_folds"         : n_splits,
        "accuracy_mean"    : round(acc_mean, 4),
        "accuracy_std"     : round(acc_std,  4),
        "f1_weighted_mean" : round(f1_mean,  4),
        "f1_weighted_std"  : round(f1_std,   4),
        "log_loss"         : ll,
        "n_samples"        : int(len(y)),
        "n_features"       : int(X.shape[1]),
    }
    return clf, metrics


def save(clf, le: LabelEncoder, metrics: dict, out_dir: Path = OUT_DIR):
    """Serialise model, label encoder, and metrics to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "product_model.pkl")
    joblib.dump(le,  out_dir / "product_encoder.pkl")
    print(f"  Saved: {out_dir / 'product_model.pkl'}")
    print(f"  Saved: {out_dir / 'product_encoder.pkl'}")

    # Update model_metrics.json — preserve entries from other models
    metrics_file = out_dir / "model_metrics.json"
    existing = {}
    if metrics_file.exists():
        with open(metrics_file) as fh:
            existing = json.load(fh)
    existing["product_recommendation"] = metrics
    with open(metrics_file, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"  Saved: {metrics_file}")


def predict(username: str, clf, le: LabelEncoder,
            feat_df: pd.DataFrame) -> str:
    """
    Recommend a product category for the given username.
    Samples a row from feat_df for that user, encodes it, and runs prediction.
    Returns the predicted product_category string.
    """
    rows = feat_df[feat_df.get("customer_id", pd.Series(dtype=str))
                   .astype(str).str.lower() == username.lower()]
    if rows.empty:
        # fall back to any random row
        rows = feat_df.sample(1, random_state=hash(username.lower()) & 0xFFFF)

    sample = rows.iloc[[0]].copy()
    for col in ["product_category", "customer_id", "transaction_id", "purchase_date"]:
        if col in sample.columns:
            sample = sample.drop(columns=[col])
    for col in sample.select_dtypes(include=["object", "string"]).columns:
        sample[col] = LabelEncoder().fit_transform(sample[col].astype(str))
    sample   = sample.select_dtypes(include="number").fillna(0)
    pred_idx = clf.predict(sample.values)[0]
    return le.classes_[pred_idx]


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Product Recommendation Model")
    print("=" * 50)

    if not FEAT_FILE.exists():
        print(f"Feature file not found: {FEAT_FILE}")
        print("Run Task 1 in the notebook first to generate merged_dataset.csv.")
        return

    df = pd.read_csv(FEAT_FILE)
    print(f"Loaded   : {df.shape[0]} rows x {df.shape[1]} columns")
    if "product_category" in df.columns:
        cats = sorted(df["product_category"].unique())
        print(f"Classes  : {cats}  ({len(cats)} categories)")
    if "customer_id" in df.columns:
        print(f"Customers: {df['customer_id'].nunique()} unique")
    print()

    X, y, le     = prepare(df)
    print(f"Features : {X.shape[1]} numeric  |  Samples : {X.shape[0]}")
    print()
    clf, metrics = train(X, y)
    print()
    save(clf, le, metrics)


if __name__ == "__main__":
    main()
