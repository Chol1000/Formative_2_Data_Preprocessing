"""
Voiceprint Verification Model — Pipeline Script

Trains a Random Forest classifier to identify a group member from audio features
extracted by the audio pipeline (features/audio_features.csv).

Target   : member  (member1 | member2 | member3)
Input    : features/audio_features.csv
Output   : outputs/trained_models/voice_model.pkl + voice_encoder.pkl
           outputs/trained_models/model_metrics.json  (updated)

Features : 32 numeric — 13 MFCC means, 13 MFCC stds, spectral roll-off (mean, std),
           RMS energy (mean, std), zero-crossing rate (mean, std)

Verification is two-step:
  1. Phrase must be in the approved whitelist
  2. Speaker identity must match the claimed username

All functions are importable for use from the command-line app (app/auth_simulation.py).

Usage:
  python models/voice_model.py
  python -m models.voice_model
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

ROOT      = Path(__file__).resolve().parent.parent
FEAT_FILE = ROOT / "features" / "audio_features.csv"
OUT_DIR   = ROOT / "outputs" / "trained_models"

META_COLS     = {"file_name", "augmentation", "duration_s", "phrase"}
VALID_PHRASES = {"yes, approve", "confirm transaction", "yes approve"}


# ── Importable functions ──────────────────────────────────────────────────────

def prepare(df: pd.DataFrame, target: str = "member"):
    """
    Drop audio metadata columns, keep numeric acoustic features,
    label-encode the target column.
    Returns (X: ndarray, y: ndarray, encoder: LabelEncoder).
    """
    d     = df.copy()
    y_raw = d.pop(target)
    d     = d.drop(columns=[c for c in META_COLS if c in d.columns], errors="ignore")
    d     = d.select_dtypes(include="number").fillna(d.median(numeric_only=True))
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw.astype(str))
    return d.values.astype(float), y, le


def train(X: np.ndarray, y: np.ndarray, n_estimators: int = 200):
    """
    Stratified K-Fold CV (up to 5 folds) with a balanced RandomForest.
    Prints accuracy, F1-weighted, and log-loss on a held-out split.
    Fits a final model on all data and returns (classifier, metrics_dict).
    """
    min_class = int(min(np.bincount(y)))
    n_splits  = max(2, min(5, min_class))
    cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf       = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", random_state=42
    )
    res = cross_validate(clf, X, y, cv=cv, scoring=["accuracy", "f1_weighted"])

    acc_mean = float(np.mean(res["test_accuracy"]))
    acc_std  = float(np.std(res["test_accuracy"]))
    f1_mean  = float(np.mean(res["test_f1_weighted"]))
    f1_std   = float(np.std(res["test_f1_weighted"]))

    print(f"  CV folds     : {n_splits}")
    print(f"  Accuracy     : {acc_mean:.4f}  (+/- {acc_std:.4f})")
    print(f"  F1 weighted  : {f1_mean:.4f}  (+/- {f1_std:.4f})")

    ll = None
    if min_class >= 2:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        clf_tmp = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced", random_state=42
        ).fit(Xtr, ytr)
        ll = round(log_loss(yte, clf_tmp.predict_proba(Xte)), 4)
        print(f"  Log loss     : {ll}")

    clf.fit(X, y)

    metrics = {
        "model"            : "Voiceprint Verification",
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
    joblib.dump(clf, out_dir / "voice_model.pkl")
    joblib.dump(le,  out_dir / "voice_encoder.pkl")
    print(f"  Saved: {out_dir / 'voice_model.pkl'}")
    print(f"  Saved: {out_dir / 'voice_encoder.pkl'}")

    # Update model_metrics.json — preserve entries from other models
    metrics_file = out_dir / "model_metrics.json"
    existing = {}
    if metrics_file.exists():
        with open(metrics_file) as fh:
            existing = json.load(fh)
    existing["voiceprint_verification"] = metrics
    with open(metrics_file, "w") as fh:
        json.dump(existing, fh, indent=2)
    print(f"  Saved: {metrics_file}")


def predict(phrase: str, username: str, clf, le: LabelEncoder,
            feat_df: pd.DataFrame) -> tuple[bool, str]:
    """
    Two-step voice verification:
      Step 1 — validate the spoken phrase against the approved whitelist.
      Step 2 — predict speaker identity from mean audio feature vector.
    Returns (is_verified: bool, reason: str).
    """
    if phrase.strip().lower() not in VALID_PHRASES:
        return False, "phrase not in approved list"
    rows = feat_df[feat_df["member"].str.lower() == username.lower()]
    if rows.empty:
        return False, "user not in training data"
    numeric  = rows.select_dtypes(include="number").drop(
        columns=["duration_s"], errors="ignore"
    ).mean(axis=0).values.reshape(1, -1)
    pred_idx = clf.predict(numeric)[0]
    pred     = le.classes_[pred_idx]
    return pred.lower() == username.lower(), pred

