"""
Facial Recognition Model — Pipeline Script

Trains a Random Forest classifier to identify a group member from image features
extracted by the image pipeline (features/image_features.csv).

Target   : member  (member1 | member2 | member3)
Input    : features/image_features.csv
Output   : outputs/trained_models/face_model.pkl + face_encoder.pkl
           outputs/trained_models/model_metrics.json  (updated)

Features : 20 numeric — 4 intensity statistics + 16-bin normalised pixel histogram

All functions are importable for use from the command-line app (app/auth_simulation.py).

Usage:
  python models/face_model.py
  python -m models.face_model
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

ROOT         = Path(__file__).resolve().parent.parent
FEAT_FILE    = ROOT / "features" / "image_features.csv"
OUT_DIR      = ROOT / "outputs" / "trained_models"
METRICS_FILE = OUT_DIR / "model_metrics.json"

META_COLS = {"file_name", "augmentation", "expression"}

# ── Importable functions ──────────────────────────────────────────────────────

def prepare(df: pd.DataFrame, target: str = "member"):
    """
    Drop image metadata columns, keep numeric features, label-encode target.
    Returns (X: ndarray, y: ndarray, encoder: LabelEncoder).
    """
    d     = df.copy()
    y_raw = d.pop(target)
    d     = d.select_dtypes(include="number").fillna(d.median(numeric_only=True))
    le    = LabelEncoder()
    y     = le.fit_transform(y_raw.astype(str))
    return d.values.astype(float), y, le
