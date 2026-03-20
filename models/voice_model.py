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
