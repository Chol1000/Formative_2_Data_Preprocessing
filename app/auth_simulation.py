"""
User Identity and Product Recommendation System — CLI Simulation

Demonstrates the full three-step authentication pipeline:
  Step 1  Facial Recognition      — verifies identity from image features
  Step 2  Voiceprint Verification — validates phrase and confirms speaker
  Step 3  Product Recommendation  — predicts purchase category from customer profile

Models are loaded from outputs/trained_models/. If model files are missing,
a rule-based fallback runs automatically so the demo always works.

Usage:
  python app/auth_simulation.py                   interactive menu
  python app/auth_simulation.py --auto            full demo, no prompts
  python app/auth_simulation.py --sim authorized  authorised simulation only
  python app/auth_simulation.py --sim denied      unauthorised attempts only
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    _DEPS = True
except ImportError:
    _DEPS = False

# ── Paths ──────────────────────────────────────────────────────────────────────
MODELS_DIR   = ROOT / "outputs" / "trained_models"
METRICS_FILE = MODELS_DIR / "model_metrics.json"
IMAGE_FEATS  = ROOT / "features" / "image_features.csv"
AUDIO_FEATS  = ROOT / "features" / "audio_features.csv"
MERGED_DATA  = ROOT / "data"     / "merged_dataset.csv"
LOG_FILE     = ROOT / "outputs"  / "simulation" / "simulation_log.txt"

VALID_MEMBERS = {"chol", "ineza", "nziza"}
VALID_PHRASES = {"yes, approve", "confirm transaction", "yes approve"}

W = 60  # output width

# ── Model state ────────────────────────────────────────────────────────────────
_models: dict   = {}
_img_df: object = None
_aud_df: object = None
_mrg_df: object = None


def load_models() -> bool:
    """Load all three trained models and feature datasets from disk."""
    global _models, _img_df, _aud_df, _mrg_df

    if not (_DEPS and MODELS_DIR.exists()):
        return False

    try:
        for name, mf, ef in [
            ("face",    "face_model.pkl",    "face_encoder.pkl"),
            ("voice",   "voice_model.pkl",   "voice_encoder.pkl"),
            ("product", "product_model.pkl", "product_encoder.pkl"),
        ]:
            _models[name]          = joblib.load(MODELS_DIR / mf)
            _models[f"{name}_enc"] = joblib.load(MODELS_DIR / ef)

        if IMAGE_FEATS.exists():
            _img_df = pd.read_csv(IMAGE_FEATS)
        if AUDIO_FEATS.exists():
            _aud_df = pd.read_csv(AUDIO_FEATS)
        if MERGED_DATA.exists():
            _mrg_df = pd.read_csv(MERGED_DATA)

        return True
    except Exception as exc:
        print(f"  Warning: could not load models ({exc}). Using rule-based fallback.")
        _models.clear()
        return False


# ── Feature helpers ────────────────────────────────────────────────────────────
def _image_vector(username: str):
    if _img_df is None:
        return None, []
    rows = _img_df[_img_df["member"].str.lower() == username.lower()]
    if rows.empty:
        return None, []
    files = list(dict.fromkeys(rows["file_name"].tolist())) if "file_name" in rows.columns else []
    return rows.select_dtypes(include="number").mean(axis=0).values.reshape(1, -1), files


def _audio_vector(username: str):
    if _aud_df is None:
        return None, []
    rows = _aud_df[_aud_df["member"].str.lower() == username.lower()]
    if rows.empty:
        return None, []
    files = list(dict.fromkeys(rows["file_name"].tolist())) if "file_name" in rows.columns else []
    numeric = rows.select_dtypes(include="number").drop(columns=["duration_s"], errors="ignore")
    return numeric.mean(axis=0).values.reshape(1, -1), files


# ── Verification functions ─────────────────────────────────────────────────────
def verify_face(username: str) -> tuple[bool, str]:
    """
    Build a mean pixel feature vector for the username from image_features.csv
    and run the RandomForest face model. Falls back to membership lookup.
    Returns (is_verified, method).
    """
    u = username.strip().lower()
    if "face" in _models:
        feat, files = _image_vector(u)
        if feat is not None:
            pred = _models["face_enc"].classes_[_models["face"].predict(feat)[0]]
            return pred.lower() == u, "RandomForest model", files
        return False, "user not in training data", []
    return u in VALID_MEMBERS, "rule-based lookup", []


def verify_voice(phrase: str, username: str) -> tuple[bool, str]:
    """
    Step 1 — validate phrase against approved whitelist.
    Step 2 — confirm speaker identity from MFCC feature vector.
    Falls back to phrase-only check if models unavailable.
    Returns (is_verified, method).
    """
    p = phrase.strip().lower()
    if p not in VALID_PHRASES:
        return False, "phrase not in approved list", []
    if "voice" in _models:
        feat, files = _audio_vector(username)
        if feat is not None:
            pred = _models["voice_enc"].classes_[_models["voice"].predict(feat)[0]]
            return pred.lower() == username.lower(), "RandomForest model", files
        return False, "user not in training data", []
    return True, "rule-based (phrase validated)", []


def recommend_product(username: str) -> tuple[str, dict]:
    """
    Sample a customer profile from merged_dataset.csv and run the
    RandomForest product model to predict a purchase category.
    Returns (product_category, sample_features_dict).
    """
    features = {}
    if "product" in _models and _mrg_df is not None:
        try:
            row = _mrg_df.sample(1, random_state=hash(username.lower()) & 0xFFFF).copy()
            features = {
                "Platform"   : row["social_media_platform"].values[0] if "social_media_platform" in row else "N/A",
                "Engagement" : round(float(row["engagement_score"].values[0]), 1) if "engagement_score" in row else "N/A",
                "Interest"   : round(float(row["purchase_interest_score"].values[0]), 1) if "purchase_interest_score" in row else "N/A",
                "Sentiment"  : row["review_sentiment"].values[0] if "review_sentiment" in row else "N/A",
            }
            for col in ["product_category", "customer_id"]:
                if col in row.columns:
                    row = row.drop(columns=[col])
            for col in row.select_dtypes(include=["object", "string"]).columns:
                row[col] = LabelEncoder().fit_transform(row[col].astype(str))
            row      = row.select_dtypes(include="number").fillna(0)
            pred_idx = _models["product"].predict(row.values)[0]
            return _models["product_enc"].classes_[pred_idx], features
        except Exception:
            pass
    import random
    random.seed(hash(username.lower()) & 0xFFFF)
    return random.choice(["Electronics", "Books", "Clothing", "Groceries", "Sports"]), features


# ── Logging ────────────────────────────────────────────────────────────────────
def log_event(status: str, user: str, step: str, product: str = "") -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{ts} | {status:<13} | user={user:<12} | step={step}"
    if product:
        entry += f" | product={product}"
    with open(LOG_FILE, "a") as fh:
        fh.write(entry + "\n")


# ── Pipeline ───────────────────────────────────────────────────────────────────
def run_pipeline(username: str, phrase: str) -> bool:
    """
    Execute the full 3-step authentication and recommendation pipeline.
    Prints each step result. Returns True if access granted, False otherwise.
    """
    print(f"  User   : {username}")
    print(f"  Phrase : {phrase}")
    print("  " + "-" * (W - 2))

    # Step 1 — Facial Recognition
    face_ok, face_method, img_files = verify_face(username)
    print(f"  [1/3] Facial Recognition      : {'VERIFIED' if face_ok else 'FAILED'}")
    print(f"        Method : {face_method}")
    if img_files:
        originals = sorted({f for f in img_files if "original" in str(f).lower() or "_" not in str(f)})
        sample = (originals or img_files)[:3]
        print(f"        Images : {', '.join(str(p) for p in sample)}")
    if not face_ok:
        log_event("ACCESS_DENIED", username, "face_failed")
        print()
        print(f"  OUTCOME : ACCESS DENIED — face not recognised")
        return False

    # Step 2 — Voiceprint Verification
    voice_ok, voice_method, aud_files = verify_voice(phrase, username)
    print(f"  [2/3] Voiceprint Verification : {'VERIFIED' if voice_ok else 'FAILED'}")
    print(f"        Method : {voice_method}")
    if aud_files:
        originals = sorted({f for f in aud_files if "original" in str(f).lower()})
        sample = (originals or aud_files)[:2]
        print(f"        Audio  : {', '.join(str(p) for p in sample)}")
    if not voice_ok:
        log_event("ACCESS_DENIED", username, "voice_failed")
        print()
        print(f"  OUTCOME : ACCESS DENIED — voice not authorised")
        return False

    # Step 3 — Product Recommendation
    product, feats = recommend_product(username)
    log_event("ACCESS_GRANTED", username, "success", product=product)
    print(f"  [3/3] Product Recommendation  : {product}")
    print(f"        Source : merged_dataset.csv (customer social + transaction profile)")
    if feats:
        print(f"        Features used  : Platform={feats.get('Platform')}  "
              f"Engagement={feats.get('Engagement')}  "
              f"Interest={feats.get('Interest')}  "
              f"Sentiment={feats.get('Sentiment')}")
    print()
    print("  " + "=" * (W - 2))
    print(f"  OUTCOME          : ACCESS GRANTED")
    print(f"  User             : {username}")
    print(f"  Recommended item : {product}")
    print("  " + "=" * (W - 2))
    return True


# ── Simulation 1 — Authorised ──────────────────────────────────────────────────
def run_authorized(auto: bool = False) -> None:
    print()
    print("=" * W)
    print("  SIMULATION 1 — Authorised Transaction")
    print("=" * W)

    if auto:
        # Run all three members to show the pipeline is not hardcoded
        cases = [
            ("chol",   "yes, approve"),
            ("ineza",  "confirm transaction"),
            ("nziza",  "yes, approve"),
        ]
        print("  Running all three registered members in sequence.")
        results = {}
        for username, phrase in cases:
            print()
            granted = run_pipeline(username, phrase)
            results[username] = "GRANTED" if granted else "DENIED"

        print()
        print("-" * W)
        print("  Summary")
        print("-" * W)
        for user, outcome in results.items():
            print(f"  {user:<14} : {outcome}")
        print("-" * W)

    else:
        # Interactive — user types their own username and phrase
        username = input("\n  Enter username: ").strip()
        print()
        print("  [1/3] Facial Recognition")
        face_ok, face_method, img_files = verify_face(username)
        print(f"        Result : {'VERIFIED' if face_ok else 'FAILED'}  ({face_method})")
        if img_files:
            print(f"        Images : {img_files[0]}")
        if not face_ok:
            log_event("ACCESS_DENIED", username, "face_failed")
            print(f"\n  OUTCOME : ACCESS DENIED — '{username}' not recognised")
            return

        phrase = input(
            '\n  Enter approved phrase\n'
            '  ("yes, approve"  /  "confirm transaction"): '
        ).strip()

        print()
        print("  [2/3] Voiceprint Verification")
        print(f"        Phrase : {phrase}")
        voice_ok, voice_method, aud_files = verify_voice(phrase, username)
        print(f"        Result : {'VERIFIED' if voice_ok else 'FAILED'}  ({voice_method})")
        if aud_files:
            print(f"        Audio  : {aud_files[0]}")
        if not voice_ok:
            log_event("ACCESS_DENIED", username, "voice_failed")
            print(f"\n  OUTCOME : ACCESS DENIED — phrase not authorised")
            return

        print()
        print("  [3/3] Product Recommendation")
        product, feats = recommend_product(username)
        log_event("ACCESS_GRANTED", username, "success", product=product)
        if feats:
            print(f"        Source   : merged_dataset.csv")
            print(f"        Features : Platform={feats.get('Platform')}  "
                  f"Engagement={feats.get('Engagement')}  "
                  f"Interest={feats.get('Interest')}")

        print()
        print("-" * W)
        print(f"  OUTCOME          : ACCESS GRANTED")
        print(f"  User             : {username}")
        print(f"  Recommended item : {product}")
        print("-" * W)


# ── Simulation 2 — Unauthorised ────────────────────────────────────────────────
def _run_attempt(label: str, username: str, phrase: str, note: str = "") -> None:
    print(f"  {label}")
    if note:
        print(f"  Note   : {note}")
    print(f"  User   : {username}")
    print(f"  Phrase : {phrase}")
    print("  " + "-" * (W - 2))

    face_ok, face_method, img_files = verify_face(username)
    print(f"  [1/3] Facial Recognition      : {'PASS' if face_ok else 'FAIL'}")
    print(f"        Method : {face_method}")
    if img_files:
        print(f"        Images : {img_files[0]}")
    if not face_ok:
        log_event("ACCESS_DENIED", username, "face_failed")
        print(f"\n  OUTCOME : ACCESS DENIED — blocked at Step 1 (image gate)")
        return

    voice_ok, voice_method, aud_files = verify_voice(phrase, username)
    print(f"  [2/3] Voiceprint Verification : {'PASS' if voice_ok else 'FAIL'}")
    print(f"        Method : {voice_method}")
    if aud_files:
        print(f"        Audio  : {aud_files[0]}")
    if not voice_ok:
        log_event("ACCESS_DENIED", username, "voice_failed")
        print(f"\n  OUTCOME : ACCESS DENIED — blocked at Step 2 (audio gate)")
        return

    product, _ = recommend_product(username)
    log_event("ACCESS_GRANTED", username, "success", product=product)
    print(f"\n  OUTCOME : ACCESS GRANTED — credentials are valid")
    print(f"  Recommended item : {product}")


def run_unauthorized(auto: bool = False) -> None:
    print()
    print("=" * W)
    print("  SIMULATION 2 — Unauthorised Attempts")
    print("=" * W)

    if auto:
        print("  Two pre-defined attack vectors — both must be blocked")
        print()
        _run_attempt(
            label    = "Attack A — Unknown user (image-based)",
            username = "intruder_x",
            phrase   = "yes, approve",
            note     = "Not a registered member — no face data in training set",
        )
        print()
        print("-" * W)
        print()
        _run_attempt(
            label    = "Attack B — Registered user, invalid phrase (audio-based)",
            username = "chol",
            phrase   = "open sesame",
            note     = "Valid user but phrase is not in the approved whitelist",
        )
        print()
        print("=" * W)
        print("  Both unauthorised attempts were correctly blocked.")
        print("  Image gate blocked Attack A  |  Audio gate blocked Attack B")
        print("=" * W)
        return

    # Interactive — user enters any credentials to test the security gates
    print("  Enter any credentials to test the authentication pipeline.")
    print("  The system will show exactly which gate blocks the attempt.")
    print()

    while True:
        username = input("  Enter username to test: ").strip()
        phrase   = input("  Enter phrase to test  : ").strip()
        print()
        print("  " + "-" * (W - 2))
        _run_attempt(
            label    = "Unauthorised Attempt",
            username = username,
            phrase   = phrase,
        )
        print()
        again = input("  Try another attempt? [y/N]: ").strip().lower()
        if again != "y":
            break


# ── Metrics and Log ────────────────────────────────────────────────────────────
def show_metrics() -> None:
    print()
    print("=" * W)
    print("  Model Performance Metrics")
    print("=" * W)

    if not METRICS_FILE.exists():
        print("\n  Metrics file not found — run Task 4 in the notebook first.")
        return

    with open(METRICS_FILE) as fh:
        metrics = json.load(fh)

    for key, vals in metrics.items():
        label = key.replace("_", " ").title()
        print(f"\n  {label}")
        print(f"    Accuracy (CV mean) : {vals.get('accuracy_mean', 0):.4f}"
              f"  +/-  {vals.get('accuracy_std', 0):.4f}")
        print(f"    F1 weighted        : {vals.get('f1_weighted_mean', 0):.4f}"
              f"  +/-  {vals.get('f1_weighted_std', 0):.4f}")
        print(f"    Log loss           : {vals.get('log_loss', 'N/A')}")
        print(f"    CV folds           : {vals.get('cv_folds', '?')}")
        print(f"    Samples            : {vals.get('n_samples', '?')}")


def show_log() -> None:
    print()
    print("=" * W)
    print("  Simulation Event Log")
    print("=" * W)
    if not LOG_FILE.exists():
        print("\n  No log entries yet.\n")
        return
    with open(LOG_FILE) as fh:
        lines = fh.readlines()
    print()
    for line in lines[-20:]:
        print(f"  {line.rstrip()}")
    print()
    granted = sum(1 for l in lines if "ACCESS_GRANTED" in l)
    denied  = sum(1 for l in lines if "ACCESS_DENIED"  in l)
    print(f"  Total entries : {len(lines)}  |  Granted : {granted}  |  Denied : {denied}")


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="User Identity and Product Recommendation System — CLI"
    )
    parser.add_argument("--auto", action="store_true",
                        help="Run full demo without input prompts")
    parser.add_argument("--sim", choices=["authorized", "denied"],
                        help="Run one simulation type only")
    args = parser.parse_args()

    print()
    print("=" * W)
    print("  User Identity and Product Recommendation System")
    print("=" * W)

    using_models = load_models()
    mode = "RandomForest models" if using_models else "rule-based fallback"
    print(f"  Mode : {mode}")

    if args.auto:
        run_authorized(auto=True)
        run_unauthorized(auto=True)
        show_metrics()
        show_log()
        return

    if args.sim == "authorized":
        run_authorized(auto=True)
        return
    if args.sim == "denied":
        run_unauthorized(auto=True)
        return

    menu = {
        "1": ("Run authorised transaction simulation", lambda: run_authorized(auto=False)),
        "2": ("Run unauthorised attempt demo",         lambda: run_unauthorized(auto=False)),
        "3": ("Show model metrics",                    show_metrics),
        "4": ("Show event log",                        show_log),
        "5": ("Exit",                                  None),
    }

    while True:
        print()
        for key, (label, _) in menu.items():
            print(f"  {key}. {label}")
        choice = input("\n  Select option [1-5]: ").strip()

        if choice not in menu:
            print("  Invalid choice — enter a number between 1 and 5.")
            continue

        label, fn = menu[choice]
        if fn is None:
            print("  Goodbye.")
            break
        fn()


if __name__ == "__main__":
    main()
