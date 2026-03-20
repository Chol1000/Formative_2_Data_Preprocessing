# Formative 2: Multimodal Data Preprocessing

A multimodal user identity verification and product recommendation system that authenticates
users through sequential facial recognition and voiceprint verification before serving a
personalised product recommendation from a trained machine learning model.

**Course:** Machine Learning Pipeline
**GitHub:** https://github.com/Chol1000/Formative_2_Data_Preprocessing.git
**Video:** https://drive.google.com/file/d/1krrdtjkNTfhp1Ld4Xl-UToLU4egcpy35/view?usp=sharing

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Repository Structure](#3-repository-structure)
4. [Setup and Installation](#4-setup-and-installation)
5. [Datasets](#5-datasets)
6. [Task 1 — Data Merging and EDA](#6-task-1--data-merging-and-eda)
7. [Task 2 — Facial Image Collection and Processing](#7-task-2--facial-image-collection-and-processing)
8. [Task 3 — Audio Collection and Processing](#8-task-3--audio-collection-and-processing)
9. [Task 4 — Model Development and Evaluation](#9-task-4--model-development-and-evaluation)
10. [Task 6 — System Simulation](#10-task-6--system-simulation)
11. [Running the Pipeline](#11-running-the-pipeline)
12. [Command-Line Application](#12-command-line-application)
13. [Authentication Flow](#13-authentication-flow)
14. [Team Contributions](#14-team-contributions)

---

## 1. Project Overview

This project implements a three-gate sequential pipeline that first confirms who the user is
before recommending what they are likely to buy:

1. **Facial Recognition** — a Random Forest classifier identifies the user from 20
   pixel-level features (intensity statistics + normalised 16-bin histogram) extracted
   from their facial image.
2. **Voiceprint Verification** — the spoken approval phrase is validated against a
   fixed whitelist, then a second Random Forest classifier confirms speaker identity
   from 32 MFCC-based acoustic features.
3. **Product Recommendation** — once both identity checks pass, a third Random Forest
   classifier predicts the product category the verified user is most likely to purchase,
   using features merged from their social media and transaction history.

If either biometric check fails, access is denied immediately and the product model is
never reached. All authentication events are written to a timestamped audit log.

---

## 2. System Architecture

```
User provides username + approval phrase
            |
            v
  [ Step 1 — Facial Recognition ]
    Loads mean pixel feature vector from image_features.csv
    → RandomForest face model predicts member identity
            |
            |-- FAIL --> ACCESS DENIED (face not recognised)
            |
            v
  [ Step 2 — Voiceprint Verification ]
    Validates phrase against approved whitelist
    → RandomForest voice model confirms speaker identity
    from mean MFCC vector in audio_features.csv
            |
            |-- FAIL --> ACCESS DENIED (invalid phrase or voice mismatch)
            |
            v
  [ Step 3 — Product Recommendation ]
    Samples user profile from merged_dataset.csv
    → RandomForest product model predicts purchase category
            |
            v
      ACCESS GRANTED + recommended product displayed
```

Every step prints its data source, inference method, and result. Every event is
appended to `outputs/simulation/simulation_log.txt`.

---

## 3. Repository Structure

```
Formative_2_Data_Preprocessing/
|
|-- app/
|   └── auth_simulation.py          CLI authentication and demo application
|
|-- assets/
|   |-- images/                     Raw facial images (9 files — 3 members × 3 expressions)
|   |   |-- chol_neutral.jpg
|   |   |-- chol_smile.jpg
|   |   |-- chol_surprised.jpg
|   |   |-- ineza_neutral.jpg
|   |   |-- ineza_smile.jpg
|   |   |-- ineza_surprised.jpg
|   |   |-- nziza_neutral.jpeg
|   |   |-- nziza_smile.jpeg
|   |   └── nziza_surprised.jpeg
|   └── audio/                      Raw audio recordings (6 files — 3 members × 2 phrases)
|       |-- chol_yes_approve.m4a
|       |-- chol_confirm_transaction.m4a
|       |-- ineza_yes_approve.m4a
|       |-- ineza_confirm_transaction.m4a
|       |-- nziza_yes_approve.m4a
|       └── nziza_confirm_transaction.m4a
|
|-- data/
|   |-- customer_social_profiles.csv    Source dataset 1 (155 rows, raw)
|   |-- customer_transactions.csv       Source dataset 2 (150 rows, raw)
|   └── merged_dataset.csv              Cleaned, merged, feature-engineered (213 × 11)
|
|-- features/
|   |-- image_features.csv          45 rows × 24 cols (9 images × 5 augmentations)
|   └── audio_features.csv          30 rows × 37 cols (6 clips × 5 augmentations)
|
|-- models/
|   |-- face_model.py               Standalone facial recognition training script
|   |-- voice_model.py              Standalone voiceprint verification training script
|   └── product_model.py            Standalone product recommendation training script
|
|-- notebooks/
|   └── Formative_2_Preprocessing.ipynb     Main notebook covering all 6 tasks
|
|-- outputs/
|   |-- eda/
|   |   |-- distributions.png               Feature histograms with KDE
|   |   |-- category_analysis.png           Product class frequency chart
|   |   |-- correlation_matrix.png          Pearson correlation heatmap
|   |   └── outliers.png                    Box plots for outlier detection
|   |-- image_processing/
|   |   |-- all_images.png                  3×3 grid of all nine facial images
|   |   |-- augmentations_all_members.png   Full augmentation matrix
|   |   └── feature_separability.png        Pairwise scatter plot of image features
|   |-- audio_processing/
|   |   |-- waveforms_and_spectrograms.png  Waveform + mel spectrogram for all 6 clips
|   |   └── augmentation_demo.png           Augmentation variants for one sample clip
|   |-- trained_models/
|   |   |-- face_model.pkl                  Serialised facial recognition model
|   |   |-- face_encoder.pkl                Label encoder for face classes
|   |   |-- voice_model.pkl                 Serialised voiceprint model
|   |   |-- voice_encoder.pkl               Label encoder for voice classes
|   |   |-- product_model.pkl               Serialised product recommendation model
|   |   |-- product_encoder.pkl             Label encoder for product classes
|   |   |-- model_metrics.json              All evaluation metrics (accuracy, F1, log loss)
|   |   |-- model_comparison.png            Side-by-side bar chart of all three models
|   |   |-- face_predictions.png            Predicted vs actual — face model
|   |   |-- voice_predictions.png           Predicted vs actual — voice model
|   |   └── product_predictions.png         Predicted vs actual — product model
|   └── simulation/
|       └── simulation_log.txt              Timestamped authentication event log
|
|-- requirements.txt
└── README.md
```

---

## 4. Setup and Installation

### Prerequisites

- Python 3.10 or higher
- `ffmpeg` — required by librosa to decode `.m4a` audio files

Install ffmpeg on macOS:
```bash
brew install ffmpeg
```

Install ffmpeg on Ubuntu/Debian:
```bash
sudo apt install ffmpeg
```

### Clone the repository

```bash
git clone https://github.com/Chol1000/Formative_2_Data_Preprocessing.git
cd Formative_2_Data_Preprocessing
```

### Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### Install Python dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation and numerical operations |
| `scikit-learn` | Random Forest classifiers, GridSearchCV, cross-validation |
| `joblib` | Model serialisation and loading |
| `matplotlib`, `seaborn` | EDA and evaluation plots |
| `opencv-python`, `Pillow` | Image loading, augmentation, feature extraction |
| `librosa`, `soundfile` | Audio loading, augmentation, acoustic feature extraction |
| `scipy` | Signal processing utilities |

---

## 5. Datasets

### Tabular data

| File | Rows | Description |
|---|---|---|
| `customer_social_profiles.csv` | 155 | Customer social media platform usage, engagement score, purchase interest, sentiment |
| `customer_transactions.csv` | 150 | Purchase amount, product category, customer rating |
| `merged_dataset.csv` | 213 × 11 | Inner-joined result with 3 engineered features added |

### Facial images

Nine images across three members and three required expressions stored in `assets/images/`:

| Member | Neutral | Smile | Surprised |
|---|---|---|---|
| chol | chol_neutral.jpg | chol_smile.jpg | chol_surprised.jpg |
| ineza | ineza_neutral.jpg | ineza_smile.jpg | ineza_surprised.jpg |
| nziza | nziza_neutral.jpeg | nziza_smile.jpeg | nziza_surprised.jpeg |

### Audio recordings

Six clips across three members and two approved phrases stored in `assets/audio/`:

| Member | "yes, approve" | "confirm transaction" |
|---|---|---|
| chol | chol_yes_approve.m4a | chol_confirm_transaction.m4a |
| ineza | ineza_yes_approve.m4a | ineza_confirm_transaction.m4a |
| nziza | nziza_yes_approve.m4a | nziza_confirm_transaction.m4a |

---

## 6. Task 1 — Data Merging and EDA

### Cleaning

Both datasets are cleaned independently before merging:

- Column names standardised to `snake_case`
- Social profiles: customer ID prefix `"A"` stripped and cast to integer
- Transactions: `transaction_id` and `purchase_date` dropped (not predictive)
- Numeric nulls filled with column median; exact duplicate rows removed

### Merge strategy

An **inner join** on `customer_id` is used so only customers present in both datasets
contribute training rows. The resulting 213 rows represent 61 unique customers across
multiple social platform and transaction combinations — multiple rows per customer are
intentional, as each captures a distinct behavioural context.

### Feature engineering

Three derived features are added after merging:

| Feature | Formula | Rationale |
|---|---|---|
| `purchase_per_engagement` | `purchase_amount / engagement_score` | Spend per unit of social activity |
| `high_interest` | 1 if `purchase_interest_score` > median else 0 | Binary purchase intent flag |
| `value_per_rating` | `purchase_amount / customer_rating` | Spend normalised by satisfaction |

Division guards of `1e-9` prevent zero-division errors. The final dataset has
**213 rows, 11 columns, zero null values**.

### EDA findings

- **Distributions** — engagement scores are uniformly distributed (50–99); purchase amounts
  are approximately normal (mean ≈ 290); the two ratio features are right-skewed as expected.
- **Class balance** — Sports is the most frequent category (59 rows, 28%); Clothing is the
  least frequent (33 rows, 15%). Ratio 1.79 — moderate imbalance handled with
  `class_weight="balanced"`.
- **Correlations** — strongest pairs: purchase_amount ↔ purchase_per_engagement (r = 0.87),
  purchase_interest_score ↔ high_interest (r = 0.86), customer_rating ↔ value_per_rating
  (r = −0.73). All feature-to-target correlations are below 0.10.
- **Outliers** — moderate outliers in `purchase_per_engagement` and `value_per_rating`
  (ratio amplification). No removal applied; Random Forest is robust to extreme values.

---

## 7. Task 2 — Facial Image Collection and Processing

### Augmentation

Each of the 9 base images is passed through 4 augmentation functions, producing 5 variants
per image and **45 total rows** in `features/image_features.csv`:

| Variant | Transformation |
|---|---|
| `original` | Unmodified |
| `rotated` | 15° clockwise rotation (`cv2.warpAffine`) |
| `flipped` | Horizontal mirror (`cv2.flip`) |
| `grayscale` | BGR → greyscale, replicated to 3 channels (`cv2.cvtColor`) |
| `brightened` | Pixel values increased by 30, clipped at 255 (NumPy) |

### Feature extraction

**20 numeric features** per image saved to `features/image_features.csv` (45 × 24):

- 4 intensity statistics: `mean_intensity`, `std_intensity`, `min_intensity`, `max_intensity`
- 16-bin normalised pixel histogram: `hist_bin_00` through `hist_bin_15`

The 4 metadata columns are `member`, `expression`, `file_name`, `augmentation`.

---

## 8. Task 3 — Audio Collection and Processing

### Augmentation

Each of the 6 base clips is passed through 4 augmentation functions, producing 5 variants
per clip and **30 total rows** in `features/audio_features.csv`:

| Variant | Transformation |
|---|---|
| `original` | Unmodified |
| `add_noise` | Gaussian noise at σ = 0.005, seed 42 |
| `time_stretch` | Duration stretched by factor 1.15, pitch unchanged |
| `pitch_shift_up` | Pitch shifted +2 semitones |
| `pitch_shift_dn` | Pitch shifted −2 semitones |

### Feature extraction

**32 acoustic features** per clip saved to `features/audio_features.csv` (30 × 37):

| Feature group | Count | Description |
|---|---|---|
| MFCC means | 13 | Mean of each Mel-Frequency Cepstral Coefficient track |
| MFCC standard deviations | 13 | Std of each MFCC track |
| Spectral roll-off | 2 | Mean and std of the 85% roll-off frequency |
| RMS energy | 2 | Mean and std of root-mean-square signal energy |
| Zero-crossing rate | 2 | Mean and std of sign changes per frame |

The 4 metadata columns are `member`, `phrase`, `augmentation`, `clip_path`, plus 1
`duration_s` column — 37 columns total, zero null values.

---

## 9. Task 4 — Model Development and Evaluation

All three models use `RandomForestClassifier` from scikit-learn, evaluated with
5-fold stratified cross-validation. Log loss is computed on a held-out 20% test split.
Final models are fitted on the full dataset after evaluation.

### Results

| Model | Script | Samples | Features | Accuracy (CV) | F1-Weighted | Log Loss |
|---|---|---|---|---|---|---|
| Facial Recognition | `models/face_model.py` | 45 | 20 | 0.9778 ± 0.0444 | 0.9771 ± 0.0457 | 0.1088 |
| Voiceprint Verification | `models/voice_model.py` | 30 | 32 | 0.9333 ± 0.1333 | 0.9333 ± 0.1333 | 0.4407 |
| Product Recommendation | `models/product_model.py` | 213 | 9 | 0.4504 ± 0.0363 | 0.4395 ± 0.0425 | 1.3335 |

The product model's 45.04% accuracy represents a **125% improvement over random chance**
(20% baseline for 5 classes) and a **63% improvement over the majority-class baseline**
(27.7% for always predicting Sports). The moderate accuracy reflects the weak feature-target
signal in the tabular data (max Pearson correlation 0.10), not a failure of the model.

### Hyperparameter tuning (product model)

`GridSearchCV` over 288 combinations (4 × 4 × 3 × 3 × 2) using F1-weighted:

| Parameter | Values |
|---|---|
| `n_estimators` | 200, 300, 500, 1000 |
| `max_depth` | None, 5, 10, 15 |
| `min_samples_leaf` | 1, 2, 4 |
| `max_features` | "sqrt", "log2", None |
| `class_weight` | "balanced", "balanced_subsample" |

Face and voice models use fixed `n_estimators=200`, `class_weight="balanced"`, `random_state=42`.

### Serialised outputs

All trained models are saved to `outputs/trained_models/` using `joblib`:

```
face_model.pkl     face_encoder.pkl
voice_model.pkl    voice_encoder.pkl
product_model.pkl  product_encoder.pkl
model_metrics.json
```

---

## 10. Task 6 — System Simulation

The system is demonstrated in two ways: inline notebook cells (Task 6 section) and the
standalone CLI application at `app/auth_simulation.py`.

### Simulation 1 — Authorised transactions

All three registered members pass all three gates and receive a product recommendation:

| Member | Phrase | Face | Voice | Product |
|---|---|---|---|---|
| chol | yes, approve | VERIFIED | VERIFIED | (model prediction) |
| ineza | confirm transaction | VERIFIED | VERIFIED | (model prediction) |
| nziza | yes, approve | VERIFIED | VERIFIED | (model prediction) |

### Simulation 2 — Unauthorised attempts

Two attack vectors are tested, each blocked at a different gate:

| Attack | Username | Phrase | Blocked at |
|---|---|---|---|
| A — Unknown user | `intruder_x` | `yes, approve` | Step 1 — face not in training data |
| B — Wrong phrase | `chol` | `open sesame` | Step 2 — phrase not in whitelist |

### Event log

Every event is appended to `outputs/simulation/simulation_log.txt`:

```
2026-03-19 15:23:14 | ACCESS_GRANTED | user=chol    | step=success      | product=Electronics
2026-03-19 15:26:06 | ACCESS_DENIED  | user=intruder_x | step=face_failed
2026-03-19 15:26:06 | ACCESS_DENIED  | user=chol    | step=voice_failed
```

---

## 11. Running the Pipeline

### Option A — Jupyter Notebook (recommended for full reproduction)

Open `notebooks/Formative_2_Preprocessing.ipynb` and select **Restart and Run All**.
The notebook executes all six tasks in sequence and saves all outputs automatically.

### Option B — Standalone model scripts

Retrain each model independently from the project root:

```bash
python3 models/face_model.py
python3 models/voice_model.py
python3 models/product_model.py
```

Each script reads from its feature file, trains the model, prints CV metrics to the
console, saves the `.pkl` files, and updates `model_metrics.json`.

---

## 12. Command-Line Application

### Launch the interactive menu

```bash
python3 app/auth_simulation.py
```

```
  1. Run authorised transaction simulation
  2. Run unauthorised attempt demo
  3. Show model metrics
  4. Show event log
  5. Exit
```

**Option 1** — Enter a username and approved phrase. The system runs all three pipeline
steps, prints the result at each gate, and shows the recommended product on success.

**Option 2** — Enter any username and phrase to test the security gates. The system shows
exactly which step blocks the attempt. You can retry as many times as needed.

**Option 3** — Prints accuracy, F1-weighted, and log loss for all three models from
`model_metrics.json`.

**Option 4** — Shows the last 20 authentication events from the audit log with a
summary count of granted and denied sessions.

### Non-interactive flags

```bash
# Full automated demo — all three members + both attack vectors, no prompts
python3 app/auth_simulation.py --auto

# Authorised simulation only
python3 app/auth_simulation.py --sim authorized

# Unauthorised attack demo only
python3 app/auth_simulation.py --sim denied
```

### Registered members and approved phrases

**Members:** `chol`, `ineza`, `nziza`

**Approved phrases (case-insensitive):**
- `yes, approve`
- `yes approve`
- `confirm transaction`

All input is handled case-insensitively — `CHOL`, `Yes Approve`, and `YES, APPROVE`
are all accepted.

---

## 13. Authentication Flow

```
Step 1 — Facial Recognition
  Input  : username
  Source : features/image_features.csv
  Method : Mean pixel feature vector → RandomForest classifier
  Pass   : predicted member == claimed username
  Fail   : ACCESS DENIED — face not recognised

Step 2 — Voiceprint Verification
  Input  : phrase + username
  Check  : phrase validated against whitelist {"yes, approve", "yes approve",
           "confirm transaction"}
  Source : features/audio_features.csv
  Method : Mean MFCC feature vector → RandomForest classifier
  Pass   : phrase is valid AND predicted member == claimed username
  Fail   : ACCESS DENIED — invalid phrase or voice mismatch

Step 3 — Product Recommendation
  Input  : username
  Source : data/merged_dataset.csv
  Method : Customer profile row → RandomForest classifier
  Output : Predicted product category (Books, Clothing, Electronics,
           Groceries, Sports)
  Result : ACCESS GRANTED + recommended product displayed
```

---

## 14. Team Contributions

This project was completed as a fully collaborative effort. All three members worked
together throughout every task — writing code, collecting data, debugging, and building
the models side by side. Each member took the lead on one area to ensure accountability
and direction.

| Member | Lead responsibility |
|---|---|
| **Chol Atem Giet Monykuch** | Task 1 (data merging, EDA, feature engineering), Task 6 (CLI simulation app), product recommendation model |
| **Ineza Melisa** | Task 3 (audio recording, augmentation, feature extraction), voiceprint verification model |
| **Nziza Aime Pacifique** | Task 2 (image collection, augmentation, feature extraction), facial recognition model |

All work in every section reflects the contribution of the full team.
