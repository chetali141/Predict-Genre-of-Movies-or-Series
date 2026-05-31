# Predict-Genre-of-Movies-or-Series

A Machine Learning Model that predicts the genre of a TV series or movie using TF-IDF text features combined with categorical metadata. Three scikit-learn models are trained and compared: Logistic Regression, Linear SVC, and Random Forest.

Dataset used: [Dataset-TV-Shows-OTT](https://github.com/vinayak-ensemble/Dataset-TV-Shows-OTT)

## Setup

```bash
# 1. Clone or unzip the project
cd genre_prediction

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip3 install -r requirements.txt
```

## Running the pipeline

### Step 1 - Preprocess Data

Loads the raw CSV, runs EDA (saves plots), cleans data, builds TF-IDF + OHE features, and saves train/test splits.

```bash
python3 src/preprocess.py --data_path data/tv-shows.csv
```

### Step 2 — Train

Train each model separately. The `--seed` flag controls reproducibility. Saved to `outputs/model_lr.pkl`, `outputs/model_svm.pkl`, `outputs/model_rf.pkl`

```bash
python3 src/train.py --model lr  --seed 42
python3 src/train.py --model svm --seed 42
python3 src/train.py --model rf  --seed 42
```

### Step 3 - Evaluate

Evaluate each model individually, then produce the side-by-side comparison chart.

```bash
# Evaluate each model
python3 src/evaluate.py --model_path outputs/model_lr.pkl  --model_name lr
python3 src/evaluate.py --model_path outputs/model_svm.pkl --model_name svm
python3 src/evaluate.py --model_path outputs/model_rf.pkl  --model_name rf

# Generate comparison chart (after evaluating all three)
python3 src/evaluate.py --compare
```

Metrics reported: **Sample F1**, **Hamming Loss**, **Subset Accuracy**

### Step 4 — Explainability

Generate coefficient bar charts and (optionally) SHAP summary plots.

```bash
# Coefficient plots for Logistic Regression on 3 genres
python3 src/explainability.py --model_path outputs/model_lr.pkl --model_name lr --genres "Comedies,Dramas,Documentaries"

# Add --shap flag for SHAP summary plots (linear models only; ~1–2 min)
python3 src/explainability.py --model_path outputs/model_lr.pkl --model_name lr --genres "Comedies,Dramas,Documentaries" --shap
```
