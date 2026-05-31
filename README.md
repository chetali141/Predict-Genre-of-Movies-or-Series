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
python3 src/preprocess.py --data_path dataset/tv-shows.csv
```

Produces:

- plots/missing_values.png
- plots/top_genres.png
- plots/movies_vs_tvshows.png
- plots/release_year_distribution.png
- plots/content_distribution.png
- plots/wordcloud.png
- outputs/tfidf.pkl
- outputs/encoder.pkl
- outputs/mlb.pkl

### Step 2 — Train

Train each model separately. The `--seed` flag controls reproducibility. Saved to `outputs/model_lr.pkl`, `outputs/model_svm.pkl`, `outputs/model_rf.pkl`

```bash
python3 src/train.py --model lr  --seed 42
python3 src/train.py --model svm --seed 42
python3 src/train.py --model rf  --seed 42
```

Supported models:

1. lr - Logistic Regression (fast, strong baseline, SHAP-compatible)
2. svm - Linear SVC (fast, usually best accuracy on TF-IDF)
3. rf - Random Forest (slower, useful for non-linear patterns)

The trained model is saved to `outputs/model_<name>.pkl`.

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

Metrics reported: **Sample F1**, **Hamming Loss**, **Subset Accuracy**.

- Console: per-label classification report + summary metric table
- plots/eval_`<model_name>`_metrics.png: bar chart of all 4 metrics
- plots/eval_comparison.png: comparison across all evaluated models (auto-generated when --compare flag is set)

### Step 4 — Explainability

Generate coefficient bar charts and SHAP summary plots.

```bash
# Coefficient plots for Logistic Regression on 3 genres
python3 src/explainability.py --model_path outputs/model_lr.pkl --model_name lr --genres "Comedies,Dramas,Documentaries"

# Add --shap flag for SHAP summary plots (linear models only; ~1–2 min)
python3 src/explainability.py --model_path outputs/model_lr.pkl --model_name lr --genres "Comedies,Dramas,Documentaries" --shap
```

## Approach Summary

### Problem Statement

Genre prediction is treated as **multi-label classification** — each title can belong to multiple genres (e.g., "Dramas, International Movies, Romantic Movies").

### Features

- **Text**: title + director + cast + description, vectorized with TF-IDF (unigrams + bigrams, 25k features)
- **Categorical**: type, rating, platform, country — one-hot encoded

### Models

| Model | Strengths |
| ------- | ----------- |
| Logistic Regression | Fast, interpretable, SHAP-compatible |
| Linear SVC | Often best accuracy on TF-IDF sparse features |
| Random Forest | Non-linear patterns, feature importance via tree depth |

Each is wrapped in `MultiOutputClassifier`, which trains one binary classifier per genre label.

### Metrics

| Metric | What it measures |
| -------- | ----------------- |
| Sample F1 | Partial credit — rewards getting most labels right |
| Hamming Loss | Label-level error rate (lower is better) |
| Subset Accuracy | Exact full match — strictest measure |

### Explainability

- **Coefficient plots**: top-N words/features driving each genre prediction
- **SHAP LinearExplainer**: feature importance with directional impact (positive/negative)
