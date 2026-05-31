"""
explainability.py - Model explainability analysis using:
  1. Top-N coefficient bar charts   (works for lr and svm)
  2. SHAP LinearExplainer summary   (works for lr and svm)

Produces per-genre plots saved to plots/.

Usage:
    # Coefficient plots for Logistic Regression on two genres
    python src/explainability.py --model_path outputs/model_lr.pkl \
                                  --model_name lr \
                                  --genres "Comedies,Dramas,Documentaries"

    # Also run SHAP (slower, ~1 min for 100 test samples)
    python src/explainability.py --model_path outputs/model_lr.pkl \
                                  --model_name lr \
                                  --genres "Comedies,Dramas" \
                                  --shap
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.sparse import load_npz

from utils import load_artifact, ensure_dirs

OUTPUTS_DIR = "outputs"
PLOTS_DIR   = "plots"
SHAP_SAMPLE_SIZE = 100   # number of test rows used for SHAP (keep small for speed)

def get_feature_names() -> np.ndarray:
    """Concatenate TF-IDF and OHE feature names into one array."""
    tfidf   = load_artifact(os.path.join(OUTPUTS_DIR, "tfidf.pkl"))
    encoder = load_artifact(os.path.join(OUTPUTS_DIR, "encoder.pkl"))
    tfidf_features = tfidf.get_feature_names_out()
    cat_features   = encoder.get_feature_names_out()
    return np.concatenate([tfidf_features, cat_features])

def plot_top_coefficients(
    model,
    model_name: str,
    genre_name: str,
    genre_names: list,
    all_features: np.ndarray,
    top_n: int = 10,
) -> None:
    """
    Bar chart of the top-N features (by coefficient weight) for one genre.

    Works with any MultiOutputClassifier whose base estimator exposes .coef_.
    """
    if genre_name not in genre_names:
        print(f"[explainability] Genre '{genre_name}' not found. Skipping.")
        return

    genre_idx = genre_names.index(genre_name)
    estimator = model.estimators_[genre_idx]

    # .coef_ shape differs: LinearSVC → (1, n_features), LR → (1, n_features)
    coefs = estimator.coef_[0]

    top_indices  = np.argsort(coefs)[-top_n:]
    top_features = [all_features[i] for i in top_indices]
    top_weights  = [coefs[i] for i in top_indices]

    plt.figure(figsize=(10, 5))
    sns.barplot(
        x=top_weights, y=top_features,
        hue=top_features, palette="magma", legend=False
    )
    plt.title(f"Top {top_n} Features → '{genre_name}'  [{model_name}]")
    plt.xlabel("Coefficient Weight")
    plt.ylabel("Feature / Word")
    plt.tight_layout()

    safe_genre = genre_name.replace(" ", "_")
    path = os.path.join(PLOTS_DIR, f"coef_{model_name}_{safe_genre}.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[explainability] Saved → {path}")

def plot_shap_summary(
    model,
    model_name: str,
    genre_name: str,
    genre_names: list,
    all_features: np.ndarray,
) -> None:
    """
    SHAP LinearExplainer summary plot for one genre.

    Uses the first SHAP_SAMPLE_SIZE rows of X_test to keep it fast.
    Only compatible with linear models (LR, LinearSVC).
    """
    try:
        import shap
    except ImportError:
        print("[explainability] 'shap' not installed. Run: pip install shap")
        return

    if genre_name not in genre_names:
        print(f"[explainability] Genre '{genre_name}' not found. Skipping SHAP.")
        return

    X_test = load_npz(os.path.join(OUTPUTS_DIR, "X_test.npz"))
    X_train = load_npz(os.path.join(OUTPUTS_DIR, "X_train.npz"))
    X_sample = X_test[:SHAP_SAMPLE_SIZE]

    genre_idx = genre_names.index(genre_name)
    estimator = model.estimators_[genre_idx]

    print(f"[explainability] Computing SHAP for '{genre_name}' ({SHAP_SAMPLE_SIZE} samples) ...")
    explainer   = shap.LinearExplainer(estimator, X_train)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(12, 6))
    shap.summary_plot(
        shap_values,
        features=X_sample,
        feature_names=all_features,
        show=False,
        max_display=15,
    )
    safe_genre = genre_name.replace(" ", "_")
    plt.title(f"SHAP Summary — '{genre_name}'  [{model_name}]")
    path = os.path.join(PLOTS_DIR, f"shap_{model_name}_{safe_genre}.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[explainability] Saved → {path}")

def run_explainability(
    model_path: str,
    model_name: str,
    genres: list,
    run_shap: bool,
) -> None:
    ensure_dirs(PLOTS_DIR, OUTPUTS_DIR)

    model        = load_artifact(model_path)
    mlb          = load_artifact(os.path.join(OUTPUTS_DIR, "mlb.pkl"))
    genre_names  = list(mlb.classes_)
    all_features = get_feature_names()

    print(f"\n[explainability] Model : {model_name}")
    print(f"[explainability] Genres: {genres}")
    print(f"[explainability] Total features: {len(all_features)}")

    for genre in genres:
        genre = genre.strip()
        print(f"\n── Genre: {genre} ──")
        plot_top_coefficients(model, model_name, genre, genre_names, all_features)
        if run_shap:
            plot_shap_summary(model, model_name, genre, genre_names, all_features)

    print("\n[explainability] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explainability plots.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to saved model .pkl (e.g. outputs/model_lr.pkl).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Short label for the model (e.g. lr, svm).",
    )
    parser.add_argument(
        "--genres",
        type=str,
        default="Comedies,Dramas,Documentaries",
        help='Comma-separated genre names to explain (e.g. "Comedies,Dramas").',
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Also generate SHAP summary plots (slower, linear models only).",
    )
    args = parser.parse_args()
    genres = [g.strip() for g in args.genres.split(",")]
    run_explainability(args.model_path, args.model_name, genres, args.shap)
