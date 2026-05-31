"""
train.py - Train a multi-label genre classification model on preprocessed features.
"""

import argparse
import os
import time

import numpy as np
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

from utils import save_artifact, load_artifact, ensure_dirs


OUTPUTS_DIR = "outputs"

MODEL_REGISTRY = {
    "lr": {
        "description": "Logistic Regression",
        "build": lambda seed: LogisticRegression(
            max_iter=5000, C=50, class_weight="balanced"
        ),
    },
    "svm": {
        "description": "Linear SVC",
        "build": lambda seed: LinearSVC(
            max_iter=5000, C=50, class_weight="balanced"
        ),
    },
    "rf": {
        "description": "Random Forest",
        "build": lambda seed: RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=seed
        ),
    },
}


def load_splits():
    X_train = load_npz(os.path.join(OUTPUTS_DIR, "X_train.npz"))
    y_train = np.load(os.path.join(OUTPUTS_DIR, "y_train.npy"))
    print(f"[train] Loaded X_train {X_train.shape}, y_train {y_train.shape}")
    return X_train, y_train


def build_model(model_key: str, seed: int):
    if model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_key}'. Choose from: {list(MODEL_REGISTRY.keys())}"
        )
    entry = MODEL_REGISTRY[model_key]
    base = entry["build"](seed)
    model = MultiOutputClassifier(base, n_jobs=-1)
    print(f"[train] Model : {entry['description']}")
    return model


def train(model_key: str, seed: int):
    ensure_dirs(OUTPUTS_DIR)

    X_train, y_train = load_splits()
    model = build_model(model_key, seed)

    print(f"[train] Fitting {MODEL_REGISTRY[model_key]['description']} ...")
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f"[train] Training complete in {elapsed:.1f}s")

    out_path = os.path.join(OUTPUTS_DIR, f"model_{model_key}.pkl")
    save_artifact(model, out_path)
    print(f"[train] Model saved → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a multi-label genre classifier.")
    parser.add_argument(
        "--model",
        type=str,
        default="lr",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to train: lr | svm | rf  (default: lr)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    train(args.model, args.seed)
