"""
utils.py - helper functions for saving/loading models, encoders, and artifacts.
"""

import os
import pickle

def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved artifact → {path}")


def load_artifact(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded artifact ← {path}")
    return obj


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
