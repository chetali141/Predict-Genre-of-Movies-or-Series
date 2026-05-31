"""
preprocess.py - Handles all data loading, exploratory data analysis (EDA), cleaning, 
feature engineering, and train/test splitting.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from wordcloud import WordCloud

from utils import save_artifact, ensure_dirs

PLOTS_DIR   = "plots"
OUTPUTS_DIR = "outputs"
MIN_GENRE_FREQUENCY = 5
TEST_SIZE   = 0.2
RANDOM_STATE = 42
TFIDF_MAX_FEATURES = 25000

def plot_missing_values(df: pd.DataFrame) -> None:
    """Bar chart of columns that have missing values."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values()
    if missing.empty:
        print("[preprocess] No missing values found.")
        return
    missing.plot.bar(figsize=(10, 5))
    plt.title("Missing Values in Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "missing_values.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[preprocess] Saved → {path}")


def plot_top_genres(df: pd.DataFrame, top_n: int = 25) -> None:
    """Horizontal bar chart of the top N genre categories."""
    df["listed_in"].value_counts().head(top_n).plot(kind="barh", figsize=(10, 6))
    plt.title(f"Top {top_n} Genre Categories")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "top_genres.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[preprocess] Saved → {path}")


def plot_type_distribution(df: pd.DataFrame) -> None:
    """Count plot of Movies vs TV Shows."""
    sns.countplot(x="type", data=df)
    plt.title("Movies vs TV Shows")
    path = os.path.join(PLOTS_DIR, "movies_vs_tvshows.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[preprocess] Saved → {path}")


def plot_release_year(df: pd.DataFrame) -> None:
    """Histogram of release year distribution."""
    sns.histplot(df["release_year"], bins=30)
    plt.title("Release Year Distribution")
    path = os.path.join(PLOTS_DIR, "release_year_distribution.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[preprocess] Saved → {path}")


def plot_platform_distribution(df: pd.DataFrame) -> None:
    """Count plot of content distribution across platforms."""
    sns.countplot(x="platform", data=df)
    plt.title("Content Distribution Across Platforms")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "content_distribution.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[preprocess] Saved → {path}")


def plot_wordcloud(df: pd.DataFrame) -> None:
    """Word cloud from all content descriptions."""
    text = " ".join(df["description"].astype(str))
    wc = WordCloud(width=1000, height=500, background_color="white").generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Most Frequent Words in Descriptions")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "wordcloud.png")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[preprocess] Saved → {path}")


def run_eda(df: pd.DataFrame) -> None:
    """Run and save all EDA plots."""
    print("\n[preprocess] Running EDA ...")
    print(f"  Dataset shape : {df.shape}")
    print(f"  Unique genres : {df['listed_in'].nunique()}")
    print(df.info())
    plot_missing_values(df)
    plot_top_genres(df)
    plot_type_distribution(df)
    plot_release_year(df)
    plot_platform_distribution(df)
    plot_wordcloud(df)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill nulls, build genre lists, filter rare genres.

    Returns a cleaned copy of the dataframe.
    """
    df = df.copy()

    # Fill nulls for categorical columns
    df["country"]  = df["country"].fillna("Unknown")
    df["rating"]   = df["rating"].fillna("Unknown")
    df["platform"] = df["platform"].fillna("Unknown")

    # Fill nulls for text columns
    df["director"] = df["director"].fillna("")
    df["cast"]     = df["cast"].fillna("")

    # Parse comma-separated genres into a list
    df["genre_list"] = df["listed_in"].apply(
        lambda x: [g.strip() for g in x.split(",")]
    )

    # Filter genres that appear fewer than MIN_GENRE_FREQUENCY times
    all_genres = [g for sublist in df["genre_list"] for g in sublist]
    genre_counts = Counter(all_genres)
    frequent = {g for g, c in genre_counts.items() if c >= MIN_GENRE_FREQUENCY}

    df["filtered_genre_list"] = df["genre_list"].apply(
        lambda x: [g for g in x if g in frequent]
    )

    # Drop rows where no genre survived the filter
    df = df[df["filtered_genre_list"].map(len) > 0].copy()

    print(f"[preprocess] After cleaning: {df.shape[0]} rows, {len(frequent)} unique genres")
    return df

def build_features(df: pd.DataFrame):
    """
    Combine text fields, encode labels, fit TF-IDF + OHE.

    Returns:
        X_train_final, X_test_final   : scipy sparse matrices
        y_train, y_test               : numpy arrays (multi-label binarized)
        tfidf, encoder, mlb           : fitted transformers
    """
    # --- Combined text feature ---
    X_text = (
        df["title"] + " " +
        df["director"] + " " +
        df["cast"] + " " +
        df["description"]
    )
    X_cat = df[["type", "rating", "platform", "country"]]

    # --- Multi-label target ---
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df["filtered_genre_list"])

    # --- Train / test split ---
    (X_text_train, X_text_test,
     X_cat_train,  X_cat_test,
     y_train,      y_test) = train_test_split(
        X_text, X_cat, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # --- TF-IDF on text ---
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_text_train_tfidf = tfidf.fit_transform(X_text_train)
    X_text_test_tfidf  = tfidf.transform(X_text_test)

    # --- One-hot encoding on categorical features ---
    encoder = OneHotEncoder(handle_unknown="ignore")
    X_cat_train_enc = encoder.fit_transform(X_cat_train)
    X_cat_test_enc  = encoder.transform(X_cat_test)

    # --- Combine text + categorical ---
    X_train_final = hstack([X_text_train_tfidf, X_cat_train_enc])
    X_test_final  = hstack([X_text_test_tfidf,  X_cat_test_enc])

    print(f"[preprocess] Train shape: {X_train_final.shape}, Test shape: {X_test_final.shape}")
    return X_train_final, X_test_final, y_train, y_test, tfidf, encoder, mlb


def main(data_path: str) -> None:
    ensure_dirs(PLOTS_DIR, OUTPUTS_DIR)

    # 1. Load
    print(f"[preprocess] Loading data from {data_path} ...")
    df = pd.read_csv(data_path)

    # 2. EDA
    run_eda(df)

    # 3. Clean
    df = clean_data(df)

    # 4. Build features
    X_train, X_test, y_train, y_test, tfidf, encoder, mlb = build_features(df)

    # 5. Persist splits and transformers
    save_npz(os.path.join(OUTPUTS_DIR, "X_train.npz"), X_train)
    save_npz(os.path.join(OUTPUTS_DIR, "X_test.npz"),  X_test)
    np.save(os.path.join(OUTPUTS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUTS_DIR, "y_test.npy"),  y_test)
    save_artifact(tfidf,   os.path.join(OUTPUTS_DIR, "tfidf.pkl"))
    save_artifact(encoder, os.path.join(OUTPUTS_DIR, "encoder.pkl"))
    save_artifact(mlb,     os.path.join(OUTPUTS_DIR, "mlb.pkl"))

    print("\n[preprocess] Done. All outputs saved to outputs/ and plots/ directories.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess TV/Movie dataset.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/tv-shows.csv",
        help="Path to the raw CSV dataset.",
    )
    args = parser.parse_args()
    main(args.data_path)
