"""
run_experiments.py  –  train and evaluate all PathSnap models
==============================================================
Trains Logistic Regression, KNN, Random Forest, and LightGBM on
features_train.parquet and evaluates on features_test.parquet.
Results are logged to results/model_comparison.csv.

Usage
-----
    # Smoke-test on dummy data (auto-generates if parquets are missing)
    python run_experiments.py --smoke

    # Full run on real parquets from A
    python run_experiments.py

    # Run specific models only
    python run_experiments.py --models logreg rf

    # Save trained models to artifacts/
    python run_experiments.py --save_models
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# Project root is the directory containing this file
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics import evaluate, append_to_comparison_csv
from make_dummy_features import FEATURE_COLS, LABEL_COL

warnings.filterwarnings("ignore", category=UserWarning)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "features_train.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "features_test.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(smoke: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test parquets. If missing and smoke=True, generate dummies."""
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        if smoke:
            print("Parquets not found — generating dummy features for smoke test …")
            os.system(f'python "{PROJECT_ROOT / "make_dummy_features.py"}"')
        else:
            raise FileNotFoundError(
                f"Missing parquets.\n  train: {TRAIN_PATH}\n  test : {TEST_PATH}\n"
                "Run with --smoke to auto-generate dummies, or wait for A's files."
            )

    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    return train, test


def prepare_xy(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Extract feature matrix and encoded labels."""
    # Only keep feature columns that are present in this dataset
    available = [c for c in FEATURE_COLS if c in train.columns]
    missing = [c for c in FEATURE_COLS if c not in train.columns]
    if missing:
        print(f"  [warn] Missing feature columns (will skip): {missing}")

    X_train = train[available].to_numpy(dtype=np.float32)
    X_test = test[available].to_numpy(dtype=np.float32)

    le = LabelEncoder()
    y_train = le.fit_transform(train[LABEL_COL])
    y_test = le.transform(test[LABEL_COL])

    return X_train, y_train, X_test, y_test, le


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_models(smoke: bool) -> dict:
    """Return a dict of {name: sklearn-compatible model}."""
    if smoke:
        # Tiny models for the smoke test
        return {
            "logreg": LogisticRegression(max_iter=100, C=1.0, solver="saga",
                                          multi_class="auto", n_jobs=-1),
            "knn":    KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
            "rf":     RandomForestClassifier(n_estimators=20, max_depth=4,
                                              n_jobs=-1, random_state=42),
            "lgbm":   lgb.LGBMClassifier(n_estimators=50, num_leaves=15,
                                          n_jobs=-1, random_state=42, verbose=-1),
        }
    else:
        return {
            "logreg": LogisticRegression(max_iter=1000, C=1.0, solver="saga",
                                          multi_class="auto", n_jobs=-1),
            "knn":    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
            "rf":     RandomForestClassifier(n_estimators=200, max_depth=None,
                                              n_jobs=-1, random_state=42),
            "lgbm":   lgb.LGBMClassifier(n_estimators=500, num_leaves=63,
                                          learning_rate=0.05, n_jobs=-1,
                                          random_state=42, verbose=-1),
        }


# ---------------------------------------------------------------------------
# Training & evaluation loop
# ---------------------------------------------------------------------------

def run(
    model_names: list[str] | None = None,
    smoke: bool = False,
    save_models: bool = False,
) -> list[dict]:

    print("=" * 60)
    print("  PathSnap — Model Training & Evaluation")
    print("=" * 60)

    train, test = load_data(smoke=smoke)
    print(f"\nTrain : {train.shape[0]} rows  |  Test : {test.shape[0]} rows")

    X_train, y_train, X_test, y_test, le = prepare_xy(train, test)
    n_classes = len(le.classes_)
    print(f"Features : {X_train.shape[1]}  |  Arc classes : {n_classes}")

    models = build_models(smoke=smoke)
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}
        if not models:
            raise ValueError(f"No matching models. Choose from: {list(build_models(False).keys())}")

    all_results = []
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    for name, model in models.items():
        print(f"\n{'─' * 40}")
        print(f"  Training : {name}")
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_secs = time.perf_counter() - t0
        print(f"  Fit time : {train_secs:.1f}s")

        # Predictions
        y_pred = model.predict(X_test)

        # Top-k predictions (top 5 if model supports predict_proba)
        y_pred_topk = None
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            top_k = min(5, n_classes)
            topk_indices = np.argsort(y_proba, axis=1)[:, -top_k:][:, ::-1]
            y_pred_topk = le.classes_[topk_indices]

        # Evaluate
        results = evaluate(
            y_true=le.classes_[y_test],
            y_pred=le.classes_[y_pred],
            y_pred_topk=y_pred_topk,
            y_proba=y_proba,
            model_name=name,
        )
        results["train_seconds"] = round(train_secs, 2)
        all_results.append(results)

        # Save to comparison CSV
        append_to_comparison_csv(results, csv_path=str(RESULTS_DIR / "model_comparison.csv"))

        # Optionally persist model
        if save_models:
            model_path = ARTIFACTS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            print(f"  Model saved → {model_path}")

    # Final leaderboard
    print("\n" + "=" * 60)
    print("  Leaderboard (sorted by miss-rate ↑ lower is better)")
    print("=" * 60)
    board = pd.DataFrame(all_results)[["model", "miss_rate", "accuracy_at_1",
                                        "accuracy_at_k", "f1_macro"]]
    board = board.sort_values("miss_rate")
    print(board.to_string(index=False))
    print()

    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PathSnap model experiments.")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke-test on dummy data (auto-generated if missing)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["logreg", "knn", "rf", "lgbm"],
                        help="Subset of models to run (default: all)")
    parser.add_argument("--save_models", action="store_true",
                        help="Persist trained models to artifacts/ with joblib")
    args = parser.parse_args()

    run(model_names=args.models, smoke=args.smoke, save_models=args.save_models)
