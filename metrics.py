"""
metrics.py  –  shared evaluation metrics for PathSnap
======================================================
Used by C (models), D (neural nets), and the API.

Primary metric  : candidate miss-rate  (lower = better)
                  fraction of GPS points where the true arc is NOT
                  in the top-k candidates produced by the model.

Secondary metrics: accuracy@1, accuracy@k, F1 (macro), log-loss,
                   and per-trip aggregates.

All public functions accept plain numpy arrays or pandas Series/DataFrames
so they can be called from any context (notebook, script, API).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix,
)
from typing import Sequence


# ---------------------------------------------------------------------------
# Core metric: candidate miss-rate
# ---------------------------------------------------------------------------

def miss_rate(
    y_true: np.ndarray | pd.Series,
    y_pred_topk: np.ndarray,
) -> float:
    """
    Fraction of samples where the true label is NOT in the top-k predictions.

    Parameters
    ----------
    y_true    : 1-D array of true arc IDs, shape (n,)
    y_pred_topk : 2-D array of predicted arc IDs, shape (n, k).
                  Each row is the top-k candidates (most likely first).

    Returns
    -------
    float in [0, 1]  –  lower is better.
    """
    y_true = np.asarray(y_true)
    y_pred_topk = np.asarray(y_pred_topk)
    if y_pred_topk.ndim == 1:
        y_pred_topk = y_pred_topk[:, np.newaxis]

    hits = np.any(y_pred_topk == y_true[:, np.newaxis], axis=1)
    return float(1.0 - hits.mean())


# ---------------------------------------------------------------------------
# Standard classification metrics (for the binary matched/missed framing)
# ---------------------------------------------------------------------------

def accuracy_at_1(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Fraction where top-1 prediction is correct."""
    return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))


def accuracy_at_k(
    y_true: np.ndarray | pd.Series,
    y_pred_topk: np.ndarray,
) -> float:
    """Fraction where true label is anywhere in the top-k predictions."""
    y_true = np.asarray(y_true)
    y_pred_topk = np.asarray(y_pred_topk)
    if y_pred_topk.ndim == 1:
        y_pred_topk = y_pred_topk[:, np.newaxis]
    hits = np.any(y_pred_topk == y_true[:, np.newaxis], axis=1)
    return float(hits.mean())


def f1(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    average: str = "macro",
) -> float:
    """Macro (or micro) F1 on top-1 predictions."""
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred),
                          average=average, zero_division=0))


def logloss(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    labels: Sequence | None = None,
) -> float:
    """
    Cross-entropy loss.

    Parameters
    ----------
    y_true  : 1-D int array of true class indices
    y_proba : 2-D float array of predicted probabilities, shape (n, n_classes)
    labels  : optional list of class labels matching columns of y_proba
    """
    kwargs = {"labels": labels} if labels is not None else {}
    return float(log_loss(np.asarray(y_true), np.asarray(y_proba), **kwargs))


# ---------------------------------------------------------------------------
# Per-trip aggregate (for final leaderboard-style reporting)
# ---------------------------------------------------------------------------

def per_trip_miss_rate(
    df: pd.DataFrame,
    trip_col: str = "trip_id",
    true_col: str = "true_arc",
    pred_topk_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute miss-rate per trip and return a summary DataFrame.

    Parameters
    ----------
    df            : DataFrame with one row per GPS point
    trip_col      : column name for trip identifier
    true_col      : column name for true arc ID
    pred_topk_cols: list of columns holding top-k predicted arc IDs.
                    If None, looks for columns named 'pred_1', 'pred_2', …

    Returns
    -------
    DataFrame with columns [trip_id, n_points, miss_rate]
    """
    if pred_topk_cols is None:
        pred_topk_cols = [c for c in df.columns if c.startswith("pred_")]

    records = []
    for trip_id, grp in df.groupby(trip_col):
        y_true = grp[true_col].to_numpy()
        y_topk = grp[pred_topk_cols].to_numpy()
        mr = miss_rate(y_true, y_topk)
        records.append({"trip_id": trip_id, "n_points": len(grp), "miss_rate": mr})

    return pd.DataFrame(records).sort_values("miss_rate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full evaluation report (printed + returned as dict)
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_pred_topk: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    model_name: str = "model",
    verbose: bool = True,
) -> dict:
    """
    Compute and optionally print all metrics.

    Parameters
    ----------
    y_true       : true arc IDs (1-D)
    y_pred       : top-1 predicted arc IDs (1-D)
    y_pred_topk  : top-k predictions, shape (n, k)  –  optional
    y_proba      : class probabilities, shape (n, n_classes)  –  optional
    model_name   : label for the report header
    verbose      : if True, print a formatted report

    Returns
    -------
    dict with keys: miss_rate, accuracy_at_1, accuracy_at_k, f1_macro, log_loss
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results: dict = {
        "model": model_name,
        "n_samples": len(y_true),
        "accuracy_at_1": accuracy_at_1(y_true, y_pred),
        "f1_macro": f1(y_true, y_pred),
    }

    if y_pred_topk is not None:
        results["miss_rate"] = miss_rate(y_true, y_pred_topk)
        results["accuracy_at_k"] = accuracy_at_k(y_true, y_pred_topk)
    else:
        # treat top-1 as top-k fallback
        results["miss_rate"] = 1.0 - results["accuracy_at_1"]
        results["accuracy_at_k"] = results["accuracy_at_1"]

    if y_proba is not None:
        try:
            results["log_loss"] = logloss(y_true, y_proba)
        except Exception:
            results["log_loss"] = float("nan")

    if verbose:
        sep = "=" * 50
        print(f"\n{sep}")
        print(f"  Evaluation report  –  {model_name}")
        print(sep)
        print(f"  Samples       : {results['n_samples']}")
        print(f"  Miss-rate     : {results['miss_rate']:.4f}  (primary metric)")
        print(f"  Accuracy @1   : {results['accuracy_at_1']:.4f}")
        print(f"  Accuracy @k   : {results['accuracy_at_k']:.4f}")
        print(f"  F1 (macro)    : {results['f1_macro']:.4f}")
        if "log_loss" in results:
            print(f"  Log-loss      : {results['log_loss']:.4f}")
        print(sep + "\n")

    return results


# ---------------------------------------------------------------------------
# results/model_comparison.csv helper
# ---------------------------------------------------------------------------

def append_to_comparison_csv(
    results: dict,
    csv_path: str = "results/model_comparison.csv",
) -> None:
    """
    Append a results dict as one row to the shared comparison CSV.
    Creates the file with a header if it doesn't exist.
    """
    import os, csv
    from datetime import datetime

    results = dict(results)
    results.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

    fieldnames = [
        "timestamp", "model", "n_samples",
        "miss_rate", "accuracy_at_1", "accuracy_at_k",
        "f1_macro", "log_loss",
    ]

    file_exists = os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Results appended to {csv_path}")


# ---------------------------------------------------------------------------
# Quick self-test  (python metrics.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 200
    n_classes = 50

    y_true = rng.integers(0, n_classes, size=n)
    y_pred = rng.integers(0, n_classes, size=n)
    y_topk = np.column_stack([y_pred] + [rng.integers(0, n_classes, size=n) for _ in range(4)])
    y_proba = rng.dirichlet(np.ones(n_classes), size=n)

    res = evaluate(y_true, y_pred, y_topk, y_proba, model_name="random_baseline")
    append_to_comparison_csv(res, csv_path="results/model_comparison.csv")
    print("Self-test passed ✓")
