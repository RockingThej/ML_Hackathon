"""
src/models/d_common.py  -  shared helpers for Part D (MLP, XGBoost-GPU, Optuna)
==============================================================================
Everything here matches the schema A produced and the metric C uses, so D's rows
in results/model_comparison.csv are directly comparable with C's.

Schema (data/processed/features_{train,test}.parquet), one row per
(GPS point x candidate arc), exactly K=15 candidates per point:
    trip_id, point_idx, candidate_arc_id, <8 features>, label
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = PROCESSED / "features_train.parquet"
TEST_PATH = PROCESSED / "features_test.parquet"
FOLDS_PATH = PROCESSED / "cv_folds.parquet"
ARTIFACTS = PROJECT_ROOT / "artifacts"
RESULTS = PROJECT_ROOT / "results"
PLOTS = RESULTS / "plots"
COMPARISON_CSV = RESULTS / "model_comparison.csv"

# Same list, same order as src/features/candidates.py::FEATURES and C's run_experiments.py
FEATURE_COLS = [
    "dist_to_arc", "dist_rank", "bearing_diff", "heading_valid",
    "speed", "arc_length", "proj_pos", "dt_prev",
]
KEY_COLS = ["trip_id", "point_idx"]
K = 15

# Chart colours (validated categorical palette, slots 1-2)
C_TRAIN = "#2a78d6"   # blue
C_VAL = "#eb6834"     # orange


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_split(split: str, smoke: bool = False, n_smoke_trips: int = 5) -> pd.DataFrame:
    """Load train/test parquet, sorted so each point's K candidates are contiguous."""
    path = TRAIN_PATH if split == "train" else TEST_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Get A's parquet files (Drive/pendrive) and put them in data/processed/."
        )
    df = pd.read_parquet(path)
    missing = [c for c in FEATURE_COLS + KEY_COLS + ["label"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}. Schema changed? Ask A.")
    if smoke:
        keep = np.sort(df["trip_id"].unique())[: n_smoke_trips if split == "train" else 2]
        df = df[df["trip_id"].isin(keep)]
    df = df.sort_values(["trip_id", "point_idx", "dist_rank"], kind="stable").reset_index(drop=True)
    return df


def load_folds() -> pd.DataFrame:
    """A's trip-level folds: columns trip_id, fold (0..4). No trip is split across folds."""
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f"{FOLDS_PATH} not found - get it from A.")
    return pd.read_parquet(FOLDS_PATH)


def split_by_fold(train: pd.DataFrame, folds: pd.DataFrame, val_fold: int):
    val_trips = set(folds.loc[folds["fold"] == val_fold, "trip_id"])
    is_val = train["trip_id"].isin(val_trips).to_numpy()
    return train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)


def check_fixed_k(df: pd.DataFrame, k: int = K) -> bool:
    """True if every point has exactly k candidate rows (needed for listwise loss)."""
    sizes = df.groupby(KEY_COLS, sort=False).size()
    return bool((sizes == k).all())


# ---------------------------------------------------------------------------
# Metrics (same definition as C's run_experiments.candidate_miss_rate)
# ---------------------------------------------------------------------------
def point_miss_rate(df: pd.DataFrame, scores: np.ndarray) -> float:
    """Fraction of GPS points whose top-scored candidate is NOT the true arc.
    Points where candidate generation missed the true arc count as misses (as in C's code)."""
    tmp = df[KEY_COLS + ["label"]].copy()
    tmp["score"] = np.asarray(scores)
    best = tmp.loc[tmp.groupby(KEY_COLS, sort=False)["score"].idxmax()]
    return float(1.0 - best["label"].mean())


def full_report(name: str, df: pd.DataFrame, prob: np.ndarray,
                train_secs: float, infer_secs: float) -> dict:
    """Build a row with exactly the columns C writes to results/model_comparison.csv."""
    from sklearn.metrics import f1_score, log_loss

    y = df["label"].to_numpy()
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    mr = point_miss_rate(df, prob)
    res = {
        "model": name,
        "n_samples": len(y),
        "miss_rate": mr,
        "accuracy_at_1": 1.0 - mr,
        "accuracy_at_k": 1.0 - mr,
        "f1_macro": f1_score(y, (prob >= 0.5).astype(int), average="macro", zero_division=0),
        "log_loss": log_loss(y, prob, labels=[0, 1]),
        "train_seconds": round(train_secs, 2),
        "infer_seconds": round(infer_secs, 4),
    }
    print(f"\n  [{name}] miss-rate={res['miss_rate']:.4f}  acc@1={res['accuracy_at_1']:.4f}  "
          f"F1={res['f1_macro']:.4f}  logloss={res['log_loss']:.4f}")
    return res


def append_result(res: dict) -> None:
    from metrics import append_to_comparison_csv  # C's shared helper
    append_to_comparison_csv(res, csv_path=str(COMPARISON_CSV))


def point_softmax(df: pd.DataFrame, logits: np.ndarray) -> np.ndarray:
    """Normalise candidate scores into a probability distribution per GPS point.
    This is what B's HMM wants as emission probabilities."""
    s = pd.Series(np.asarray(logits, dtype=np.float64), index=df.index)
    g = s.groupby([df["trip_id"], df["point_idx"]], sort=False)
    e = np.exp(s - g.transform("max"))
    return (e / e.groupby([df["trip_id"], df["point_idx"]], sort=False).transform("sum")).to_numpy()


def save_scores(df: pd.DataFrame, prob: np.ndarray, point_prob: np.ndarray, name: str) -> Path:
    """Per-candidate scores for B's HMM. Written to data/processed/ (git-ignored, share via Drive)."""
    out = df[["trip_id", "point_idx", "candidate_arc_id"]].copy()
    out["prob"] = np.asarray(prob, dtype=np.float32)          # P(candidate is true arc)
    out["point_prob"] = np.asarray(point_prob, dtype=np.float32)  # sums to 1 per point
    path = PROCESSED / f"scores_{name}_test.parquet"
    out.to_parquet(path, index=False)
    print(f"  Scores for HMM -> {path}")
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.secs = time.perf_counter() - self.t0
