from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
TRAIN_PATH = PROCESSED / 'features_train.parquet'
TEST_PATH = PROCESSED / 'features_test.parquet'
FOLDS_PATH = PROCESSED / 'cv_folds.parquet'
ARTIFACTS = PROJECT_ROOT / 'artifacts'
RESULTS = PROJECT_ROOT / 'results'
PLOTS = RESULTS / 'plots'
COMPARISON_CSV = RESULTS / 'model_comparison.csv'
FEATURE_COLS = ['dist_to_arc', 'dist_rank', 'bearing_diff', 'heading_valid', 'speed', 'arc_length', 'proj_pos', 'dt_prev']
KEY_COLS = ['trip_id', 'point_idx']
K = 15
C_TRAIN = '#2a78d6'
C_VAL = '#eb6834'

def load_split(split: str, smoke: bool=False, n_smoke_trips: int=5) -> pd.DataFrame:
    path = TRAIN_PATH if split == 'train' else TEST_PATH
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Get A's parquet files (Drive/pendrive) and put them in data/processed/.")
    df = pd.read_parquet(path)
    missing = [c for c in FEATURE_COLS + KEY_COLS + ['label'] if c not in df.columns]
    if missing:
        raise ValueError(f'{path.name} is missing columns {missing}. Schema changed? Ask A.')
    if smoke:
        keep = np.sort(df['trip_id'].unique())[:n_smoke_trips if split == 'train' else 2]
        df = df[df['trip_id'].isin(keep)]
    df = df.sort_values(['trip_id', 'point_idx', 'dist_rank'], kind='stable').reset_index(drop=True)
    return df

def load_folds() -> pd.DataFrame:
    if not FOLDS_PATH.exists():
        raise FileNotFoundError(f'{FOLDS_PATH} not found - get it from A.')
    return pd.read_parquet(FOLDS_PATH)

def split_by_fold(train: pd.DataFrame, folds: pd.DataFrame, val_fold: int):
    val_trips = set(folds.loc[folds['fold'] == val_fold, 'trip_id'])
    is_val = train['trip_id'].isin(val_trips).to_numpy()
    return (train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True))

def check_fixed_k(df: pd.DataFrame, k: int=K) -> bool:
    sizes = df.groupby(KEY_COLS, sort=False).size()
    return bool((sizes == k).all())

def point_miss_rate(df: pd.DataFrame, scores: np.ndarray) -> float:
    tmp = df[KEY_COLS + ['label']].copy()
    tmp['score'] = np.asarray(scores)
    best = tmp.loc[tmp.groupby(KEY_COLS, sort=False)['score'].idxmax()]
    return float(1.0 - best['label'].mean())

def full_report(name: str, df: pd.DataFrame, prob: np.ndarray, train_secs: float, infer_secs: float) -> dict:
    from sklearn.metrics import f1_score, log_loss
    y = df['label'].to_numpy()
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-07, 1 - 1e-07)
    mr = point_miss_rate(df, prob)
    res = {'model': name, 'n_samples': len(y), 'miss_rate': mr, 'accuracy_at_1': 1.0 - mr, 'accuracy_at_k': 1.0 - mr, 'f1_macro': f1_score(y, (prob >= 0.5).astype(int), average='macro', zero_division=0), 'log_loss': log_loss(y, prob, labels=[0, 1]), 'train_seconds': round(train_secs, 2), 'infer_seconds': round(infer_secs, 4)}
    print(f"\n  [{name}] miss-rate={res['miss_rate']:.4f}  acc@1={res['accuracy_at_1']:.4f}  F1={res['f1_macro']:.4f}  logloss={res['log_loss']:.4f}")
    return res

def append_result(res: dict) -> None:
    from metrics import append_to_comparison_csv
    append_to_comparison_csv(res, csv_path=str(COMPARISON_CSV))

def point_softmax(df: pd.DataFrame, logits: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(logits, dtype=np.float64), index=df.index)
    g = s.groupby([df['trip_id'], df['point_idx']], sort=False)
    e = np.exp(s - g.transform('max'))
    return (e / e.groupby([df['trip_id'], df['point_idx']], sort=False).transform('sum')).to_numpy()

def save_scores(df: pd.DataFrame, prob: np.ndarray, point_prob: np.ndarray, name: str) -> Path:
    out = df[['trip_id', 'point_idx', 'candidate_arc_id']].copy()
    out['prob'] = np.asarray(prob, dtype=np.float32)
    out['point_prob'] = np.asarray(point_prob, dtype=np.float32)
    path = PROCESSED / f'scores_{name}_test.parquet'
    out.to_parquet(path, index=False)
    print(f'  Scores for HMM -> {path}')
    return path

class Timer:

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.secs = time.perf_counter() - self.t0
