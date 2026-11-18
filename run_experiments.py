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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, log_loss, accuracy_score
import lightgbm as lgb
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
from metrics import evaluate, append_to_comparison_csv
TRAIN_PATH = PROJECT_ROOT / 'data' / 'processed' / 'features_train.parquet'
TEST_PATH = PROJECT_ROOT / 'data' / 'processed' / 'features_test.parquet'
ARTIFACTS = PROJECT_ROOT / 'artifacts'
RESULTS = PROJECT_ROOT / 'results'
CV_PATH = PROJECT_ROOT / 'data' / 'processed' / 'cv_folds.parquet'
FEATURE_COLS = ['dist_to_arc', 'dist_rank', 'bearing_diff', 'heading_valid', 'speed', 'arc_length', 'proj_pos', 'dt_prev']

def candidate_miss_rate(df: pd.DataFrame, score_col: str='score') -> float:
    best = df.sort_values(score_col, ascending=False).groupby(['trip_id', 'point_idx'], sort=False).first().reset_index()
    return float(1.0 - best['label'].mean())

def top1_accuracy(df: pd.DataFrame, score_col: str='score') -> float:
    best = df.sort_values(score_col, ascending=False).groupby(['trip_id', 'point_idx'], sort=False).first().reset_index()
    return float(best['label'].mean())

def load_data(smoke: bool=False) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f'Train parquet not found: {TRAIN_PATH}')
    if not TEST_PATH.exists():
        raise FileNotFoundError(f'Test parquet not found: {TEST_PATH}')
    print('Loading parquets …')
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    if smoke:
        trips = train['trip_id'].unique()[:5]
        train = train[train['trip_id'].isin(trips)].copy()
        trips_t = test['trip_id'].unique()[:3]
        test = test[test['trip_id'].isin(trips_t)].copy()
        print(f'  [smoke] using {len(train):,} train / {len(test):,} test rows')
    else:
        print(f'  Train: {len(train):,} rows | Test: {len(test):,} rows')
    return (train, test)

def build_models(smoke: bool) -> dict:
    if smoke:
        return {'logreg': LogisticRegression(max_iter=200, C=1.0, solver='saga', n_jobs=-1), 'gnb': GaussianNB(), 'knn': KNeighborsClassifier(n_neighbors=5, n_jobs=-1), 'rf': RandomForestClassifier(n_estimators=30, max_depth=6, n_jobs=-1, random_state=42), 'lgbm': lgb.LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1, random_state=42, verbose=-1)}
    return {'logreg': LogisticRegression(max_iter=1000, C=1.0, solver='saga', n_jobs=-1), 'gnb': GaussianNB(), 'knn': KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1), 'rf': RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42), 'lgbm': lgb.LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1, colsample_bytree=0.8, subsample=0.8)}

def run(model_names: list[str] | None=None, smoke: bool=False, save_models: bool=False) -> list[dict]:
    print('=' * 60)
    print('  PathSnap — Model Training & Evaluation')
    print('=' * 60)
    train, test = load_data(smoke=smoke)
    X_train = train[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = train['label'].to_numpy()
    X_test = test[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_test = test['label'].to_numpy()
    print(f'Features : {X_train.shape[1]}  |  Positive rate train: {y_train.mean():.4f}')
    ARTIFACTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    models = build_models(smoke=smoke)
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}
    all_results = []
    for name, model in models.items():
        print(f"\n{'─' * 50}")
        print(f'  Training : {name}')
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_secs = time.perf_counter() - t0
        print(f'  Fit time : {train_secs:.1f}s')
        t1 = time.perf_counter()
        if hasattr(model, 'predict_proba'):
            scores = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else model.predict(X_test).astype(float)
        infer_secs = time.perf_counter() - t1
        test_scored = test[['trip_id', 'point_idx', 'label']].copy()
        test_scored['score'] = scores
        mr = candidate_miss_rate(test_scored)
        acc1 = top1_accuracy(test_scored)
        y_pred = (scores >= 0.5).astype(int)
        f1_mac = f1_score(y_test, y_pred, average='macro', zero_division=0)
        try:
            ll = log_loss(y_test, scores)
        except Exception:
            ll = float('nan')
        results = {'model': name, 'n_samples': len(y_test), 'miss_rate': mr, 'accuracy_at_1': acc1, 'accuracy_at_k': acc1, 'f1_macro': f1_mac, 'log_loss': ll, 'train_seconds': round(train_secs, 2), 'infer_seconds': round(infer_secs, 4)}
        print(f'  Miss-rate   : {mr:.4f}  ← primary metric')
        print(f'  Accuracy @1 : {acc1:.4f}')
        print(f'  F1 macro    : {f1_mac:.4f}')
        print(f'  Log-loss    : {ll:.4f}')
        print(f'  Infer time  : {infer_secs:.3f}s')
        all_results.append(results)
        append_to_comparison_csv(results, csv_path=str(RESULTS / 'model_comparison.csv'))
        if save_models:
            path = ARTIFACTS / f'{name}.joblib'
            joblib.dump(model, path)
            print(f'  Saved → {path}')
    print('\n' + '=' * 60)
    print('  Leaderboard  (miss-rate ↑ lower is better)')
    print('=' * 60)
    board = pd.DataFrame(all_results)[['model', 'miss_rate', 'accuracy_at_1', 'f1_macro', 'log_loss', 'train_seconds', 'infer_seconds']].sort_values('miss_rate')
    print(board.to_string(index=False))
    print()
    return all_results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run PathSnap model experiments.')
    parser.add_argument('--smoke', action='store_true', help='Quick run on first 5 trips only')
    parser.add_argument('--models', nargs='+', default=None, choices=['logreg', 'gnb', 'knn', 'rf', 'lgbm'], help='Subset of models to run (default: all 5)')
    parser.add_argument('--save_models', action='store_true', help='Persist trained models to artifacts/')
    args = parser.parse_args()
    run(model_names=args.models, smoke=args.smoke, save_models=args.save_models)
