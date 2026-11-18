from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, confusion_matrix
from typing import Sequence

def miss_rate(y_true: np.ndarray | pd.Series, y_pred_topk: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred_topk = np.asarray(y_pred_topk)
    if y_pred_topk.ndim == 1:
        y_pred_topk = y_pred_topk[:, np.newaxis]
    hits = np.any(y_pred_topk == y_true[:, np.newaxis], axis=1)
    return float(1.0 - hits.mean())

def accuracy_at_1(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))

def accuracy_at_k(y_true: np.ndarray | pd.Series, y_pred_topk: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred_topk = np.asarray(y_pred_topk)
    if y_pred_topk.ndim == 1:
        y_pred_topk = y_pred_topk[:, np.newaxis]
    hits = np.any(y_pred_topk == y_true[:, np.newaxis], axis=1)
    return float(hits.mean())

def f1(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, average: str='macro') -> float:
    return float(f1_score(np.asarray(y_true), np.asarray(y_pred), average=average, zero_division=0))

def logloss(y_true: np.ndarray | pd.Series, y_proba: np.ndarray, labels: Sequence | None=None) -> float:
    kwargs = {'labels': labels} if labels is not None else {}
    return float(log_loss(np.asarray(y_true), np.asarray(y_proba), **kwargs))

def per_trip_miss_rate(df: pd.DataFrame, trip_col: str='trip_id', true_col: str='true_arc', pred_topk_cols: list[str] | None=None) -> pd.DataFrame:
    if pred_topk_cols is None:
        pred_topk_cols = [c for c in df.columns if c.startswith('pred_')]
    records = []
    for trip_id, grp in df.groupby(trip_col):
        y_true = grp[true_col].to_numpy()
        y_topk = grp[pred_topk_cols].to_numpy()
        mr = miss_rate(y_true, y_topk)
        records.append({'trip_id': trip_id, 'n_points': len(grp), 'miss_rate': mr})
    return pd.DataFrame(records).sort_values('miss_rate', ascending=False).reset_index(drop=True)

def evaluate(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series, y_pred_topk: np.ndarray | None=None, y_proba: np.ndarray | None=None, model_name: str='model', verbose: bool=True) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    results: dict = {'model': model_name, 'n_samples': len(y_true), 'accuracy_at_1': accuracy_at_1(y_true, y_pred), 'f1_macro': f1(y_true, y_pred)}
    if y_pred_topk is not None:
        results['miss_rate'] = miss_rate(y_true, y_pred_topk)
        results['accuracy_at_k'] = accuracy_at_k(y_true, y_pred_topk)
    else:
        results['miss_rate'] = 1.0 - results['accuracy_at_1']
        results['accuracy_at_k'] = results['accuracy_at_1']
    if y_proba is not None:
        try:
            results['log_loss'] = logloss(y_true, y_proba)
        except Exception:
            results['log_loss'] = float('nan')
    if verbose:
        sep = '=' * 50
        print(f'\n{sep}')
        print(f'  Evaluation report  –  {model_name}')
        print(sep)
        print(f"  Samples       : {results['n_samples']}")
        print(f"  Miss-rate     : {results['miss_rate']:.4f}  (primary metric)")
        print(f"  Accuracy @1   : {results['accuracy_at_1']:.4f}")
        print(f"  Accuracy @k   : {results['accuracy_at_k']:.4f}")
        print(f"  F1 (macro)    : {results['f1_macro']:.4f}")
        if 'log_loss' in results:
            print(f"  Log-loss      : {results['log_loss']:.4f}")
        print(sep + '\n')
    return results

def append_to_comparison_csv(results: dict, csv_path: str='results/model_comparison.csv') -> None:
    import os
    from datetime import datetime
    results = dict(results)
    results.setdefault('timestamp', datetime.now().isoformat(timespec='seconds'))
    fieldnames = ['timestamp', 'model', 'n_samples', 'miss_rate', 'accuracy_at_1', 'accuracy_at_k', 'f1_macro', 'log_loss']
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)
    if os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        df = df[df['model'] != results['model']].copy()
        new_row = pd.DataFrame([{k: results.get(k, None) for k in fieldnames}])
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = pd.DataFrame([{k: results.get(k, None) for k in fieldnames}])
    df.sort_values('miss_rate', inplace=True)
    df.to_csv(csv_path, index=False)
    print(f'Results recorded in {csv_path} (deduplicated by model)')
if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n = 200
    n_classes = 50
    y_true = rng.integers(0, n_classes, size=n)
    y_pred = rng.integers(0, n_classes, size=n)
    y_topk = np.column_stack([y_pred] + [rng.integers(0, n_classes, size=n) for _ in range(4)])
    y_proba = rng.dirichlet(np.ones(n_classes), size=n)
    res = evaluate(y_true, y_pred, y_topk, y_proba, model_name='random_baseline')
    append_to_comparison_csv(res, csv_path='results/model_comparison.csv')
    print('Self-test passed ✓')
