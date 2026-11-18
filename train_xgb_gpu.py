from __future__ import annotations
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.d_common import ARTIFACTS, C_TRAIN, C_VAL, FEATURE_COLS, PLOTS, RESULTS, Timer, append_result, full_report, load_folds, load_split, point_softmax, save_scores, split_by_fold
DEFAULT_PARAMS = dict(n_estimators=2000, learning_rate=0.05, max_depth=8, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)

def xgb_device() -> str:
    import warnings
    try:
        X = np.random.rand(64, 4).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            m = xgb.XGBClassifier(n_estimators=2, device='cuda', tree_method='hist', verbosity=0)
            m.fit(X, (X[:, 0] > 0.5).astype(int))
        cfg = json.loads(m.get_booster().save_config())
        return 'cuda' if cfg['learner']['generic_param']['device'].startswith('cuda') else 'cpu'
    except Exception:
        return 'cpu'

def make_model(params: dict, device: str, early_stopping: int | None=50) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**params, objective='binary:logistic', eval_metric='logloss', tree_method='hist', device=device, early_stopping_rounds=early_stopping, random_state=42, n_jobs=-1, verbosity=0)

def predict_proba_cpu(model: xgb.XGBClassifier, X: np.ndarray) -> np.ndarray:
    model.set_params(device='cpu')
    return model.predict_proba(X)[:, 1]

def plot_learning_curve(ev: dict, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ev['validation_0']['logloss'], color=C_TRAIN, lw=2, label='train')
    ax.plot(ev['validation_1']['logloss'], color=C_VAL, lw=2, label='validation')
    ax.set(title='XGBoost log-loss per boosting round', xlabel='boosting round', ylabel='log-loss')
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Learning curve -> {path}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='xgb')
    ap.add_argument('--params', default=None, help='JSON of tuned params (overrides defaults)')
    ap.add_argument('--val_fold', type=int, default=0)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    args = ap.parse_args()
    params = dict(DEFAULT_PARAMS)
    if args.params:
        params.update(json.load(open(args.params)))
    if args.smoke:
        params['n_estimators'] = 100
    device = xgb_device() if args.device == 'auto' else args.device
    print('=' * 60 + f'\n  Part D - XGBoost  (device={device})\n' + '=' * 60)
    if device == 'cpu':
        print('  No CUDA build/GPU found -> CPU. On the 4050 laptop: pip install xgboost (>=2.0 has CUDA).')
    train = load_split('train', smoke=args.smoke)
    folds = load_folds()
    if args.smoke:
        folds = folds[folds['trip_id'].isin(train['trip_id'].unique())].copy()
        folds['fold'] = np.where(folds['trip_id'] == folds['trip_id'].max(), args.val_fold, -1)
    tr, va = split_by_fold(train, folds, args.val_fold)
    Xtr, ytr = (tr[FEATURE_COLS].to_numpy(np.float32), tr['label'].to_numpy())
    Xva, yva = (va[FEATURE_COLS].to_numpy(np.float32), va['label'].to_numpy())
    model = make_model(params, device)
    with Timer() as t_fit:
        model.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=200)
    print(f'  best iteration: {model.best_iteration}  fit: {t_fit.secs:.1f}s')
    ev = model.evals_result()
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame({'round': range(len(ev['validation_0']['logloss'])), 'train_logloss': ev['validation_0']['logloss'], 'val_logloss': ev['validation_1']['logloss']}).to_csv(RESULTS / f'{args.name}_learning_curve.csv', index=False)
    plot_learning_curve(ev, PLOTS / f'{args.name}_learning_curve.png')
    test = load_split('test', smoke=args.smoke)
    Xte = test[FEATURE_COLS].to_numpy(np.float32)
    with Timer() as t_inf:
        prob = predict_proba_cpu(model, Xte)
    res = full_report(args.name, test, prob, t_fit.secs, t_inf.secs)
    if args.smoke:
        print('  (smoke: nothing written to CSV or artifacts)')
        return
    append_result(res)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACTS / f'{args.name}.joblib')
    print(f'  Saved -> artifacts/{args.name}.joblib')
    logit = np.log(np.clip(prob, 1e-07, 1 - 1e-07) / np.clip(1 - prob, 1e-07, 1))
    save_scores(test, prob, point_softmax(test, logit), args.name)
if __name__ == '__main__':
    main()
