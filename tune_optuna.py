from __future__ import annotations
import argparse
import json
import warnings
import joblib
import numpy as np
import optuna
from src.models.d_common import ARTIFACTS, FEATURE_COLS, PLOTS, RESULTS, Timer, append_result, full_report, load_folds, load_split, point_miss_rate, point_softmax, save_scores, split_by_fold
warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)

def space_xgb(t: optuna.Trial) -> dict:
    return dict(n_estimators=t.suggest_int('n_estimators', 200, 1500, step=100), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True), max_depth=t.suggest_int('max_depth', 3, 12), min_child_weight=t.suggest_float('min_child_weight', 0.5, 20, log=True), subsample=t.suggest_float('subsample', 0.5, 1.0), colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1.0), reg_lambda=t.suggest_float('reg_lambda', 0.001, 10, log=True), reg_alpha=t.suggest_float('reg_alpha', 0.001, 10, log=True), gamma=t.suggest_float('gamma', 0.001, 5, log=True))

def space_lgbm(t: optuna.Trial) -> dict:
    return dict(n_estimators=t.suggest_int('n_estimators', 100, 800, step=50), learning_rate=t.suggest_float('learning_rate', 0.01, 0.3, log=True), num_leaves=t.suggest_int('num_leaves', 15, 255, log=True), min_child_samples=t.suggest_int('min_child_samples', 10, 200, log=True), subsample=t.suggest_float('subsample', 0.5, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float('colsample_bytree', 0.5, 1.0), reg_lambda=t.suggest_float('reg_lambda', 0.001, 10, log=True), reg_alpha=t.suggest_float('reg_alpha', 0.001, 10, log=True))

def space_mlp(t: optuna.Trial) -> dict:
    n_layers = t.suggest_int('n_layers', 1, 3)
    width = t.suggest_categorical('width', [32, 64, 128, 256])
    return dict(hidden=[width // 2 ** i for i in range(n_layers)], dropout=t.suggest_float('dropout', 0.0, 0.5), lr=t.suggest_float('lr', 0.0001, 0.01, log=True), weight_decay=t.suggest_float('weight_decay', 1e-06, 0.001, log=True), batch_points=t.suggest_categorical('batch_points', [256, 512, 1024, 2048]))

def fit_predict(model_name, params, tr, va, ctx, trial=None):
    if model_name == 'xgb':
        from train_xgb_gpu import make_model, predict_proba_cpu
        m = make_model(params, ctx['device'], early_stopping=None)
        m.fit(tr[FEATURE_COLS].to_numpy(np.float32), tr['label'].to_numpy(), verbose=False)
        s = predict_proba_cpu(m, va[FEATURE_COLS].to_numpy(np.float32))
        m.set_params(device=ctx['device'])
        return (m, s)
    if model_name == 'lgbm':
        import lightgbm as lgb
        m = lgb.LGBMClassifier(**params, n_jobs=-1, random_state=42, verbose=-1)
        m.fit(tr[FEATURE_COLS].to_numpy(np.float32), tr['label'].to_numpy())
        return (m, m.predict_proba(va[FEATURE_COLS].to_numpy(np.float32))[:, 1])
    if model_name == 'mlp':
        from train_mlp import train_one
        cfg = {**ctx['mlp_base'], **params}
        scorer, _, _ = train_one(tr, va, cfg, ctx['device'], verbose=False, trial=trial)
        return (scorer, scorer.decision_function(va[FEATURE_COLS].to_numpy()))
    raise ValueError(model_name)

def build_params(model_name, best: dict) -> dict:
    if model_name == 'mlp':
        n, w = (best['n_layers'], best['width'])
        return {'hidden': [w // 2 ** i for i in range(n)], **{k: best[k] for k in ['dropout', 'lr', 'weight_decay', 'batch_points']}}
    p = dict(best)
    if model_name == 'lgbm':
        p['subsample_freq'] = 1
    return p

def save_plots(study, model_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm
    PLOTS.mkdir(parents=True, exist_ok=True)
    for kind, fn in [('history', ovm.plot_optimization_history), ('importance', ovm.plot_param_importances)]:
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, 'figure') else ax[0].figure
            fig.set_size_inches(8, 4.5)
            fig.tight_layout()
            fig.savefig(PLOTS / f'optuna_{model_name}_{kind}.png', dpi=150)
            plt.close(fig)
            print(f'  Plot -> results/plots/optuna_{model_name}_{kind}.png')
        except Exception as e:
            print(f'  (skipped {kind} plot: {e})')

def main():
    ap = argparse.ArgumentParser(description=None)
    ap.add_argument('--model', choices=['xgb', 'lgbm', 'mlp'], required=True, help='the model the team picked from results/model_comparison.csv')
    ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--timeout', type=int, default=None, help='seconds; stop the search after this')
    ap.add_argument('--sampler', choices=['tpe', 'gp'], default='tpe')
    ap.add_argument('--n_folds', type=int, default=5, help="use the first N of A's 5 folds")
    ap.add_argument('--trip_frac', type=float, default=1.0, help='fraction of training trips used during the SEARCH (final refit uses all)')
    ap.add_argument('--storage', default=None, help='e.g. sqlite:///results/optuna.db to resume')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()
    if args.smoke:
        args.trials, args.n_folds = (3, 2)
    print('=' * 60 + f'\n  Part D - Optuna ({args.sampler.upper()}) on {args.model}\n' + '=' * 60)
    ctx = {}
    if args.model == 'xgb':
        from train_xgb_gpu import xgb_device
        ctx['device'] = xgb_device() if args.device == 'auto' else args.device
    elif args.model == 'mlp':
        from train_mlp import pick_device
        ctx['device'] = pick_device(args.device)
        ctx['mlp_base'] = {'loss': 'listwise', 'epochs': 5 if args.smoke else 30, 'patience': 5, 'pos_weight': False, 'seed': args.seed}
    print(f"  device: {ctx.get('device', 'cpu')}")
    train = load_split('train', smoke=args.smoke)
    folds = load_folds()
    folds = folds[folds['trip_id'].isin(train['trip_id'].unique())].copy()
    if args.smoke:
        folds['fold'] = np.arange(len(folds)) % 2
    rng = np.random.default_rng(args.seed)
    search_trips = set(train['trip_id'].unique())
    if args.trip_frac < 1.0:
        tids = np.array(sorted(search_trips))
        search_trips = set(rng.choice(tids, max(2, int(len(tids) * args.trip_frac)), replace=False))
    search = train[train['trip_id'].isin(search_trips)].reset_index(drop=True)
    fold_ids = sorted(folds['fold'].unique())[:args.n_folds]
    print(f'  search data: {len(search):,} rows, {len(search_trips)} trips, folds {fold_ids}')
    space = {'xgb': space_xgb, 'lgbm': space_lgbm, 'mlp': space_mlp}[args.model]

    def objective(trial):
        params = space(trial)
        scores = []
        for step, f in enumerate(fold_ids):
            tr, va = split_by_fold(search, folds, f)
            if len(va) == 0 or len(tr) == 0:
                continue
            _, s = fit_predict(args.model, params, tr, va, ctx, trial=None)
            scores.append(point_miss_rate(va, s))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    sampler = optuna.samplers.GPSampler(seed=args.seed) if args.sampler == 'gp' else optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(study_name=f'pathsnap_{args.model}_{args.sampler}', direction='minimize', sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0), storage=args.storage, load_if_exists=True)
    with Timer() as t_search:
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=False)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f'\n  search time {t_search.secs / 60:.1f} min | {len(done)} complete, {len(study.trials) - len(done)} pruned/failed')
    print(f'  best CV miss-rate: {study.best_value:.4f}')
    print(f'  best params: {json.dumps(study.best_params, indent=2)}')
    RESULTS.mkdir(exist_ok=True)
    tag = args.model + ('_smoke' if args.smoke else '')
    study.trials_dataframe().to_csv(RESULTS / f'optuna_{tag}_trials.csv', index=False)
    best = build_params(args.model, study.best_params)
    with open(RESULTS / f'optuna_{tag}_best_params.json', 'w') as fh:
        json.dump(best, fh, indent=2)
    save_plots(study, tag)
    print('\n  Refitting best params on all training trips ...')
    test = load_split('test', smoke=args.smoke)
    if args.model == 'mlp':
        tr_all, va0 = split_by_fold(train, folds, fold_ids[0])
    else:
        tr_all, va0 = (train, train.iloc[:0])
    with Timer() as t_fit:
        model, _ = fit_predict(args.model, best, tr_all, va0 if len(va0) else tr_all.iloc[:15], ctx)
    Xte = test[FEATURE_COLS].to_numpy(np.float32)
    with Timer() as t_inf:
        if args.model == 'mlp':
            logits = model.decision_function(Xte)
            prob = 1 / (1 + np.exp(-logits.astype(np.float64)))
        elif args.model == 'xgb':
            from train_xgb_gpu import predict_proba_cpu
            prob = predict_proba_cpu(model, Xte)
        else:
            prob = model.predict_proba(Xte)[:, 1]
    if args.model != 'mlp':
        p = np.clip(prob, 1e-07, 1 - 1e-07)
        logits = np.log(p / (1 - p))
    name = f'{args.model}_tuned'
    res = full_report(name, test, prob, t_fit.secs, t_inf.secs)
    if args.smoke:
        print('  (smoke: nothing written to CSV or artifacts)')
        return
    append_result(res)
    ARTIFACTS.mkdir(exist_ok=True)
    if args.model == 'xgb':
        model.set_params(device='cpu')
    joblib.dump(model, ARTIFACTS / 'best_model.joblib')
    joblib.dump(model, ARTIFACTS / f'{name}.joblib')
    print(f'  Saved -> artifacts/best_model.joblib and artifacts/{name}.joblib')
    save_scores(test, prob, point_softmax(test, logits), name)
if __name__ == '__main__':
    main()
