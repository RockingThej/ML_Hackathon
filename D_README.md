# Part D: neural net, GPU trees, Bayesian tuning, UI

Owner: D (RTX 3050, 6 GB) · branch `d-models`

## What's in this folder

| File | Purpose |
|---|---|
| `src/models/d_common.py` | Shared loaders, A's folds, C's miss-rate metric, CSV row in C's exact format |
| `src/models/mlp.py` | `CandidateMLP` (PyTorch) + `MLPScorer` (joblib-loadable, has `predict_proba`) |
| `train_mlp.py` | Trains the MLP with **epoch-wise train/val loss**, early stopping, loss-curve plot, test evaluation |
| `train_xgb_gpu.py` | XGBoost on CUDA (falls back to CPU automatically), learning curve, test evaluation |
| `tune_optuna.py` | **Bayesian optimisation** (TPE or GP) on A's CV folds, then refit + test + `artifacts/best_model.joblib` |
| `ui/app.py` | Streamlit demo: choose a trip → `/predict` → raw GPS vs matched route on a Folium map, plus a model-comparison tab |
| `docs/D_notes_for_team.md` | What C must add to the API (arcs endpoint, model order) and the score format for B's HMM |
| `requirements_D.txt` | Extra packages |

Every script writes its row to `results/model_comparison.csv` using **C's columns and C's miss-rate definition**, so the leaderboard stays comparable.

## Setup (RTX 3050 laptop)

```bash
git checkout -b d-models          # or: git checkout d-models
# 1. PyTorch with CUDA: copy the exact command from https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_D.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True
```
Copy A's files into `data/processed/` (they're git-ignored): `features_train.parquet`, `features_test.parquet`, `cv_folds.parquet`.

## Final setup (agreed 18 Nov)

| Item | Model / file |
|---|---|
| Leaderboard (`results/model_comparison.csv`) | C's 5 baselines + **mlp_listwise** + **xgb** (+ `xgb_tuned` after tuning) |
| Final model (API + HMM) | **XGBoost**: B's HMM is built on its scores; the tuned version is saved as `artifacts/best_model.joblib` |
| Best single classifier | `mlp_listwise` (miss-rate 0.1868), with epoch-wise loss curves in `results/plots/` |
| Removed | `mlp_bce` (not in the plan) |

## Run order

| # | Command | Time on the 4050 (rough) | Output |
|---|---|---|---|
| 1 | `python train_mlp.py --smoke` | < 1 min | checks the pipeline end to end, writes nothing |
| 2 | `python train_xgb_gpu.py --smoke` | < 1 min | same |
| 3 | `python train_mlp.py` | 1–15 min | `mlp_listwise` row, loss curve, `artifacts/mlp_listwise.joblib` |
| 5 | `python train_xgb_gpu.py` | 1–3 min | `xgb` row, learning curve, `artifacts/xgb.joblib` |
| 6 | **Team decision**: pick the best model from `results/model_comparison.csv` | | |
| 7 | `python tune_optuna.py --model <xgb\|lgbm\|mlp> --smoke` | 1–2 min | checks tuning |
| 8 | `python tune_optuna.py --model xgb --trials 60` | 20–40 min | best params, Optuna plots, `xgb_tuned` row, `artifacts/best_model.joblib` |
| 9 | `streamlit run ui/app.py` | | UI on Laptop 2 |

Tuning options:
- `--model lgbm` runs on the CPU. Add `--trip_frac 0.5` to make the search faster; the final refit still uses all trips.
- `--model mlp` is slow; use `--trials 25 --n_folds 3`.
- `--sampler gp` uses Gaussian-process Bayesian optimisation instead of TPE, if your evaluator wants "true" BO.
- `--storage sqlite:///results/optuna.db` lets you resume a search after a crash.
- `python train_mlp.py --cv` runs all 5 folds and saves per-fold loss curves for the report.

## After training: hand-offs

| To | What | How |
|---|---|---|
| C | `artifacts/best_model.joblib` (+ `xgb.joblib`) | Drive/pendrive (`*.joblib` is git-ignored). C puts `best_model.joblib` first in `get_model()` |
| C | `POST /trips/{id}/arcs` endpoint | snippet in `docs/D_notes_for_team.md` |
| B | `data/processed/scores_<model>_test.parquet` | Drive. Use `log(point_prob)` as the HMM emission score |
| Report | `results/plots/*.png`, `results/*_loss_history.csv`, `results/optuna_*` | committed with the code |

## What to say in the report (your sections)

| Section | Content |
|---|---|
| MLP | 8 standardised features → 128 → 64 → 1 logit, ReLU, dropout 0.2, AdamW, ReduceLROnPlateau, early stopping on validation loss (patience 8). Show the loss curve |
| Why train loss > val loss | Two reasons. Dropout is active while the training loss is computed but switched off for validation. And the validation fold's trips are easier, with fewer candidate misses. The gap is expected and is not a bug |
| MLP result | The listwise MLP (softmax over each point's 15 candidates) is the best single classifier (miss-rate 0.1868). XGBoost (0.1890) is the final model because B's HMM is built on its scores, and the gap is small |
| XGBoost-GPU | `tree_method=hist`, `device=cuda`, early stopping on a trip-level fold. Learning curve |
| Bayesian optimisation | Objective = mean CV miss-rate on A's trip-grouped folds (no trip in both train and validation), TPE sampler, median pruning, N trials. Optimisation-history and importance plots, default vs. tuned row |
| UI | Screenshot of the map (raw GPS dashed blue, matched route orange) and the comparison tab |

## Commit plan (small commits)

```bash
git add src/models/d_common.py         && git commit -m "D: shared data/metric helpers matching A's schema and C's miss-rate"
git add src/models/mlp.py              && git commit -m "D: PyTorch candidate MLP + sklearn-style scorer wrapper"
git add train_mlp.py                   && git commit -m "D: MLP training with epoch-wise train/val loss, early stopping, curves"
git add train_xgb_gpu.py               && git commit -m "D: XGBoost GPU model with CPU fallback and learning curve"
git add tune_optuna.py                 && git commit -m "D: Optuna Bayesian optimisation on trip-level CV folds"
git add ui/app.py                      && git commit -m "D: Streamlit UI - trip map matching + model comparison"
git add docs/D_notes_for_team.md requirements_D.txt D_README.md && git commit -m "D: docs, requirements, API/HMM hand-off notes"
git push -u origin d-models
# after each real run:
git add results/ && git commit -m "D: results - <model> run" && git push
```
