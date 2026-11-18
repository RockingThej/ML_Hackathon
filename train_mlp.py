"""
train_mlp.py  -  Part D: PyTorch MLP candidate scorer with epoch-wise loss curves
=================================================================================
Two loss modes:
  listwise (default) : softmax cross-entropy over each GPS point's 15 candidates.
                       Directly optimises "is the top candidate the true arc?",
                       i.e. the miss-rate. Points with no true arc among the 15
                       candidates are skipped in the loss (nothing to learn).
  bce                : plain per-row binary cross-entropy (like C's classifiers).

Validation = one of A's trip-level folds (default fold 0), used for early stopping.

Outputs
  results/mlp_loss_history.csv           epoch, train_loss, val_loss, val_miss_rate
  results/plots/mlp_loss_curve.png       train vs val loss + val miss-rate
  results/model_comparison.csv           one row, same columns as C
  artifacts/mlp.joblib                   MLPScorer (API-loadable, has predict_proba)
  data/processed/scores_mlp_test.parquet per-candidate probabilities for B's HMM

Usage
  python train_mlp.py --smoke            # 2-minute sanity check on 5 trips
  python train_mlp.py                    # real run (GPU if available)
  python train_mlp.py --loss bce --name mlp_bce
  python train_mlp.py --cv               # all 5 folds, mean val miss-rate + curves per fold
"""
from __future__ import annotations

import argparse
import copy
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from src.models.d_common import (
    ARTIFACTS, C_TRAIN, C_VAL, FEATURE_COLS, K, PLOTS, RESULTS, Timer,
    append_result, check_fixed_k, full_report, load_folds, load_split,
    point_miss_rate, point_softmax, save_scores, split_by_fold,
)
from src.models.mlp import CandidateMLP, MLPScorer


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda":
        print(f"  Device: CUDA - {torch.cuda.get_device_name(0)}")
    else:
        print("  Device: CPU (no CUDA found - fine for --smoke, slow for the full run)")
    return dev


def to_grouped(df: pd.DataFrame, scaler: StandardScaler, device):
    """(P, K, F) features and (P, K) labels on the device."""
    X = scaler.transform(df[FEATURE_COLS].to_numpy(np.float64)).astype(np.float32)
    y = df["label"].to_numpy(np.float32)
    P = len(df) // K
    return (torch.from_numpy(X.reshape(P, K, -1)).to(device),
            torch.from_numpy(y.reshape(P, K)).to(device))


def loss_fn(logits, y, mode: str, pos_weight=None):
    if mode == "listwise":
        has_pos = y.sum(1) > 0
        if not has_pos.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[has_pos], y[has_pos].argmax(1))
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)


@torch.no_grad()
def predict_logits(model, X, batch=4096):
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)])


def train_one(train_df, val_df, cfg: dict, device, verbose=True, trial=None):
    """Train with early stopping on val loss. Returns (scorer, history_df, best_val_miss)."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    for d, nm in [(train_df, "train"), (val_df, "val")]:
        if not check_fixed_k(d):
            raise ValueError(f"{nm}: not exactly {K} candidates per point - listwise reshape impossible")

    scaler = StandardScaler().fit(train_df[FEATURE_COLS].to_numpy(np.float64))  # train rows only
    Xtr, ytr = to_grouped(train_df, scaler, device)
    Xva, yva = to_grouped(val_df, scaler, device)

    model = CandidateMLP(len(FEATURE_COLS), tuple(cfg["hidden"]), cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    pos_weight = None
    if cfg["loss"] == "bce" and cfg.get("pos_weight"):
        p = ytr.mean().item()
        pos_weight = torch.tensor((1 - p) / max(p, 1e-6), device=device)

    hist, best, best_state, bad = [], np.inf, None, 0
    P, bs = len(Xtr), cfg["batch_points"]
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(P, device=device)
        tot, n = 0.0, 0
        for i in range(0, P, bs):
            idx = perm[i:i + bs]
            loss = loss_fn(model(Xtr[idx]), ytr[idx], cfg["loss"], pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        train_loss = tot / n

        val_logits = predict_logits(model, Xva)
        val_loss = loss_fn(val_logits, yva, cfg["loss"], pos_weight).item()
        val_miss = point_miss_rate(val_df, val_logits.flatten().cpu().numpy())
        sched.step(val_loss)
        hist.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "val_miss_rate": val_miss, "lr": opt.param_groups[0]["lr"]})
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"val_miss={val_miss:.4f}")

        if val_loss < best - 1e-5:
            best, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["patience"]:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best val_loss={best:.4f})")
                break
        if trial is not None:  # Optuna pruning on the val miss-rate
            trial.report(val_miss, epoch)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

    model.load_state_dict(best_state)
    hist = pd.DataFrame(hist)
    best_miss = float(hist.loc[hist["val_loss"].idxmin(), "val_miss_rate"])
    return MLPScorer(model, scaler, FEATURE_COLS, cfg), hist, best_miss


def plot_curves(histories: dict, path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    multi = len(histories) > 1
    for i, (lbl, h) in enumerate(histories.items()):
        a = 0.35 + 0.65 * (i + 1) / len(histories) if multi else 1.0
        ax1.plot(h["epoch"], h["train_loss"], color=C_TRAIN, lw=2, alpha=a,
                 label="train" if i == 0 else None)
        ax1.plot(h["epoch"], h["val_loss"], color=C_VAL, lw=2, alpha=a,
                 label="validation" if i == 0 else None)
        ax2.plot(h["epoch"], h["val_miss_rate"], color=C_VAL, lw=2, alpha=a,
                 label=lbl if multi else "validation")
    ax1.set(title="Loss per epoch", xlabel="epoch", ylabel="loss")
    ax2.set(title="Validation miss-rate per epoch (lower is better)", xlabel="epoch",
            ylabel="miss-rate")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Loss curve -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loss", choices=["listwise", "bce"], default="listwise")
    ap.add_argument("--name", default=None, help="model name in the CSV (default mlp_<loss>)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_points", type=int, default=1024, help="GPS points per batch (x15 rows)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--pos_weight", action="store_true", help="bce only: weight positives by neg/pos")
    ap.add_argument("--val_fold", type=int, default=0)
    ap.add_argument("--cv", action="store_true", help="run all 5 folds (report only, no test eval)")
    ap.add_argument("--params", default=None, help="JSON file of tuned params (from tune_optuna.py)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_save", action="store_true", help="don't write CSV row / artifacts")
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in ["loss", "epochs", "batch_points", "lr", "weight_decay",
                                          "hidden", "dropout", "patience", "pos_weight", "seed"]}
    if args.params:
        cfg.update(json.load(open(args.params)))
        print(f"  Loaded tuned params: {cfg}")
    if args.smoke:
        cfg["epochs"] = min(cfg["epochs"], 5)
    name = args.name or f"mlp_{cfg['loss']}"

    print("=" * 60 + f"\n  Part D - MLP ({cfg['loss']})\n" + "=" * 60)
    device = pick_device(args.device)
    train = load_split("train", smoke=args.smoke)
    folds = load_folds()
    if args.smoke:  # make sure the smoke subset still has a val trip
        folds = folds[folds["trip_id"].isin(train["trip_id"].unique())].copy()
        folds["fold"] = np.where(folds["trip_id"] == folds["trip_id"].max(), args.val_fold, -1)

    RESULTS.mkdir(exist_ok=True)
    if args.cv:
        hists, rows = {}, []
        for f in sorted(folds["fold"].unique()):
            tr, va = split_by_fold(train, folds, f)
            print(f"\n--- fold {f}: {tr.trip_id.nunique()} train trips / {va.trip_id.nunique()} val trips")
            _, h, m = train_one(tr, va, cfg, device)
            hists[f"fold {f}"] = h
            rows.append({"fold": f, "best_val_miss_rate": m, "epochs_run": len(h)})
            h.to_csv(RESULTS / f"{name}_loss_history_fold{f}.csv", index=False)
        summ = pd.DataFrame(rows)
        summ.to_csv(RESULTS / f"{name}_cv_summary.csv", index=False)
        print("\n", summ.to_string(index=False))
        print(f"\n  CV miss-rate: {summ.best_val_miss_rate.mean():.4f} +/- {summ.best_val_miss_rate.std():.4f}")
        plot_curves(hists, PLOTS / f"{name}_loss_curve_cv.png", f"{name}: 5-fold CV")
        return

    tr, va = split_by_fold(train, folds, args.val_fold)
    print(f"  train: {len(tr):,} rows ({tr.trip_id.nunique()} trips) | "
          f"val (fold {args.val_fold}): {len(va):,} rows ({va.trip_id.nunique()} trips)")
    with Timer() as t_fit:
        scorer, hist, best_miss = train_one(tr, va, cfg, device)
    hist.to_csv(RESULTS / f"{name}_loss_history.csv", index=False)
    plot_curves({"run": hist}, PLOTS / f"{name}_loss_curve.png", f"{name} (val = fold {args.val_fold})")

    test = load_split("test", smoke=args.smoke)
    with Timer() as t_inf:
        logits = scorer.decision_function(test[FEATURE_COLS].to_numpy())
    prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    res = full_report(name, test, prob, t_fit.secs, t_inf.secs)

    if args.no_save or args.smoke:
        print("  (smoke / no_save: nothing written to CSV or artifacts)")
        return
    append_result(res)
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(scorer, ARTIFACTS / f"{name}.joblib")
    torch.save(scorer.model.state_dict(), ARTIFACTS / f"{name}.pt")
    print(f"  Saved -> artifacts/{name}.joblib (+ .pt)")
    save_scores(test, prob, point_softmax(test, logits), name)


if __name__ == "__main__":
    main()
