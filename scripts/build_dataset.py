"""Build data/processed/features_{train,test}.parquet (+ sidecar files).

Run from the repo root:  python scripts/build_dataset.py
Scalers are fitted on TRAIN only and saved to models/scaler.joblib; the parquet
files are written UNSCALED (trees don't need scaling, the MLP applies the scaler).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.data.loaders import list_trips, load_trip
from src.features.candidates import build_features, FEATURES

RAW = Path("data/raw")
OUT = Path("data/processed")
K = 15
OUT.mkdir(parents=True, exist_ok=True)
Path("models").mkdir(exist_ok=True)


def build_partition(name):
    frames, side, stats = [], [], []
    for tid in list_trips(RAW / name):
        trip = load_trip(RAW / name, tid)
        df = build_features(trip, K=K, with_labels=True)
        true_arc, true_dist = df.attrs["true_arc"], df.attrs["true_dist"]
        n = len(true_arc)
        has = df.groupby("point_idx")["has_truth"].first().to_numpy().astype(bool)
        hit = df.groupby("point_idx")["label"].max().to_numpy().astype(bool)
        side.append(pd.DataFrame({
            "trip_id": np.int32(tid), "point_idx": np.arange(n, dtype=np.int32),
            "true_arc_id": true_arc.astype(np.int32),
            "true_dist": true_dist.astype(np.float32), "has_truth": has}))
        stats.append({"partition": name, "trip_id": tid, "n_points": n,
                      "n_labelled": int(has.sum()),
                      "miss_rate_labelled": float(1 - hit[has].mean()) if has.any() else np.nan})
        out = df[df["has_truth"] == 1].drop(columns="has_truth")
        out.attrs = {}
        frames.append(out)
        print(f"{name} {tid:3d}: pts={n:6d} labelled={has.mean():.2f} "
              f"miss={stats[-1]['miss_rate_labelled']:.4f}", flush=True)
    return pd.concat(frames, ignore_index=True), pd.concat(side), pd.DataFrame(stats)


train, side_tr, st_tr = build_partition("train")
test, side_te, st_te = build_partition("student_test")

# scaler on TRAIN only
scaler = StandardScaler().fit(train[FEATURES].to_numpy(dtype=np.float64))
joblib.dump({"scaler": scaler, "features": FEATURES}, "models/scaler.joblib")

# trip-level CV folds (never split a trip across folds)
trips = np.array(sorted(train.trip_id.unique()))
folds = np.zeros(len(trips), dtype=int)
for f, (_, va) in enumerate(GroupKFold(5).split(trips, groups=trips)):
    folds[va] = f
pd.DataFrame({"trip_id": trips.astype(np.int32), "fold": folds}).to_parquet(OUT / "cv_folds.parquet")

train.to_parquet(OUT / "features_train.parquet", index=False)
test.to_parquet(OUT / "features_test.parquet", index=False)
pd.concat([side_tr, side_te]).to_parquet(OUT / "point_truth.parquet", index=False)
pd.concat([st_tr, st_te]).to_csv(OUT / "candidate_miss_rate.csv", index=False)

for nm, d, st in [("train", train, st_tr), ("test", test, st_te)]:
    w = st.n_labelled.sum()
    print(f"\n{nm}: rows={len(d):,} points labelled={w}/{st.n_points.sum()} "
          f"positives={d.label.mean():.3f} overall miss="
          f"{(st.miss_rate_labelled*st.n_labelled).sum()/w:.4f}")
