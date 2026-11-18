"""
make_dummy_features.py  –  generate synthetic features_train.parquet / features_test.parquet
==============================================================================================
Run this whenever A's real parquets haven't arrived yet.
Outputs land in data/processed/ so the pipeline can be tested end-to-end.

Usage
-----
    python make_dummy_features.py                   # defaults
    python make_dummy_features.py --n_train 500 --n_test 100 --n_arcs 80 --seed 7

The column schema is designed to match what A's feature pipeline will produce
(based on the PathSnap spec).  Extra columns from A are ignored; missing ones
will raise a clear error so you notice the mismatch early.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column schema  (keep in sync with A's features pipeline)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Geometry / distance features
    "dist_to_arc",          # perpendicular distance from GPS point to arc (metres)
    "dist_to_arc_start",    # distance to arc start node (metres)
    "dist_to_arc_end",      # distance to arc end node (metres)
    "arc_length",           # length of the arc in metres
    "bearing_diff",         # absolute difference between GPS heading and arc bearing (degrees)

    # Network features
    "node_degree_start",    # degree of arc start node in road graph
    "node_degree_end",      # degree of arc end node in road graph
    "is_oneway",            # 1 if arc is one-way, 0 otherwise

    # Temporal / sequence features
    "speed_mps",            # inferred speed at this GPS point (metres/second)
    "time_delta",           # seconds since previous GPS point
    "cumulative_dist",      # cumulative distance travelled so far in trip (metres)

    # Transition features (how likely is it to reach this arc from previous arc?)
    "transition_prob",      # log-probability of arc transition (from HMM or graph BFS)
]

LABEL_COL = "arc_id"       # integer arc identifier (ground truth)
META_COLS = ["trip_id", "point_idx"]   # not used as features


def make_split(
    n_rows: int,
    n_arcs: int,
    rng: np.random.Generator,
    split: str,
    n_trips: int = 20,
) -> pd.DataFrame:
    """Return a synthetic feature DataFrame with the canonical schema."""

    trip_ids = rng.integers(0, n_trips, size=n_rows)
    point_idxs = np.arange(n_rows)

    data: dict[str, np.ndarray] = {
        "trip_id": trip_ids,
        "point_idx": point_idxs,
        # label
        LABEL_COL: rng.integers(0, n_arcs, size=n_rows),
        # geometry
        "dist_to_arc": np.abs(rng.normal(loc=5.0, scale=8.0, size=n_rows)).clip(0, 100),
        "dist_to_arc_start": np.abs(rng.normal(loc=20.0, scale=15.0, size=n_rows)),
        "dist_to_arc_end": np.abs(rng.normal(loc=20.0, scale=15.0, size=n_rows)),
        "arc_length": rng.uniform(10, 300, size=n_rows),
        "bearing_diff": rng.uniform(0, 180, size=n_rows),
        # network
        "node_degree_start": rng.integers(1, 6, size=n_rows).astype(float),
        "node_degree_end": rng.integers(1, 6, size=n_rows).astype(float),
        "is_oneway": rng.integers(0, 2, size=n_rows).astype(float),
        # temporal
        "speed_mps": np.abs(rng.normal(loc=8.0, scale=4.0, size=n_rows)).clip(0, 40),
        "time_delta": rng.uniform(0.5, 5.0, size=n_rows),
        "cumulative_dist": np.cumsum(rng.uniform(5, 50, size=n_rows)),
        # transition
        "transition_prob": rng.uniform(-10, 0, size=n_rows),
    }

    df = pd.DataFrame(data)
    df["split"] = split
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy feature parquets for PathSnap.")
    parser.add_argument("--n_train", type=int, default=2000, help="Number of training rows")
    parser.add_argument("--n_test", type=int, default=400, help="Number of test rows")
    parser.add_argument("--n_arcs", type=int, default=100, help="Number of distinct arc IDs")
    parser.add_argument("--n_trips_train", type=int, default=40)
    parser.add_argument("--n_trips_test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.out_dir, "features_train.parquet")
    test_path = os.path.join(args.out_dir, "features_test.parquet")

    print(f"Generating dummy training set  ({args.n_train} rows, {args.n_arcs} arc classes) …")
    train_df = make_split(args.n_train, args.n_arcs, rng, "train", args.n_trips_train)
    train_df.to_parquet(train_path, index=False)
    print(f"  → {train_path}")

    print(f"Generating dummy test set      ({args.n_test} rows) …")
    test_df = make_split(args.n_test, args.n_arcs, rng, "test", args.n_trips_test)
    test_df.to_parquet(test_path, index=False)
    print(f"  → {test_path}")

    print("\nSchema:")
    print(train_df.dtypes.to_string())
    print(f"\nTrain shape : {train_df.shape}")
    print(f"Test  shape : {test_df.shape}")
    print(f"Arc classes : {train_df[LABEL_COL].nunique()}")
    print("\nDone ✓  –  Pipeline can now be tested before A's real features arrive.")


if __name__ == "__main__":
    main()
