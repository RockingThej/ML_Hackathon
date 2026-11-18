"""
src/features/pipeline.py - End-to-end data processing and feature table generation.

Processes all 80 train trips and 10 student_test trips, generating:
- data/processed/features_train.parquet
- data/processed/features_test.parquet
- data/processed/candidate_miss_rate_summary.csv
- data/processed/cv_folds.json & cv_folds.csv
"""

import os
import sys

# Ensure repository root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.data.loaders import load_trip, get_trip_ids
from src.features.candidates import build_trip_features
from src.cv.splits import create_trip_grouped_cv_splits, save_cv_folds


def run_pipeline(
    data_dir: str = "data/raw",
    output_dir: str = "data/processed",
    radius_m: float = 50.0,
    max_candidates: int = 15,
    n_cv_splits: int = 5
) -> Dict[str, Any]:
    """
    Execute full feature pipeline and save all contract artifacts.
    """
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(_ROOT, data_dir)
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_ROOT, output_dir)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 60)
    print("STEP 1: Processing 80 Training Trips...")
    print("=" * 60)
    train_folders = get_trip_ids(data_dir, split="train")
    print(f"Found {len(train_folders)} train trips.")

    train_dfs = []
    miss_rate_records = []

    for idx, folder in enumerate(train_folders):
        t0 = time.time()
        trip = load_trip(folder, data_dir=data_dir, split="train")
        df_trip, miss_rate = build_trip_features(
            trip, radius_m=radius_m, max_candidates=max_candidates
        )
        train_dfs.append(df_trip)
        elapsed = time.time() - t0

        miss_rate_records.append({
            "split": "train",
            "trip_id": trip.trip_id,
            "folder": folder,
            "points": len(trip.track),
            "candidates_rows": len(df_trip),
            "miss_rate_pct": miss_rate,
            "time_sec": elapsed
        })

        if (idx + 1) % 10 == 0 or (idx + 1) == len(train_folders):
            print(f"[{idx+1:02d}/{len(train_folders):02d}] Trip {folder}: "
                  f"{len(df_trip)} candidates, miss rate = {miss_rate:.2f}%, {elapsed:.2f}s")

    df_train_all = pd.concat(train_dfs, ignore_index=True)
    train_parquet_path = os.path.join(output_dir, "features_train.parquet")
    df_train_all.to_parquet(train_parquet_path, index=False, engine="pyarrow")
    print(f"\nSaved {train_parquet_path} ({len(df_train_all):,} rows, {os.path.getsize(train_parquet_path)/1e6:.2f} MB)")

    print("\n" + "=" * 60)
    print("STEP 2: Processing 10 Student-Test Trips...")
    print("=" * 60)
    test_folders = get_trip_ids(data_dir, split="student_test")
    print(f"Found {len(test_folders)} student-test trips.")

    test_dfs = []
    for idx, folder in enumerate(test_folders):
        t0 = time.time()
        trip = load_trip(folder, data_dir=data_dir, split="student_test")
        df_trip, miss_rate = build_trip_features(
            trip, radius_m=radius_m, max_candidates=max_candidates
        )
        test_dfs.append(df_trip)
        elapsed = time.time() - t0

        miss_rate_records.append({
            "split": "student_test",
            "trip_id": trip.trip_id,
            "folder": folder,
            "points": len(trip.track),
            "candidates_rows": len(df_trip),
            "miss_rate_pct": miss_rate,
            "time_sec": elapsed
        })
        print(f"[{idx+1:02d}/{len(test_folders):02d}] Trip {folder}: "
              f"{len(df_trip)} candidates, miss rate = {miss_rate:.2f}%, {elapsed:.2f}s")

    df_test_all = pd.concat(test_dfs, ignore_index=True)
    test_parquet_path = os.path.join(output_dir, "features_test.parquet")
    df_test_all.to_parquet(test_parquet_path, index=False, engine="pyarrow")
    print(f"\nSaved {test_parquet_path} ({len(df_test_all):,} rows, {os.path.getsize(test_parquet_path)/1e6:.2f} MB)")

    print("\n" + "=" * 60)
    print("STEP 3: Saving Candidate Miss Rate Summary...")
    print("=" * 60)
    df_miss = pd.DataFrame(miss_rate_records)
    miss_summary_path = os.path.join(output_dir, "candidate_miss_rate_summary.csv")
    df_miss.to_csv(miss_summary_path, index=False)
    
    overall_train_miss = df_miss[df_miss["split"] == "train"]["miss_rate_pct"].mean()
    overall_test_miss = df_miss[df_miss["split"] == "student_test"]["miss_rate_pct"].mean()
    print(f"Mean Train Candidate Miss Rate: {overall_train_miss:.2f}%")
    print(f"Mean Student-Test Candidate Miss Rate: {overall_test_miss:.2f}%")
    print(f"Saved {miss_summary_path}")

    print("\n" + "=" * 60)
    print("STEP 4: Generating Group-Aware 5-Fold Cross Validation Splits...")
    print("=" * 60)
    train_trip_ids = [r["trip_id"] for r in miss_rate_records if r["split"] == "train"]
    folds = create_trip_grouped_cv_splits(train_trip_ids, n_splits=n_cv_splits)
    json_path, csv_path = save_cv_folds(folds, output_dir=output_dir)
    print(f"Saved {json_path} and {csv_path}")
    for f in folds:
        print(f"  Fold {f['fold']}: {f['num_train_trips']} train trips, {f['num_val_trips']} val trips")

    total_time = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE in {total_time:.2f}s ({total_time/60.0:.2f} min)!")
    print("=" * 60)

    return {
        "features_train_path": train_parquet_path,
        "features_test_path": test_parquet_path,
        "miss_summary_path": miss_summary_path,
        "cv_folds_json": json_path,
        "cv_folds_csv": csv_path,
        "train_rows": len(df_train_all),
        "test_rows": len(df_test_all),
        "mean_train_miss_rate": overall_train_miss,
        "mean_test_miss_rate": overall_test_miss,
        "elapsed_sec": total_time,
    }


if __name__ == "__main__":
    # Ensure root in sys.path
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    run_pipeline()
