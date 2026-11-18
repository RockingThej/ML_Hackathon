import numpy as np
import pandas as pd

KEYS = ["trip_id", "point_idx", "candidate_arc_id"]


def load_scores(path):
    scores = pd.read_parquet(path)
    missing = set(KEYS + ["point_prob"]) - set(scores.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return scores


def build_emission_log_probs_ml(features, scores, trip_id,
                                eps=1e-12, weight=1.0, fallback=None):
    s = scores[scores["trip_id"] == int(trip_id)][
        ["point_idx", "candidate_arc_id", "point_prob"]
    ]
    if s.empty:
        raise KeyError(f"No scores found for trip {trip_id}")

    emissions = []
    n_fallback = 0
    for i, (point_idx, group) in enumerate(features.groupby("point_idx", sort=True)):
        group = group.sort_values("dist_rank").reset_index(drop=True)
        merged = group[["point_idx", "candidate_arc_id"]].merge(
            s, on=["point_idx", "candidate_arc_id"], how="left"
        )
        if merged["point_prob"].isna().any():
            if fallback is not None:
                emissions.append(np.asarray(fallback[i], dtype=float))
                n_fallback += 1
                continue
            raise ValueError(f"Trip {trip_id}, point {point_idx}: missing scores")
        p = np.clip(merged["point_prob"].to_numpy(dtype=float), eps, None)
        p = p / p.sum()
        emissions.append(weight * np.log(p))
    if n_fallback:
        print(f"  trip {trip_id}: {n_fallback} point(s) without ML scores -> Gaussian fallback")
    return emissions