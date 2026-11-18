
"""HMM map matching for PathSnap."""

import numpy as np
import pandas as pd

from src.data.loaders import load_trip
from src.features.candidates import build_features, prepare
from src.route_reconstruction.road_graph import RoadGraph
from src.hmm.viterbi import viterbi_decode_variable
from src.hmm.transitions import build_transition_log_probs


def build_emission_log_probs(features, sigma=20.0, heading_sigma=45.0):
    """Calculate emission log probabilities for candidate road arcs."""

    grouped = [
        group.sort_values("dist_rank").reset_index(drop=True)
        for _, group in features.groupby("point_idx", sort=True)
    ]

    emissions = []

    for group in grouped:
        distance = group["dist_to_arc"].to_numpy(dtype=float)
        bearing = group["bearing_diff"].to_numpy(dtype=float)
        heading_valid = group["heading_valid"].to_numpy(dtype=bool)

        distance_score = -0.5 * (distance / sigma) ** 2

        bearing_score = np.zeros(len(group), dtype=float)
        bearing_score[heading_valid] = (
            -0.5 * (bearing[heading_valid] / heading_sigma) ** 2
        )

        scores = distance_score + bearing_score
        scores -= np.max(scores)
        scores -= np.log(np.sum(np.exp(scores)))

        emissions.append(scores)

    return emissions


def build_emission_log_probs_from_model(features, model, K=15):
    """Emission = log P(candidate is true arc), normalised over the point's K candidates."""
    from src.features.candidates import FEATURES
    f = features.sort_values(["point_idx", "dist_rank"])
    p = model.predict_proba(f[FEATURES].to_numpy("float32"))[:, 1].reshape(-1, K)
    p = np.clip(p, 1e-6, None)
    p = p / p.sum(1, keepdims=True)
    return list(np.log(p))


def match_trip(root, trip_id, K=15, sigma=20.0,
               heading_sigma=45.0, beta=50.0, model=None):
    """Perform HMM map matching for one trip."""

    trip = load_trip(root, trip_id)

    print("Generating candidate features...")
    features = build_features(trip, K=K, with_labels=False)

    xy, _, _, _, _ = prepare(trip)

    print("Building road graph...")
    graph = RoadGraph(trip)

    print("Calculating emission probabilities...")
    if model is not None and hasattr(model, "predict_proba"):
        emissions = build_emission_log_probs_from_model(features, model, K=K)
    else:
        emissions = build_emission_log_probs(
            features, sigma=sigma, heading_sigma=heading_sigma
        )

    print("Calculating transition probabilities...")
    transitions = build_transition_log_probs(
        features, xy, graph, beta=beta
    )

    print("Running Viterbi decoding...")
    decoded_indices, best_score = viterbi_decode_variable(
        emissions, transitions
    )

    grouped = [
        group.sort_values("dist_rank").reset_index(drop=True)
        for _, group in features.groupby("point_idx", sort=True)
    ]

    predicted_arcs = []
    selected_distances = []
    selected_ranks = []

    for t, candidate_index in enumerate(decoded_indices):
        selected = grouped[t].iloc[int(candidate_index)]

        predicted_arcs.append(int(selected["candidate_arc_id"]))
        selected_distances.append(float(selected["dist_to_arc"]))
        selected_ranks.append(int(selected["dist_rank"]))

    result = pd.DataFrame({
        "trip_id": int(trip_id),
        "point_idx": np.arange(len(predicted_arcs)),
        "predicted_arc_id": predicted_arcs,
        "candidate_rank": selected_ranks,
        "gps_distance_to_arc": selected_distances
    })

    return result, decoded_indices, best_score


if __name__ == "__main__":
    print("Starting PathSnap HMM...")

    result, decoded, score = match_trip(
        root="data/raw/train",
        trip_id=1
    )

    print("\n========== HMM MAP MATCHING ==========")
    print("Trip ID: 1")
    print("GPS points decoded:", len(decoded))
    print("Predicted route arcs:", len(result))
    print("Best Viterbi log probability:", round(float(score), 4))

    print("\nFirst 10 predictions:")
    print(result.head(10).to_string(index=False))

    result.to_csv("hmm_trip_1_predictions.csv", index=False)

    print("\nSaved: hmm_trip_1_predictions.csv")