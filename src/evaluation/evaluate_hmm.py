
"""Evaluate HMM predictions against the ground-truth route."""

import numpy as np
import pandas as pd

from src.data.loaders import load_trip
from src.features.candidates import build_features
from src.route_reconstruction.road_graph import RoadGraph


def evaluate_trip(root, trip_id, prediction_file):
    trip = load_trip(root, trip_id)

    # Generate ground-truth labels using Person A's implementation.
    print("Generating ground-truth labels...")
    features = build_features(
        trip,
        K=15,
        with_labels=True
    )

    true_arcs = features.attrs["true_arc"]
    true_dist = features.attrs["true_dist"]

    # Read HMM predictions.
    predictions = pd.read_csv(prediction_file)
    predictions = predictions.sort_values("point_idx")

    predicted_arcs = predictions["predicted_arc_id"].to_numpy(dtype=int)

    if len(predicted_arcs) != len(true_arcs):
        raise ValueError("Prediction count does not match ground-truth count.")

    # Nearest-candidate baseline.
    nearest = (
        features.sort_values(["point_idx", "dist_rank"])
        .groupby("point_idx", sort=True)
        .first()
    )

    baseline_arcs = nearest["candidate_arc_id"].to_numpy(dtype=int)

    # Check whether each true arc exists in the candidate set.
    candidate_found = (
        features.groupby("point_idx", sort=True)["label"]
        .max()
        .to_numpy(dtype=bool)
    )

    # Only evaluate GPS points whose ground-truth distance is <= 50 m.
    valid = true_dist <= 50.0

    hmm_correct = predicted_arcs == true_arcs
    baseline_correct = baseline_arcs == true_arcs

    def accuracy(correct, mask):
        if np.sum(mask) == 0:
            return 0.0
        return 100.0 * np.sum(correct & mask) / np.sum(mask)

    print("\n========== HMM EVALUATION ==========")
    print("Trip ID:", trip_id)
    print("Total GPS points:", len(true_arcs))
    print("Valid ground-truth points:", int(np.sum(valid)))

    print("\n---------- Candidate Recall ----------")
    print(
        "True arc found in top 15 candidates:",
        f"{100 * np.mean(candidate_found):.2f}%"
    )

    print("\n---------- HMM Accuracy ----------")
    print(
        "HMM accuracy on valid points:",
        f"{accuracy(hmm_correct, valid):.2f}%"
    )

    print(
        "HMM accuracy when true arc is a candidate:",
        f"{accuracy(hmm_correct, valid & candidate_found):.2f}%"
    )

    print("\n---------- Nearest Candidate Baseline ----------")
    print(
        "Baseline accuracy on valid points:",
        f"{accuracy(baseline_correct, valid):.2f}%"
    )

    # Check sequence continuity.
    graph = RoadGraph(trip)

    connected = []

    for i in range(len(predicted_arcs) - 1):
        arc1 = int(predicted_arcs[i])
        arc2 = int(predicted_arcs[i + 1])

        connected.append(
            arc1 == arc2 or graph.are_connected(arc1, arc2)
        )

    connectivity = 100.0 * np.mean(connected) if connected else 0.0

    print("\n---------- Route Continuity ----------")
    print("Consecutive arc connections:", f"{connectivity:.2f}%")

    # Save detailed evaluation results.
    report = pd.DataFrame({
        "point_idx": np.arange(len(true_arcs)),
        "true_arc_id": true_arcs,
        "predicted_arc_id": predicted_arcs,
        "baseline_arc_id": baseline_arcs,
        "true_distance": true_dist,
        "candidate_found": candidate_found,
        "valid_ground_truth": valid,
        "hmm_correct": hmm_correct,
        "baseline_correct": baseline_correct
    })

    report.to_csv("results/hmm_trip_1_evaluation.csv", index=False)

    print("\nSaved: results/hmm_trip_1_evaluation.csv")


if __name__ == "__main__":
    evaluate_trip(
        root="data/raw/train",
        trip_id=1,
        prediction_file="results/hmm_trip_1_predictions.csv"
    )