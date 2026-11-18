"""
features.py - Feature engineering for candidate road segment classification.

Extracts spatial, dynamic, geometric, topological, and candidate-relative features
for each (GPS observation, candidate road segment) pair.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd


def compute_trajectory_dynamics(
    track_x: np.ndarray,
    track_y: np.ndarray,
    timestamps: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute instantaneous speed, acceleration, and heading for GPS points.
    Uses central differences where available, with forward/backward differences at edges.
    """
    N = len(track_x)
    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)
    dt = np.ones(N, dtype=np.float64)

    # Time difference
    dt[1:] = np.maximum(0.1, timestamps[1:] - timestamps[:-1])
    dt[0] = dt[1] if N > 1 else 1.0

    # Coordinate differences
    if N > 1:
        dx[1:] = track_x[1:] - track_x[:-1]
        dy[1:] = track_y[1:] - track_y[:-1]
        dx[0] = dx[1]
        dy[0] = dy[1]

    # Speeds
    step_dist = np.hypot(dx, dy)
    speed = step_dist / dt

    # Accelerations
    accel = np.zeros(N, dtype=np.float64)
    if N > 1:
        accel[1:] = (speed[1:] - speed[:-1]) / dt[1:]
        accel[0] = accel[1]

    # Vehicle heading (bearing) in radians [-pi, pi]
    veh_heading = np.arctan2(dy, dx)

    return {
        "dt": dt,
        "speed": speed,
        "accel": accel,
        "veh_heading": veh_heading
    }


def compute_network_topology(arcs: np.ndarray) -> Dict[str, Any]:
    """
    Precompute node degree and arc reversibility for the road network.
    """
    out_degrees = {}
    in_degrees = {}
    arc_set = set()

    for u, v in arcs:
        u_int, v_int = int(u), int(v)
        out_degrees[u_int] = out_degrees.get(u_int, 0) + 1
        in_degrees[v_int] = in_degrees.get(v_int, 0) + 1
        arc_set.add((u_int, v_int))

    # Bidirectional flag: does (v, u) also exist in network?
    is_bidirectional = np.zeros(len(arcs), dtype=np.int8)
    for i, (u, v) in enumerate(arcs):
        if (int(v), int(u)) in arc_set:
            is_bidirectional[i] = 1

    return {
        "out_degrees": out_degrees,
        "in_degrees": in_degrees,
        "is_bidirectional": is_bidirectional
    }


def extract_candidate_features(
    trip_id: str,
    track_x: np.ndarray,
    track_y: np.ndarray,
    timestamps: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    arcs: np.ndarray,
    candidates_per_obs: List[List[Dict[str, Any]]],
    topology: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Build a comprehensive tabular feature matrix for all (obs_idx, candidate) pairs in a trip.
    """
    if topology is None:
        topology = compute_network_topology(arcs)

    dynamics = compute_trajectory_dynamics(track_x, track_y, timestamps)
    out_deg = topology["out_degrees"]
    in_deg = topology["in_degrees"]
    is_bidir = topology["is_bidirectional"]

    rows = []
    num_obs = len(candidates_per_obs)

    for obs_idx in range(num_obs):
        px = track_x[obs_idx]
        py = track_y[obs_idx]
        v_speed = dynamics["speed"][obs_idx]
        v_accel = dynamics["accel"][obs_idx]
        v_heading = dynamics["veh_heading"][obs_idx]
        dt = dynamics["dt"][obs_idx]

        cands = candidates_per_obs[obs_idx]
        n_cands = len(cands)
        if n_cands == 0:
            continue

        min_perp_dist = min(c["perp_dist"] for c in cands)

        for rank, c in enumerate(cands):
            arc_id = int(c["arc_id"])
            u, v = int(arcs[arc_id, 0]), int(arcs[arc_id, 1])

            ax, ay = nodes_x[u], nodes_y[u]
            bx, by = nodes_y[v], nodes_y[v] # Note: bx, by
            bx = nodes_x[v]
            by = nodes_y[v]

            # Endpoint distances
            dist_to_start = float(np.hypot(px - ax, py - ay))
            dist_to_end = float(np.hypot(px - bx, py - by))
            dist_to_mid = float(np.hypot(px - 0.5 * (ax + bx), py - 0.5 * (ay + by)))

            # Angular difference between vehicle heading and road arc heading
            arc_h = float(c["arc_heading"])
            diff = abs(v_heading - arc_h)
            diff_rad = min(diff, 2.0 * np.pi - diff)
            cos_diff = float(np.cos(diff_rad))

            perp_d = float(c["perp_dist"])
            arc_len = float(c["arc_length"])

            row = {
                "trip_id": trip_id,
                "obs_idx": obs_idx,
                "arc_id": arc_id,
                "from_node": u,
                "to_node": v,
                "timestamp": float(timestamps[obs_idx]),
                # Spatial features
                "perp_dist": perp_d,
                "proj_frac": float(c["proj_frac"]),
                "dist_to_start": dist_to_start,
                "dist_to_end": dist_to_end,
                "dist_to_mid": dist_to_mid,
                "arc_length": arc_len,
                # Dynamic features
                "veh_speed": float(v_speed),
                "veh_accel": float(v_accel),
                "heading_diff_rad": float(diff_rad),
                "heading_cos": cos_diff,
                "speed_length_ratio": float(v_speed * dt / max(1.0, arc_len)),
                "time_delta": float(dt),
                # Topological features
                "node_out_degree": int(out_deg.get(u, 0)),
                "node_in_degree": int(in_deg.get(v, 0)),
                "is_bidirectional": int(is_bidir[arc_id]),
                # Relative candidate features
                "dist_rank": int(rank),
                "dist_diff_from_min": float(perp_d - min_perp_dist),
                "num_candidates": int(n_cands),
            }

            if "label" in c:
                row["label"] = int(c["label"])

            rows.append(row)

    return pd.DataFrame(rows)
