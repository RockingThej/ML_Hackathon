"""
labeling.py - Ground-truth labeling of GPS observations against reference routes.

Assigns each GPS observation in a trajectory to its corresponding ground-truth arc
from the chronological .route file using monotonic dynamic programming projection,
producing binary classification labels (y in {0, 1}) for candidate sets.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from src.candidate_generation import point_to_segment_distance


def label_trajectory_against_route(
    track_x: np.ndarray,
    track_y: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    search_window: int = 15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map each GPS observation to an arc in the ground-truth route sequence.
    
    Parameters:
    -----------
    track_x, track_y: 1D arrays of trajectory coordinates in meters.
    nodes_x, nodes_y: 1D arrays of network node coordinates in meters.
    arcs: 2D array [from_node, to_node] of all network arcs.
    route: 1D array of arc IDs in chronological traversal order.
    search_window: Forward search window in route indices to enforce forward progression.
    
    Returns:
    --------
    assigned_true_arcs: 1D array of true arc_ids for each GPS observation.
    assigned_distances: 1D array of perpendicular distances to the assigned true arc.
    """
    num_pts = len(track_x)
    num_route = len(route)
    if num_route == 0:
        return np.full(num_pts, -1, dtype=np.int64), np.full(num_pts, np.nan)

    # Route segment coordinates
    u_list = arcs[route, 0]
    v_list = arcs[route, 1]
    ax = nodes_x[u_list]
    ay = nodes_y[u_list]
    bx = nodes_x[v_list]
    by = nodes_y[v_list]

    curr_route_idx = 0
    assigned_true_arcs = np.zeros(num_pts, dtype=np.int64)
    assigned_distances = np.zeros(num_pts, dtype=np.float64)

    for i in range(num_pts):
        px, py = track_x[i], track_y[i]
        start_j = max(0, curr_route_idx - 2)
        end_j = min(num_route, curr_route_idx + search_window)

        best_dist = float("inf")
        best_j = curr_route_idx

        for j in range(start_j, end_j):
            dist, _ = point_to_segment_distance(px, py, ax[j], ay[j], bx[j], by[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j

        # Monotonically advance route pointer
        curr_route_idx = max(curr_route_idx, best_j)
        assigned_true_arcs[i] = route[best_j]
        assigned_distances[i] = best_dist

    return assigned_true_arcs, assigned_distances


def label_candidates(
    candidates_per_obs: List[List[Dict[str, float]]],
    true_arc_per_obs: np.ndarray
) -> List[List[Dict[str, float]]]:
    """
    Annotate candidate dictionaries with binary label 'label' in {0, 1}.
    """
    labeled_candidates = []
    for i, cands in enumerate(candidates_per_obs):
        true_arc = true_arc_per_obs[i] if i < len(true_arc_per_obs) else -1
        obs_cands = []
        for c in cands:
            c_copy = dict(c)
            c_copy["label"] = 1 if (c["arc_id"] == true_arc) else 0
            obs_cands.append(c_copy)
        labeled_candidates.append(obs_cands)
    return labeled_candidates
