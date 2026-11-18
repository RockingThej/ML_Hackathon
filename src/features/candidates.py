"""
src/features/candidates.py - Candidate generation and feature extraction per trip.

This module exposes the main contract function `build_trip_features(trip_data, ...)`
required by:
1. Feature pipeline (features_train.parquet, features_test.parquet)
2. Person B (HMM and Viterbi decoder)
3. Person C / API server (FastAPI inference on evaluation trips)
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from src.data.projection import project_coordinates
from src.data.loaders import TripData


def point_to_segment_metrics(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float
) -> Tuple[float, float]:
    """
    Compute perpendicular distance from (px, py) to segment (ax, ay) -> (bx, by)
    and projection fraction t in [0, 1].
    """
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0.0:
        return float(np.hypot(px - ax, py - ay)), 0.0
    
    t_unclamped = ((px - ax) * dx + (py - ay) * dy) / l2
    t = max(0.0, min(1.0, t_unclamped))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = float(np.hypot(px - proj_x, py - proj_y))
    return dist, float(t)


class FastSpatialRoadIndex:
    """
    Spatial index over road network arcs in metric space (EPSG:32632).
    Densely samples points along segments so arcs passing near GPS observations
    are reliably identified without missing long road segments.
    """

    def __init__(
        self,
        nodes_x: np.ndarray,
        nodes_y: np.ndarray,
        arcs: np.ndarray,
        step_meters: float = 12.0
    ):
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.arcs = arcs
        self.num_arcs = len(arcs)

        ax = nodes_x[arcs[:, 0]]
        ay = nodes_y[arcs[:, 0]]
        bx = nodes_x[arcs[:, 1]]
        by = nodes_y[arcs[:, 1]]

        dx = bx - ax
        dy = by - ay
        self.lengths = np.hypot(dx, dy)
        self.headings = np.arctan2(dy, dx)

        # Dense point sampling along each segment
        pts_list = []
        arc_id_list = []
        for i in range(self.num_arcs):
            L = self.lengths[i]
            if L == 0.0:
                pts_list.append((ax[i], ay[i]))
                arc_id_list.append(i)
            else:
                n_samples = max(2, int(np.ceil(L / step_meters)) + 1)
                fractions = np.linspace(0.0, 1.0, n_samples)
                for frac in fractions:
                    pts_list.append((ax[i] + frac * dx[i], ay[i] + frac * dy[i]))
                    arc_id_list.append(i)

        self.sample_points = np.array(pts_list, dtype=np.float64)
        self.sample_arc_ids = np.array(arc_id_list, dtype=np.int64)
        self.kdtree = KDTree(self.sample_points)

    def query(
        self,
        px: float,
        py: float,
        radius_m: float = 50.0,
        min_candidates: int = 3,
        max_candidates: int = 15
    ) -> List[Dict[str, float]]:
        """
        Query candidate arcs within radius_m of (px, py).
        Guarantees at least min_candidates and at most max_candidates.
        """
        idx_in_ball = self.kdtree.query_ball_point([px, py], r=radius_m)
        candidate_arc_ids = set(self.sample_arc_ids[idx_in_ball])

        # Fallback to nearest neighbors if under min_candidates
        if len(candidate_arc_ids) < min_candidates:
            _, nearest_pts = self.kdtree.query(
                [px, py],
                k=min(len(self.sample_points), min_candidates * 4)
            )
            if not isinstance(nearest_pts, (list, np.ndarray)):
                nearest_pts = [nearest_pts]
            for pt_idx in nearest_pts:
                candidate_arc_ids.add(int(self.sample_arc_ids[pt_idx]))
                if len(candidate_arc_ids) >= min_candidates:
                    break

        candidates = []
        for arc_id in candidate_arc_ids:
            u, v = self.arcs[arc_id]
            ax, ay = self.nodes_x[u], self.nodes_y[u]
            bx, by = self.nodes_x[v], self.nodes_y[v]
            dist, t = point_to_segment_metrics(px, py, ax, ay, bx, by)

            candidates.append({
                "arc_id": arc_id,
                "from_node": u,
                "to_node": v,
                "perp_dist": dist,
                "proj_frac": t,
                "dist_to_start": float(np.hypot(px - ax, py - ay)),
                "dist_to_end": float(np.hypot(px - bx, py - by)),
                "arc_length": float(self.lengths[arc_id]),
                "arc_heading": float(self.headings[arc_id]),
            })

        # Sort by perpendicular distance ascending
        candidates.sort(key=lambda c: c["perp_dist"])

        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        return candidates


def label_ground_truth(
    track_x: np.ndarray,
    track_y: np.ndarray,
    nodes_x: np.ndarray,
    nodes_y: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    candidates_per_point: List[List[Dict[str, Any]]],
    search_window: int = 15
) -> Tuple[np.ndarray, float]:
    """
    Assign ground-truth arc for each GPS observation using monotonic progression
    along the reference route, and compute the candidate miss rate.
    
    Returns:
    --------
    true_arc_ids: 1D array of length N containing true arc_id for each point.
    miss_rate: percentage of points where true arc is missing from candidate set.
    """
    N = len(track_x)
    num_route = len(route)
    if num_route == 0:
        return np.full(N, -1, dtype=np.int64), 0.0

    u_list = arcs[route, 0]
    v_list = arcs[route, 1]
    ax = nodes_x[u_list]
    ay = nodes_y[u_list]
    bx = nodes_x[v_list]
    by = nodes_y[v_list]

    curr_route_idx = 0
    true_arc_ids = np.zeros(N, dtype=np.int64)

    for i in range(N):
        px, py = track_x[i], track_y[i]
        start_j = max(0, curr_route_idx - 2)
        end_j = min(num_route, curr_route_idx + search_window)

        best_dist = float("inf")
        best_j = curr_route_idx

        for j in range(start_j, end_j):
            dist, _ = point_to_segment_metrics(px, py, ax[j], ay[j], bx[j], by[j])
            if dist < best_dist:
                best_dist = dist
                best_j = j

        curr_route_idx = max(curr_route_idx, best_j)
        true_arc_ids[i] = route[best_j]

    # Calculate candidate miss rate
    misses = 0
    for i in range(N):
        c_arcs = {c["arc_id"] for c in candidates_per_point[i]}
        if true_arc_ids[i] not in c_arcs:
            misses += 1

    miss_rate = (misses / N) * 100.0 if N > 0 else 0.0
    return true_arc_ids, miss_rate


def build_trip_features(
    trip_data: TripData,
    radius_m: float = 50.0,
    min_candidates: int = 3,
    max_candidates: int = 15,
    road_index: Optional[FastSpatialRoadIndex] = None
) -> Tuple[pd.DataFrame, float]:
    """
    Main contract function: takes a trip's raw data and returns the feature DataFrame
    and the candidate miss rate.
    
    Columns returned:
    ['trip_id', 'point_idx', 'candidate_arc_id', 'perp_dist', 'proj_frac',
     'dist_to_start', 'dist_to_end', 'arc_length', 'veh_speed', 'veh_accel',
     'heading_diff_rad', 'heading_cos', 'dist_rank', 'dist_diff_from_min',
     'time_delta', 'num_candidates', 'node_out_degree', 'node_in_degree',
     'is_bidirectional', 'label']
    
    All numeric features are cast to float32 for downstream ML models.
    """
    track = trip_data.track
    nodes = trip_data.nodes
    arcs = trip_data.arcs
    route = trip_data.route
    trip_id = trip_data.trip_id

    # 1. Project coordinates to EPSG:32632 (meters)
    track_x, track_y = project_coordinates(track["lon"].values, track["lat"].values)
    nodes_x, nodes_y = project_coordinates(nodes[:, 0], nodes[:, 1])

    # 2. Build spatial road index if not provided
    if road_index is None:
        road_index = FastSpatialRoadIndex(nodes_x, nodes_y, arcs, step_meters=12.0)

    # 3. Compute trajectory dynamics (speed, acceleration, heading)
    N = len(track_x)
    timestamps = track["timestamp"].values
    dt = np.ones(N, dtype=np.float64)
    dt[1:] = np.maximum(0.1, timestamps[1:] - timestamps[:-1])
    dt[0] = dt[1] if N > 1 else 1.0

    dx = np.zeros(N, dtype=np.float64)
    dy = np.zeros(N, dtype=np.float64)
    if N > 1:
        dx[1:] = track_x[1:] - track_x[:-1]
        dy[1:] = track_y[1:] - track_y[:-1]
        dx[0] = dx[1]
        dy[0] = dy[1]

    step_dist = np.hypot(dx, dy)
    speed = step_dist / dt

    accel = np.zeros(N, dtype=np.float64)
    if N > 1:
        accel[1:] = (speed[1:] - speed[:-1]) / dt[1:]
        accel[0] = accel[1]

    veh_heading = np.arctan2(dy, dx)

    # 4. Road network topology
    out_degrees = {}
    in_degrees = {}
    arc_pairs = set()
    for u, v in arcs:
        u_i, v_i = int(u), int(v)
        out_degrees[u_i] = out_degrees.get(u_i, 0) + 1
        in_degrees[v_i] = in_degrees.get(v_i, 0) + 1
        arc_pairs.add((u_i, v_i))

    # 5. Candidate generation for each point
    candidates_per_point = []
    for i in range(N):
        cands = road_index.query(
            track_x[i], track_y[i],
            radius_m=radius_m,
            min_candidates=min_candidates,
            max_candidates=max_candidates
        )
        candidates_per_point.append(cands)

    # 6. Ground-truth labeling (if route available)
    if route is not None and len(route) > 0:
        true_arcs, miss_rate = label_ground_truth(
            track_x, track_y, nodes_x, nodes_y, arcs, route,
            candidates_per_point=candidates_per_point
        )
    else:
        true_arcs = np.full(N, -1, dtype=np.int64)
        miss_rate = 0.0

    # 7. Build candidate feature rows
    rows = []
    for point_idx in range(N):
        px = track_x[point_idx]
        py = track_y[point_idx]
        v_spd = speed[point_idx]
        v_acc = accel[point_idx]
        v_hdg = veh_heading[point_idx]
        t_gap = dt[point_idx]
        t_arc = true_arcs[point_idx]

        cands = candidates_per_point[point_idx]
        n_cands = len(cands)
        if n_cands == 0:
            continue

        min_perp_d = cands[0]["perp_dist"]

        for rank, c in enumerate(cands):
            arc_id = int(c["arc_id"])
            u = int(c["from_node"])
            v = int(c["to_node"])

            # Heading difference
            arc_h = c["arc_heading"]
            diff = abs(v_hdg - arc_h)
            diff_rad = min(diff, 2.0 * np.pi - diff)
            cos_diff = np.cos(diff_rad)

            # Reversibility
            is_bidir = 1.0 if (v, u) in arc_pairs else 0.0

            perp_d = c["perp_dist"]
            label_val = 1 if (arc_id == t_arc) else (0 if t_arc != -1 else -1)

            rows.append({
                "trip_id": int(trip_id),
                "point_idx": int(point_idx),
                "candidate_arc_id": int(arc_id),
                # Features
                "perp_dist": np.float32(perp_d),
                "proj_frac": np.float32(c["proj_frac"]),
                "dist_to_start": np.float32(c["dist_to_start"]),
                "dist_to_end": np.float32(c["dist_to_end"]),
                "arc_length": np.float32(c["arc_length"]),
                "veh_speed": np.float32(v_spd),
                "veh_accel": np.float32(v_acc),
                "heading_diff_rad": np.float32(diff_rad),
                "heading_cos": np.float32(cos_diff),
                "dist_rank": np.float32(rank),
                "dist_diff_from_min": np.float32(perp_d - min_perp_d),
                "time_delta": np.float32(t_gap),
                "num_candidates": np.float32(n_cands),
                "node_out_degree": np.float32(out_degrees.get(u, 0)),
                "node_in_degree": np.float32(in_degrees.get(v, 0)),
                "is_bidirectional": np.float32(is_bidir),
                # Label
                "label": int(label_val),
            })

    df = pd.DataFrame(rows)
    return df, miss_rate
