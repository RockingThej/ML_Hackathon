"""
candidate_generation.py - Fast and mathematically complete candidate road segment generation
using metric spatial indexing (KDTree on densely sampled segment points).
"""

from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from scipy.spatial import KDTree


def point_to_segment_distance(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float
) -> Tuple[float, float]:
    """
    Compute Euclidean distance from point (px, py) to line segment (ax, ay) -> (bx, by),
    along with the normalized projection fraction t in [0, 1].
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


class SpatialRoadIndex:
    """
    Spatial index over road network arcs in metric space.
    Samples points along segments to guarantee that segments passing near
    a GPS observation are retrieved even if their endpoints or midpoint are far away.
    """

    def __init__(self, nodes_x: np.ndarray, nodes_y: np.ndarray, arcs: np.ndarray, step_meters: float = 12.0):
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

    def query_candidates(
        self,
        px: float,
        py: float,
        radius_m: float = 50.0,
        min_candidates: int = 3,
        max_candidates: Optional[int] = 20
    ) -> List[Dict[str, float]]:
        """
        Query road segment candidates for a single GPS point (px, py).
        Returns list of dicts with:
        ['arc_id', 'perp_dist', 'proj_frac', 'arc_length', 'arc_heading']
        sorted by perpendicular distance ascending.
        """
        idx_in_ball = self.kdtree.query_ball_point([px, py], r=radius_m)
        candidate_arc_ids = set(self.sample_arc_ids[idx_in_ball])

        # Fallback to nearest neighbors if fewer than min_candidates
        if len(candidate_arc_ids) < min_candidates:
            _, nearest_pts = self.kdtree.query([px, py], k=min(len(self.sample_points), min_candidates * 4))
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
            dist, t = point_to_segment_distance(px, py, ax, ay, bx, by)

            candidates.append({
                "arc_id": arc_id,
                "from_node": u,
                "to_node": v,
                "perp_dist": dist,
                "proj_frac": t,
                "arc_length": float(self.lengths[arc_id]),
                "arc_heading": float(self.headings[arc_id]),
            })

        # Sort by perpendicular distance
        candidates.sort(key=lambda c: c["perp_dist"])

        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        return candidates

    def batch_query_candidates(
        self,
        track_x: np.ndarray,
        track_y: np.ndarray,
        radius_m: float = 50.0,
        min_candidates: int = 3,
        max_candidates: Optional[int] = 20
    ) -> List[List[Dict[str, float]]]:
        """Query candidates for an entire trajectory batch."""
        all_candidates = []
        for px, py in zip(track_x, track_y):
            cands = self.query_candidates(
                px, py,
                radius_m=radius_m,
                min_candidates=min_candidates,
                max_candidates=max_candidates
            )
            all_candidates.append(cands)
        return all_candidates
