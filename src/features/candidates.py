import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from src.data.projection import make_transformer, project
FEATURES = ['dist_to_arc', 'dist_rank', 'bearing_diff', 'heading_valid', 'speed', 'arc_length', 'proj_pos', 'dt_prev']

def prepare(trip):
    track, nodes, arcs = (trip['track'], trip['nodes'], trip['arcs'])
    tr, epsg = make_transformer(track[:, 0].mean(), track[:, 1].mean())
    xy = project(tr, track[:, :2])
    node_xy = project(tr, nodes)
    A, B = (node_xy[arcs[:, 0]], node_xy[arcs[:, 1]])
    return (xy, track[:, 2], A, B, epsg)

def point_segment(P, A, B):
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(-1), 1e-12)
    t = np.clip(((P - A) * AB).sum(-1) / L2, 0.0, 1.0)
    proj = A + t[..., None] * AB
    return (np.sqrt(((P - proj) ** 2).sum(-1)), t)

def headings(xy):
    n = len(xy)
    d = np.zeros_like(xy)
    d[1:] = xy[1:] - xy[:-1]
    if n > 1:
        d[0] = xy[1] - xy[0]
    ok = np.hypot(d[:, 0], d[:, 1]) > 1.0
    return (np.arctan2(d[:, 1], d[:, 0]), ok)

def knn_candidates(xy, A, B, K=15, pre=45):
    mid = (A + B) / 2.0
    pre = min(max(pre, K), len(mid))
    _, idx = cKDTree(mid).query(xy, k=pre)
    idx = idx.reshape(len(xy), -1)
    d, t = point_segment(xy[:, None, :], A[idx], B[idx])
    order = np.argsort(d, axis=1)[:, :K]
    r = np.arange(len(xy))[:, None]
    return (idx[r, order], d[r, order], t[r, order])

def label_points(xy, A, B, route):
    n, m = (len(xy), len(route))
    cost = np.empty((n, m), dtype=np.float32)
    uniq, inv = np.unique(route, return_inverse=True)
    for s in range(0, n, 2000):
        sl = slice(s, s + 2000)
        d, _ = point_segment(xy[sl, None, :], A[uniq][None], B[uniq][None])
        cost[sl] = d[:, inv]
    back = np.empty((n, m), dtype=np.int32)
    dp = cost[0].copy()
    ar = np.arange(m)
    for i in range(1, n):
        pm = np.minimum.accumulate(dp)
        change = np.concatenate([[True], dp[1:] < pm[:-1]])
        back[i] = np.maximum.accumulate(np.where(change, ar, 0))
        dp = cost[i] + pm
    j = np.empty(n, dtype=np.int64)
    j[-1] = int(np.argmin(dp))
    for i in range(n - 1, 0, -1):
        j[i - 1] = back[i, j[i]]
    return (route[j], cost[np.arange(n), j].astype(np.float64))

def build_features(trip, K=15, with_labels=True, max_label_dist=50.0):
    xy, t, A, B, _ = prepare(trip)
    n = len(xy)
    cand, dist, proj = knn_candidates(xy, A, B, K=K)
    Kc = cand.shape[1]
    dt_prev = np.zeros(n)
    dt_prev[1:] = np.diff(t)
    step = np.zeros(n)
    step[1:] = np.hypot(*(xy[1:] - xy[:-1]).T)
    dt_sp = dt_prev.copy()
    if n > 1:
        step[0] = step[1]
        dt_sp[0] = dt_prev[1]
    speed = step / np.maximum(dt_sp, 0.001)
    hd, ok = headings(xy)
    arc_vec = B[cand] - A[cand]
    arc_len = np.hypot(arc_vec[..., 0], arc_vec[..., 1])
    arc_dir = np.arctan2(arc_vec[..., 1], arc_vec[..., 0])
    diff = np.abs((hd[:, None] - arc_dir + np.pi) % (2 * np.pi) - np.pi)
    bdiff = np.where(ok[:, None], np.degrees(diff), 90.0)
    f32 = np.float32
    df = pd.DataFrame({'trip_id': np.full(n * Kc, trip['trip_id'], dtype=np.int32), 'point_idx': np.repeat(np.arange(n, dtype=np.int32), Kc), 'candidate_arc_id': cand.ravel().astype(np.int32), 'dist_to_arc': dist.ravel().astype(f32), 'dist_rank': np.tile(np.arange(Kc), n).astype(f32), 'bearing_diff': bdiff.ravel().astype(f32), 'heading_valid': np.repeat(ok, Kc).astype(f32), 'speed': np.repeat(speed, Kc).astype(f32), 'arc_length': arc_len.ravel().astype(f32), 'proj_pos': proj.ravel().astype(f32), 'dt_prev': np.repeat(dt_prev, Kc).astype(f32)})
    if with_labels:
        if trip.get('route') is None:
            raise ValueError('with_labels=True needs a .route file')
        true_arc, true_dist = label_points(xy, A, B, trip['route'])
        df['label'] = (df['candidate_arc_id'].to_numpy() == np.repeat(true_arc, Kc)).astype(np.int8)
        df['has_truth'] = np.repeat(true_dist <= max_label_dist, Kc).astype(np.int8)
        df.attrs['true_arc'] = true_arc
        df.attrs['true_dist'] = true_dist
    return df
