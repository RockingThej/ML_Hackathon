import nbformat as nbf
nb = nbf.v4.new_notebook()
C = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))
M = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))
M("# 01 EDA - PathSnap (Person A)\nRun from the repo root or notebooks/. Raw data in `data/raw/{train,student_test}`.")
C('''import sys, os
if os.path.basename(os.getcwd()) == "notebooks": os.chdir("..")
sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, folium
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.data.loaders import list_trips, load_trip
from src.features.candidates import prepare, point_segment, build_features
RAW = Path("data/raw")''')
M("## 1. Dataset inventory")
C('''rows = []
for part in ["train", "student_test"]:
    for tid in list_trips(RAW / part):
        d = RAW / part / f"{tid:08d}"; t = load_trip(RAW / part, tid)
        rows.append(dict(partition=part, trip_id=tid, n_points=len(t["track"]), n_nodes=len(t["nodes"]),
            n_arcs=len(t["arcs"]), route_len=len(t["route"]),
            size_mb=sum(f.stat().st_size for f in d.iterdir()) / 1e6,
            files=len(list(d.iterdir()))))
inv = pd.DataFrame(rows)
print(inv.groupby("partition").agg(trips=("trip_id", "count"), pts_min=("n_points", "min"), pts_med=("n_points", "median"),
      pts_max=("n_points", "max"), pts_total=("n_points", "sum"), MB=("size_mb", "sum")).round(1))
print("files per trip:", inv.files.unique())
inv.head()''')
M("## 2. Missing/invalid GPS values and timestamp gaps")
C('''qual = []
for part in ["train", "student_test"]:
    for tid in list_trips(RAW / part):
        tr = load_trip(RAW / part, tid)["track"]; dt = np.diff(tr[:, 2])
        qual.append(dict(partition=part, trip_id=tid, nan=int(np.isnan(tr).sum()),
            bad_lonlat=int(((abs(tr[:, 0]) > 180) | (abs(tr[:, 1]) > 90)).sum()),
            dup_or_back_ts=int((dt <= 0).sum()), pct_1s=(dt == 1).mean() * 100, max_gap_s=dt.max()))
q = pd.DataFrame(qual)
print("total NaN:", q.nan.sum(), "| invalid lon/lat:", q.bad_lonlat.sum(), "| non-increasing timestamps:", q.dup_or_back_ts.sum())
print("overall % of steps that are 1 s:", round(q.pct_1s.mean(), 1))
q.sort_values("pct_1s").head(8)''')
C('''plt.figure(figsize=(7, 3)); plt.hist(q.pct_1s, bins=30); plt.xlabel("% of steps exactly 1 s"); plt.ylabel("trips"); plt.show()
q.sort_values("max_gap_s", ascending=False).head(5)''')
M("## 3. Network size and connectivity (each trip has its OWN network)")
C('''net = []
for part in ["train", "student_test"]:
    for tid in list_trips(RAW / part):
        t = load_trip(RAW / part, tid); n = len(t["nodes"]); a = t["arcs"]
        g = coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(n, n))
        nc, lab = connected_components(g, directed=False)
        used = np.zeros(n, bool); used[a.ravel()] = True
        net.append(dict(partition=part, trip_id=tid, nodes=n, arcs=len(a), components=nc,
                        isolated_nodes=int((~used).sum()), largest_comp_pct=np.bincount(lab).max() / n * 100))
net = pd.DataFrame(net); print(net.describe().round(1)); net.head()''')
M("## 4. GPS trajectory plots (projected to per-trip UTM, metres)")
C('''fig, ax = plt.subplots(2, 3, figsize=(14, 8))
for a, tid in zip(ax.ravel(), list_trips(RAW / "train")[:6]):
    t = load_trip(RAW / "train", tid); xy, _, A, B, epsg = prepare(t)
    r = t["route"]
    for i in r: a.plot([A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]], "g-", lw=1)
    a.plot(xy[:, 0], xy[:, 1], ".", ms=1, c="b"); a.set_title(f"trip {tid} (EPSG:{epsg})"); a.set_aspect("equal")
plt.tight_layout(); plt.show()''')
M("## 5. Route vs track coverage (KEY finding)\nThe `.route` is a list of arcs, not a label per point, and it often covers only PART of the track.")
C('''cov = []
for tid in list_trips(RAW / "train"):
    t = load_trip(RAW / "train", tid); xy, _, A, B, _ = prepare(t); u = np.unique(t["route"]); d = np.empty(len(xy))
    for s in range(0, len(xy), 1000):
        dd, _ = point_segment(xy[s:s+1000, None, :], A[u][None], B[u][None]); d[s:s+1000] = dd.min(1)
    cov.append(dict(trip_id=tid, track_km=np.hypot(*np.diff(xy, axis=0).T).sum() / 1e3,
        route_km=np.hypot(*(B[t["route"]] - A[t["route"]]).T).sum() / 1e3, frac_within_50m=(d < 50).mean()))
cov = pd.DataFrame(cov); print(cov.describe().round(2)); cov.sort_values("frac_within_50m").head(6)''')
M("## 6. Folium map: GPS points + road network + reference route")
C('''from pyproj import Transformer
tid = list_trips(RAW / "train")[3]; t = load_trip(RAW / "train", tid)
c = t["track"][:, :2].mean(0)
m = folium.Map(location=[c[1], c[0]], zoom_start=14)
lo, hi = t["track"][:, :2].min(0) - .005, t["track"][:, :2].max(0) + .005
nd, ar = t["nodes"], t["arcs"]
inside = ((nd[ar[:, 0]] >= lo) & (nd[ar[:, 0]] <= hi)).all(1)
for a_ in ar[inside][:4000]:
    folium.PolyLine([nd[a_[0]][::-1], nd[a_[1]][::-1]], color="gray", weight=1).add_to(m)
for i in t["route"]:
    folium.PolyLine([nd[ar[i, 0]][::-1], nd[ar[i, 1]][::-1]], color="green", weight=4).add_to(m)
for p in t["track"][::5]:
    folium.CircleMarker([p[1], p[0]], radius=2, color="blue").add_to(m)
Path("reports").mkdir(exist_ok=True); m.save("reports/map_trip.html"); m''')
M("## 7. Candidate miss rate (from build_dataset.py)")
C('''mr = pd.read_csv("data/processed/candidate_miss_rate.csv")
print(mr.groupby("partition").apply(lambda g: pd.Series({"points": g.n_points.sum(), "labelled": g.n_labelled.sum(),
      "miss_rate": (g.miss_rate_labelled * g.n_labelled).sum() / g.n_labelled.sum()})))
mr.sort_values("miss_rate_labelled", ascending=False).head(5)''')
M('''## 8. Data-quality issues and assumptions
1. **Trips are worldwide**: ~23 UTM zones in train. Distances use a UTM zone chosen per trip from its mean longitude (EPSG 326xx/327xx), NOT one fixed EPSG.
2. **Each trip has its own road network**: arc_id/node_id are only valid within a trip.
3. **Route covers only part of the track in many trips** (e.g. trips 54 and 33). Points whose best route arc is >50 m away have no ground truth and are excluded from the labelled parquet (kept in `point_truth.parquet`).
4. **Time gaps**: most trips are ~97-100% 1 s steps but some are much lower (e.g. trip 4 ~54%); features use the real dt.
5. **Arcs are directed**; reverse arcs have identical geometry, so bearing_diff is what separates them.
6. **Labelling assumption**: points are assigned to route arcs monotonically in route order (DP minimising total distance).
7. Stationary points (<1 m move) have no reliable heading: `bearing_diff`=90 and `heading_valid`=0.''')
nbf.write(nb, "notebooks/01_eda.ipynb")
