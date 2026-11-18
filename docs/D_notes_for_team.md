# Part D: notes for the team

## For C (API)

### 1. Load D's models
`get_model()` in `api/main.py` currently tries `lgbm → rf → logreg → best_model`.
Once the tuned model exists, put `best_model.joblib` **first** so the API serves it:

```python
for candidate in ["best_model.joblib", "lgbm.joblib", "rf.joblib", "logreg.joblib"]:
```

Every D model has a sklearn-style `predict_proba`, so nothing else in `/predict` needs to change:

| File | Type | Notes |
|---|---|---|
| `artifacts/best_model.joblib` | whichever model Optuna tuned | saved with `device="cpu"`, so it runs on a laptop without a GPU |
| `artifacts/xgb.joblib` | `xgboost.XGBClassifier` | needs `xgboost` installed on Laptop 1 |
| `artifacts/mlp_listwise.joblib` | `src.models.mlp.MLPScorer` | needs `torch`; the scaler is built in, so pass **raw** features |

### 2. Arc geometry endpoint (the UI needs it to draw the matched route)
`/predict` returns arc IDs only, and Laptop 2 has no raw data. Please add this endpoint. It returns only the requested arcs and never touches the `.route` file, so no ground truth is exposed:

```python
class ArcRequest(BaseModel):
    arc_ids: List[int]


@app.post("/trips/{trip_id}/arcs")
def get_arcs(trip_id: int, req: ArcRequest):
    """Lat/lon endpoints of the requested arcs (for drawing the matched route)."""
    root = DATA_TEST if (DATA_TEST / f"{trip_id:08d}").exists() else DATA_TRAIN
    if not (root / f"{trip_id:08d}").exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    trip = load_trip(root, trip_id)
    nodes, arcs = trip["nodes"], trip["arcs"]  # nodes: lon, lat
    out = []
    for a in req.arc_ids:
        if 0 <= a < len(arcs):
            u, v = arcs[a]
            out.append({"arc_id": int(a),
                        "coords": [[float(nodes[u, 1]), float(nodes[u, 0])],
                                   [float(nodes[v, 1]), float(nodes[v, 0])]]})
    return {"trip_id": trip_id, "arcs": out}
```

Until this endpoint exists, the UI falls back to local `data/raw/` if it is present, and otherwise draws only the GPS trace.

### 3. `make_dummy_features.py` does not match A's schema
The dummy generator writes `arc_id`, `speed_mps`, `time_delta` and similar columns, but A's real parquet has `candidate_arc_id`, `dist_rank`, `speed`, `dt_prev`, `label`. D's scripts were therefore tested on a **5-trip subset of the real data** (`--smoke`), not on dummy data.

## For B (HMM)

D writes per-candidate scores that you can use as **emission probabilities**:

`data/processed/scores_<model>_test.parquet` (git-ignored; shared over Drive)

| column | meaning |
|---|---|
| `trip_id`, `point_idx`, `candidate_arc_id` | join keys, same as the feature files |
| `prob` | P(this candidate is the true arc), independent per row |
| `point_prob` | softmax across the point's 15 candidates (**sums to 1 per point**); use `log(point_prob)` as the emission log-probability |

To score a new trip inside the API or the HMM (the rows must be in `build_features` order):
```python
import joblib
from src.features.candidates import build_features, FEATURES
m = joblib.load("artifacts/mlp_listwise.joblib")
df = build_features(trip, K=15, with_labels=False)
df["point_prob"] = m.point_proba(df[FEATURES].to_numpy(), k=15)   # MLP only
```

## Metric definition used by D
D uses C's definition: **miss-rate** is the fraction of labelled GPS points whose top-scored candidate is not the true arc. If the true arc wasn't among the 15 candidates, that point counts as a miss. This keeps D's rows in `results/model_comparison.csv` directly comparable with C's.
