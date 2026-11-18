"""
api/main.py  –  PathSnap FastAPI Backend (Laptop 1)
===================================================
Endpoints:
  GET  /health                  – Health check & model status
  GET  /trips                   – List evaluation trips (and training trips)
  POST /predict                 – Run end-to-end map-matching pipeline for a trip
  GET  /trips/{trip_id}/track   – Get raw GPS points for visualization
  GET  /trips/{trip_id}/network – Get road network nodes and arcs for Folium map

Architecture:
  - Model & API run on Laptop 1 (evaluator UI runs on Laptop 2)
  - Ground truth labels NEVER exposed to client; evaluation computed server-side
  - Uses A's candidate generator (src.features.candidates.build_features)
  - Uses trained classifier (LGBM / Random Forest) from artifacts/
  - Structured request logging with millisecond latencies
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure repo root is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import load_trip, list_trips
from src.features.candidates import build_features, FEATURES as FEATURE_COLS

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pathsnap.api")

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PathSnap API",
    description="Distributed Map-Matching & Route Reconstruction API (Laptop 1)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "%s %s -> status=%d (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response

# ---------------------------------------------------------------------------
# Paths and model cache
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_TRAIN = DATA_DIR / "train"
# Which test folder to serve. Default student_test; for the evaluators run with
#   PowerShell:  $env:PATHSNAP_TEST_SET = "evaluator_test"
TEST_SET = os.environ.get("PATHSNAP_TEST_SET", "student_test")
DATA_TEST = DATA_DIR / TEST_SET
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

_loaded_model: Any = None
_model_name: str = "none"


def get_model():
    """Load and cache the best available trained model."""
    global _loaded_model, _model_name
    if _loaded_model is not None:
        return _loaded_model, _model_name

    # Priority order: best_model (tuned) -> xgb -> mlp -> lgbm -> rf -> logreg
    for candidate in [
        "best_model.joblib",
        "xgb.joblib",
        "mlp_listwise.joblib",
        "lgbm.joblib",
        "rf.joblib",
        "logreg.joblib",
    ]:
        p = ARTIFACTS_DIR / candidate
        if p.exists():
            try:
                _loaded_model = joblib.load(p)
                _model_name = candidate.replace(".joblib", "")
                logger.info("Successfully loaded model: %s from %s", _model_name, p)
                return _loaded_model, _model_name
            except Exception as e:
                logger.warning("Could not load %s: %s", p, e)

    logger.warning("No pre-trained model found in %s; falling back to distance heuristic.", ARTIFACTS_DIR)
    _model_name = "distance_heuristic"
    return None, _model_name


@app.on_event("startup")
def on_startup():
    get_model()
    logger.info("PathSnap API initialization complete.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ArcRequest(BaseModel):
    arc_ids: List[int]


class PredictRequest(BaseModel):
    trip_id: int = Field(..., description="Trip ID (e.g. 0, 7, 9, 40)")
    top_k: int = Field(15, description="Number of candidates to generate per GPS point")
    use_hmm: bool = Field(False, description="Whether to also run HMM Viterbi decoding")


class MetricSummary(BaseModel):
    point_accuracy: float = Field(..., description="Observation-level top-1 accuracy")
    candidate_miss_rate: float = Field(..., description="Fraction of points where true arc missed")
    n_evaluated_points: int = Field(..., description="Number of GPS points evaluated against truth")


class PredictResponse(BaseModel):
    trip_id: int
    n_points: int
    model_name: str
    predicted_route: List[int] = Field(..., description="Reconstructed sequence of arc_ids")
    hmm_route: Optional[List[int]] = Field(None, description="HMM Viterbi decoded sequence of arc_ids")
    metrics: Optional[MetricSummary] = Field(None, description="Server-side evaluation if ground truth is present")
    hmm_metrics: Optional[MetricSummary] = Field(None, description="HMM server-side evaluation if ground truth is present")
    inference_time_ms: float


class TripListResponse(BaseModel):
    test_set: str = Field(..., description="Name of the test folder being served")
    test_trips: List[int]
    train_trips: List[int]
    student_test_trips: List[int] = Field(default_factory=list, description="Same as test_trips (kept for older UIs)")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    _, model_name = get_model()
    return {
        "status": "healthy",
        "model_loaded": model_name,
        "train_trips_available": len(list_trips(DATA_TRAIN)) if DATA_TRAIN.exists() else 0,
        "test_set": TEST_SET,
        "test_trips_available": len(list_trips(DATA_TEST)) if DATA_TEST.exists() else 0,
    }


@app.get("/trips", response_model=TripListResponse)
def get_trips():
    """List available test-set trips (folder from PATHSNAP_TEST_SET) and training trips."""
    train = list_trips(DATA_TRAIN) if DATA_TRAIN.exists() else []
    test = list_trips(DATA_TEST) if DATA_TEST.exists() else []
    return TripListResponse(test_set=TEST_SET, test_trips=test, train_trips=train,
                             student_test_trips=test)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Predict road segments (arc_ids) for each GPS observation in a trip.
    Ground truth is kept strictly on the server and used only to report evaluation metrics.
    """
    trip_id = req.trip_id
    t0 = time.perf_counter()

    # Determine data directory
    trip_root = None
    if DATA_TEST.exists() and (DATA_TEST / f"{trip_id:08d}").exists():
        trip_root = DATA_TEST
    elif DATA_TRAIN.exists() and (DATA_TRAIN / f"{trip_id:08d}").exists():
        trip_root = DATA_TRAIN
    else:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found in train or {TEST_SET} datasets.")

    try:
        trip = load_trip(trip_root, trip_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load trip files: {e}")

    # Generate candidate features using A's candidate generator
    try:
        # Generate with labels if route is available for server-side evaluation
        has_route = trip.get("route") is not None
        df_feats = build_features(trip, K=req.top_k, with_labels=has_route)
    except Exception as e:
        logger.exception("Feature generation failed for trip %d", trip_id)
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")

    model, model_name = get_model()

    # Score candidates
    if model is not None and hasattr(model, "predict_proba"):
        X = df_feats[FEATURE_COLS].to_numpy(dtype=np.float32)
        # Probability of being the true arc (class 1)
        scores = model.predict_proba(X)[:, 1]
    elif model is not None and hasattr(model, "decision_function"):
        X = df_feats[FEATURE_COLS].to_numpy(dtype=np.float32)
        scores = model.decision_function(X)
    else:
        # Heuristic fallback: lower distance to arc gets higher score
        scores = -df_feats["dist_to_arc"].to_numpy()

    df_feats["score"] = scores

    # Select top candidate per point
    best_candidates = (
        df_feats.sort_values("score", ascending=False)
        .groupby("point_idx", sort=True)
        .first()
        .reset_index()
    )

    predicted_route = best_candidates["candidate_arc_id"].tolist()
    n_points = len(trip["track"])

    # Compute server-side evaluation if ground truth exists
    metric_summary = None
    if has_route and "label" in df_feats.columns:
        # Only evaluate on points with usable ground truth
        evaluated = best_candidates[best_candidates.get("has_truth", 1) == 1]
        if len(evaluated) > 0:
            acc = float(evaluated["label"].mean())
            miss = 1.0 - acc
            metric_summary = MetricSummary(
                point_accuracy=round(acc, 4),
                candidate_miss_rate=round(miss, 4),
                n_evaluated_points=len(evaluated),
            )

    # HMM route decoding if requested
    hmm_route = None
    hmm_metric_summary = None
    if req.use_hmm:
        try:
            from src.hmm.matcher import match_trip
            result_df, _, _ = match_trip(str(trip_root), trip_id, K=req.top_k, model=model)
            hmm_route = result_df["predicted_arc_id"].tolist()
            if has_route and "true_arc" in df_feats.attrs:
                true_arcs = df_feats.attrs["true_arc"]
                true_dists = df_feats.attrs.get("true_dist", np.zeros(len(true_arcs)))
                valid = (true_dists <= 50.0) & (np.arange(len(true_arcs)) < len(hmm_route))
                if valid.sum() > 0:
                    hmm_arr = np.array(hmm_route)
                    correct = (hmm_arr[valid] == true_arcs[valid]).sum()
                    h_acc = float(correct / valid.sum())
                    hmm_metric_summary = MetricSummary(
                        point_accuracy=round(h_acc, 4),
                        candidate_miss_rate=round(1.0 - h_acc, 4),
                        n_evaluated_points=int(valid.sum()),
                    )
        except Exception as e:
            logger.warning("HMM decoding failed for trip %d: %s", trip_id, e)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PredictResponse(
        trip_id=trip_id,
        n_points=n_points,
        model_name=model_name,
        predicted_route=predicted_route,
        hmm_route=hmm_route,
        metrics=metric_summary,
        hmm_metrics=hmm_metric_summary,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/trips/{trip_id}/arcs")
def get_arcs(trip_id: int, req: ArcRequest):
    """Lat/lon endpoints of the requested arcs (for drawing the matched route on Laptop 2)."""
    root = DATA_TEST if (DATA_TEST / f"{trip_id:08d}").exists() else DATA_TRAIN
    if not (root / f"{trip_id:08d}").exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    trip = load_trip(root, trip_id)
    nodes, arcs = trip["nodes"], trip["arcs"]  # nodes: lon, lat
    out = []
    for a in req.arc_ids:
        if 0 <= a < len(arcs):
            u, v = arcs[a]
            out.append({
                "arc_id": int(a),
                "coords": [
                    [float(nodes[u, 1]), float(nodes[u, 0])],
                    [float(nodes[v, 1]), float(nodes[v, 0])],
                ],
            })
    return {"trip_id": trip_id, "arcs": out}


@app.get("/trips/{trip_id}/track")
def get_track(trip_id: int):
    """Return raw GPS trajectory points for the map view."""
    root = DATA_TEST if (DATA_TEST / f"{trip_id:08d}").exists() else DATA_TRAIN
    if not (root / f"{trip_id:08d}").exists():
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    trip = load_trip(root, trip_id)
    track = trip["track"]  # lon, lat, timestamp
    return {
        "trip_id": trip_id,
        "points": [
            {"lon": float(pt[0]), "lat": float(pt[1]), "timestamp": float(pt[2])}
            for pt in track
        ],
    }
