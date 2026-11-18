"""
api/main.py  –  PathSnap FastAPI backend (C's deliverable)
===========================================================
Endpoints
---------
  GET  /trips                   – list available trip IDs for the evaluator
  POST /predict                 – run the full prediction pipeline for a trip
  GET  /health                  – quick liveness check

Run locally:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Or from project root:
    python -m uvicorn api.main:app --reload
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("pathsnap.api")

# ---------------------------------------------------------------------------
# App init
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PathSnap API",
    description="Map-matching & route reconstruction API for the PathSnap hackathon.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten before production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    log.info(
        "%s %s  status=%d  %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )
    return response

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
DATA_TRAIN   = PROJECT_ROOT / "data" / "raw" / "train"
DATA_TEST    = PROJECT_ROOT / "data" / "raw" / "student_test"
ARTIFACTS    = PROJECT_ROOT / "artifacts"

# ---------------------------------------------------------------------------
# Model loading (lazy, cached at startup)
# ---------------------------------------------------------------------------

_model: Any = None
_label_encoder: Any = None


def _load_model() -> tuple[Any, Any]:
    global _model, _label_encoder
    if _model is not None:
        return _model, _label_encoder

    model_path = ARTIFACTS / "best_model.joblib"
    le_path    = ARTIFACTS / "label_encoder.joblib"

    if model_path.exists() and le_path.exists():
        log.info("Loading model from %s", model_path)
        _model = joblib.load(model_path)
        _label_encoder = joblib.load(le_path)
        log.info("Model loaded: %s", type(_model).__name__)
    else:
        log.warning(
            "No trained model found at %s. "
            "/predict will return dummy output until a model is trained.",
            model_path,
        )
        _model = None
        _label_encoder = None

    return _model, _label_encoder


@app.on_event("startup")
def startup_event():
    _load_model()
    log.info("PathSnap API started ✓")


# ---------------------------------------------------------------------------
# Data loading helpers (raw text files)
# ---------------------------------------------------------------------------

def _find_trip_dir(trip_id: str) -> Path:
    """Return directory containing trip files, searching train then test."""
    for base in (DATA_TRAIN, DATA_TEST):
        d = base / trip_id
        if d.exists():
            return d
    raise FileNotFoundError(f"Trip '{trip_id}' not found in train or student_test.")


def _load_track(trip_dir: Path, trip_id: str) -> pd.DataFrame:
    p = trip_dir / f"{trip_id}.track"
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["lon", "lat", "timestamp"])
    return df


def _load_arcs(trip_dir: Path, trip_id: str) -> pd.DataFrame:
    p = trip_dir / f"{trip_id}.arcs"
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["from_node", "to_node"])
    df.index.name = "arc_id"
    return df.reset_index()


def _load_nodes(trip_dir: Path, trip_id: str) -> pd.DataFrame:
    p = trip_dir / f"{trip_id}.nodes"
    df = pd.read_csv(p, sep=r"\s+", header=None, names=["lon", "lat"])
    df.index.name = "node_id"
    return df.reset_index()


def _load_route(trip_dir: Path, trip_id: str) -> list[int] | None:
    """Load ground truth route (may not exist for evaluator trips)."""
    p = trip_dir / f"{trip_id}.route"
    if not p.exists():
        return None
    return pd.read_csv(p, header=None)[0].tolist()


# ---------------------------------------------------------------------------
# Minimal candidate generation (placeholder — will be replaced by A's work)
# ---------------------------------------------------------------------------

def _generate_candidates(
    track: pd.DataFrame,
    arcs: pd.DataFrame,
    nodes: pd.DataFrame,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    For each GPS point, return the nearest top_k arcs by midpoint proximity.
    This is a simple spatial heuristic placeholder.
    A's features pipeline will replace this with proper projected-distance candidates.

    Returns DataFrame with columns: [point_idx, arc_id, dist_m, (feature cols...)]
    """
    from math import radians, cos, sqrt

    def haversine_approx(lat1, lon1, lat2, lon2):
        """Fast flat-earth approximation in metres."""
        dlat = radians(lat2 - lat1) * 6_371_000
        dlon = radians(lon2 - lon1) * 6_371_000 * cos(radians((lat1 + lat2) / 2))
        return sqrt(dlat**2 + dlon**2)

    # Compute arc midpoints
    arc_mids = []
    for _, arc in arcs.iterrows():
        fn = int(arc["from_node"])
        tn = int(arc["to_node"])
        if fn < len(nodes) and tn < len(nodes):
            mid_lon = (nodes.iloc[fn]["lon"] + nodes.iloc[tn]["lon"]) / 2
            mid_lat = (nodes.iloc[fn]["lat"] + nodes.iloc[tn]["lat"]) / 2
        else:
            mid_lon, mid_lat = nodes.iloc[fn]["lon"], nodes.iloc[fn]["lat"]
        arc_mids.append((arc["arc_id"], mid_lon, mid_lat))

    arc_mid_arr = np.array([(a, lo, la) for a, lo, la in arc_mids])

    rows = []
    for idx, row in track.iterrows():
        gps_lon, gps_lat = row["lon"], row["lat"]

        # Euclidean approximation in degrees (fast, good enough for candidate gen)
        dlat = (arc_mid_arr[:, 2] - gps_lat) * 111_320
        dlon = (arc_mid_arr[:, 1] - gps_lon) * 111_320 * np.cos(np.radians(gps_lat))
        dists = np.sqrt(dlat**2 + dlon**2)

        nearest_idx = np.argsort(dists)[:top_k]
        for rank, nidx in enumerate(nearest_idx):
            rows.append({
                "point_idx": idx,
                "arc_id": int(arc_mid_arr[nidx, 0]),
                "dist_m": float(dists[nidx]),
                "rank": rank,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def _predict_route(trip_id: str) -> dict:
    """Full prediction pipeline. Returns structured result dict."""
    trip_dir = _find_trip_dir(trip_id)

    track = _load_track(trip_dir, trip_id)
    arcs  = _load_arcs(trip_dir, trip_id)
    nodes = _load_nodes(trip_dir, trip_id)
    true_route = _load_route(trip_dir, trip_id)

    log.info("Trip %s: %d GPS points, %d arcs", trip_id, len(track), len(arcs))

    # Candidate generation
    candidates = _generate_candidates(track, arcs, nodes, top_k=5)

    # Model scoring
    model, le = _load_model()

    if model is not None:
        # Build a minimal feature vector per candidate (extend when A's features arrive)
        feature_cols = ["dist_m", "rank"]
        X = candidates[feature_cols].to_numpy(dtype=np.float32)

        try:
            # predict_proba → pick top candidate per point
            proba = model.predict_proba(X)
            candidates["score"] = proba.max(axis=1)
        except Exception:
            # Fallback: score by inverse distance
            candidates["score"] = 1.0 / (candidates["dist_m"] + 1e-6)
    else:
        # No model yet — score by inverse distance (nearest arc wins)
        candidates["score"] = 1.0 / (candidates["dist_m"] + 1e-6)

    # Pick best-scoring arc per GPS point
    best = (
        candidates
        .sort_values("score", ascending=False)
        .groupby("point_idx", sort=True)
        .first()
        .reset_index()[["point_idx", "arc_id"]]
    )

    predicted_route = best["arc_id"].tolist()

    # Accuracy vs ground truth (server-side only)
    accuracy: float | None = None
    if true_route is not None:
        min_len = min(len(predicted_route), len(true_route))
        correct = sum(p == t for p, t in zip(predicted_route[:min_len], true_route[:min_len]))
        accuracy = correct / len(true_route)
        log.info("Trip %s accuracy: %.4f", trip_id, accuracy)

    return {
        "trip_id": trip_id,
        "n_gps_points": len(track),
        "predicted_route": predicted_route,
        "accuracy": accuracy,
        "model_used": type(model).__name__ if model else "distance_heuristic",
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    trip_id: str
    top_k: int = 5


class PredictResponse(BaseModel):
    trip_id: str
    n_gps_points: int
    predicted_route: list[int]
    accuracy: float | None
    model_used: str
    elapsed_ms: float


class TripListResponse(BaseModel):
    train_trips: list[str]
    test_trips: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}


@app.get("/trips", response_model=TripListResponse)
def list_trips():
    """Return all available trip IDs."""
    train = sorted(p.name for p in DATA_TRAIN.iterdir() if p.is_dir()) if DATA_TRAIN.exists() else []
    test  = sorted(p.name for p in DATA_TEST.iterdir()  if p.is_dir()) if DATA_TEST.exists()  else []
    return TripListResponse(train_trips=train, test_trips=test)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    Run the full PathSnap pipeline for a trip and return predicted route + accuracy.
    Ground-truth labels stay server-side; accuracy is computed internally.
    """
    log.info("POST /predict  trip_id=%s", req.trip_id)
    t0 = time.perf_counter()

    try:
        result = _predict_route(req.trip_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception("Prediction failed for trip %s", req.trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return PredictResponse(**result, elapsed_ms=round(elapsed_ms, 1))
