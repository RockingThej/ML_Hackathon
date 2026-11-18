"""
loaders.py - Robust data loaders for PathSnap GPS trajectory and road network data.

Raw file types:
- .track: 3 whitespace-separated columns [longitude, latitude, timestamp_seconds]
- .nodes: 2 whitespace-separated columns [longitude, latitude] (0-based row index is node_id)
- .arcs:  2 whitespace-separated columns [from_node_id, to_node_id] (0-based row index is arc_id)
- .route: 1 integer column [arc_id] representing chronological route traversal
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd


@dataclass
class TripData:
    trip_id: str
    split: str
    track: pd.DataFrame        # lon, lat, timestamp, dt, speed_approx
    nodes: np.ndarray          # shape (N_nodes, 2): [lon, lat]
    arcs: np.ndarray           # shape (N_arcs, 2): [from_node, to_node]
    route: Optional[np.ndarray] # shape (N_route_arcs,): arc IDs traversed
    metadata: Dict[str, Any]


def load_track(track_path: str) -> pd.DataFrame:
    """Load .track file into a DataFrame."""
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")
    
    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64}
    )
    
    # Compute sampling intervals and basic sanity metrics
    df["dt"] = df["timestamp"].diff().fillna(1.0)
    # Check for NaN / infinite values
    if df[["lon", "lat", "timestamp"]].isna().any().any():
        raise ValueError(f"Found NaN values in track file: {track_path}")
    
    return df


def load_nodes(nodes_path: str) -> np.ndarray:
    """Load .nodes file into a 2D numpy array [lon, lat]."""
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    
    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    return nodes


def load_arcs(arcs_path: str) -> np.ndarray:
    """Load .arcs file into a 2D numpy array [from_node, to_node]."""
    if not os.path.exists(arcs_path):
        raise FileNotFoundError(f"Arcs file not found: {arcs_path}")
    
    arcs = np.loadtxt(arcs_path, dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)
    return arcs


def load_route(route_path: str) -> Optional[np.ndarray]:
    """Load .route file into a 1D numpy array of arc IDs."""
    if not os.path.exists(route_path):
        return None
    
    try:
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = np.array([int(route)], dtype=np.int64)
        return route
    except Exception:
        # Route file might be empty
        return np.array([], dtype=np.int64)


def get_trip_ids(data_dir: str, split: str = "train") -> List[str]:
    """Return sorted list of trip IDs in a given split directory."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        # Check if wrapped inside 'Student folder'
        alt = os.path.join(data_dir, "Student folder", split)
        if os.path.exists(alt):
            split_dir = alt
        else:
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
    return sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])


def load_trip(trip_id: str, data_dir: str, split: str = "train") -> TripData:
    """Load all raw data for a specific trip."""
    # Find trip directory
    trip_dir = os.path.join(data_dir, split, trip_id)
    if not os.path.exists(trip_dir):
        alt = os.path.join(data_dir, "Student folder", split, trip_id)
        if os.path.exists(alt):
            trip_dir = alt
        else:
            raise FileNotFoundError(f"Trip directory not found for trip {trip_id} in {data_dir}")

    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    track = load_track(track_path)
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)
    route = load_route(route_path)

    metadata = {
        "trip_id": trip_id,
        "split": split,
        "num_track_points": len(track),
        "num_nodes": len(nodes),
        "num_arcs": len(arcs),
        "num_route_arcs": len(route) if route is not None else 0,
        "duration_seconds": float(track["timestamp"].iloc[-1] - track["timestamp"].iloc[0]),
        "lon_min": float(track["lon"].min()),
        "lon_max": float(track["lon"].max()),
        "lat_min": float(track["lat"].min()),
        "lat_max": float(track["lat"].max()),
    }

    return TripData(
        trip_id=trip_id,
        split=split,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        metadata=metadata
    )


def scan_all_trips(data_dir: str, splits: List[str] = ["train", "student_test"]) -> pd.DataFrame:
    """Scan all trips across splits and return a consolidated inventory table."""
    records = []
    for split in splits:
        try:
            trip_ids = get_trip_ids(data_dir, split)
        except Exception:
            continue
        for tid in trip_ids:
            try:
                trip = load_trip(tid, data_dir, split)
                records.append({
                    "split": split,
                    "trip_id": tid,
                    "num_track_points": trip.metadata["num_track_points"],
                    "num_nodes": trip.metadata["num_nodes"],
                    "num_arcs": trip.metadata["num_arcs"],
                    "num_route_arcs": trip.metadata["num_route_arcs"],
                    "duration_s": trip.metadata["duration_seconds"],
                    "lon_min": trip.metadata["lon_min"],
                    "lon_max": trip.metadata["lon_max"],
                    "lat_min": trip.metadata["lat_min"],
                    "lat_max": trip.metadata["lat_max"],
                })
            except Exception as e:
                records.append({
                    "split": split,
                    "trip_id": tid,
                    "error": str(e)
                })
    return pd.DataFrame(records)
