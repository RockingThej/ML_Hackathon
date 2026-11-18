"""
src/data/loaders.py - Loaders for PathSnap raw data files (.track, .nodes, .arcs, .route).
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd


@dataclass
class TripData:
    trip_id: int              # integer trip ID (e.g. 1, 7, 40)
    trip_id_str: str          # string folder name (e.g. "00000001")
    split: str                # "train" or "student_test"
    track: pd.DataFrame       # columns: lon, lat, timestamp, dt
    nodes: np.ndarray         # shape (N_nodes, 2): [lon, lat], row index = node_id
    arcs: np.ndarray          # shape (N_arcs, 2): [from_node, to_node], row index = arc_id
    route: Optional[np.ndarray] # shape (N_route_arcs,): sequence of true arc_ids
    metadata: Dict[str, Any]


def load_track(track_path: str) -> pd.DataFrame:
    """
    Load .track file: lon, lat, timestamp (no header, whitespace-separated).
    Returns DataFrame with columns ['lon', 'lat', 'timestamp', 'dt'].
    """
    if not os.path.exists(track_path):
        raise FileNotFoundError(f"Track file not found: {track_path}")
    
    df = pd.read_csv(
        track_path,
        sep=r"\s+",
        header=None,
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64}
    )
    df["dt"] = df["timestamp"].diff().fillna(1.0).astype(np.float32)
    return df


def load_nodes(nodes_path: str) -> np.ndarray:
    """
    Load .nodes file: lon, lat (row index = node_id).
    Returns float64 numpy array of shape (N, 2).
    """
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Nodes file not found: {nodes_path}")
    
    nodes = np.loadtxt(nodes_path, dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)
    return nodes


def load_arcs(arcs_path: str) -> np.ndarray:
    """
    Load .arcs file: from_node, to_node (row index = arc_id).
    Returns int64 numpy array of shape (N, 2).
    """
    if not os.path.exists(arcs_path):
        raise FileNotFoundError(f"Arcs file not found: {arcs_path}")
    
    arcs = np.loadtxt(arcs_path, dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)
    return arcs


def load_route(route_path: str) -> Optional[np.ndarray]:
    """
    Load .route file: sequence of true arc_ids.
    Returns 1D int64 numpy array of arc_ids, or None if file doesn't exist.
    """
    if not os.path.exists(route_path):
        return None
    
    try:
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = np.array([int(route)], dtype=np.int64)
        return route
    except Exception:
        return np.array([], dtype=np.int64)


def get_trip_ids(data_dir: str = "data/raw", split: str = "train") -> List[str]:
    """Return sorted list of trip folder strings in a split."""
    split_dir = os.path.join(data_dir, split)
    if not os.path.exists(split_dir):
        # Fallback to direct path or parent dir
        alt = os.path.join("data", "raw", split)
        if os.path.exists(alt):
            split_dir = alt
        else:
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
            
    return sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])


def load_trip(trip_id_or_folder: Union[int, str], data_dir: str = "data/raw", split: str = "train") -> TripData:
    """
    Load all 4 files for a given trip. Accepts either integer trip ID (e.g. 1) or string folder (e.g. '00000001').
    """
    if isinstance(trip_id_or_folder, int):
        folder_name = f"{trip_id_or_folder:08d}"
        trip_int = trip_id_or_folder
    else:
        folder_name = trip_id_or_folder
        trip_int = int(trip_id_or_folder)

    trip_dir = os.path.join(data_dir, split, folder_name)
    if not os.path.exists(trip_dir):
        alt = os.path.join("data", "raw", split, folder_name)
        if os.path.exists(alt):
            trip_dir = alt
        else:
            raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    track_path = os.path.join(trip_dir, f"{folder_name}.track")
    nodes_path = os.path.join(trip_dir, f"{folder_name}.nodes")
    arcs_path = os.path.join(trip_dir, f"{folder_name}.arcs")
    route_path = os.path.join(trip_dir, f"{folder_name}.route")

    track = load_track(track_path)
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)
    route = load_route(route_path)

    metadata = {
        "trip_id": trip_int,
        "trip_folder": folder_name,
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
        trip_id=trip_int,
        trip_id_str=folder_name,
        split=split,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        metadata=metadata
    )


def scan_all_trips(data_dir: str = "data/raw", splits: List[str] = ["train", "student_test"]) -> pd.DataFrame:
    """Scan all trips across splits and return a comprehensive summary table."""
    records = []
    for split in splits:
        trip_folders = get_trip_ids(data_dir, split)
        for folder in trip_folders:
            trip = load_trip(folder, data_dir, split)
            m = trip.metadata
            records.append({
                "split": split,
                "trip_id": m["trip_id"],
                "trip_folder": m["trip_folder"],
                "num_track_points": m["num_track_points"],
                "num_nodes": m["num_nodes"],
                "num_arcs": m["num_arcs"],
                "num_route_arcs": m["num_route_arcs"],
                "duration_s": m["duration_seconds"],
                "lon_min": m["lon_min"],
                "lon_max": m["lon_max"],
                "lat_min": m["lat_min"],
                "lat_max": m["lat_max"],
            })
    return pd.DataFrame(records)
