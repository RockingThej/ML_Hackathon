"""Loaders for the 4 raw file types of a PathSnap trip.

Layout: <root>/<8-digit trip folder>/<same>.{track,nodes,arcs,route}
trip_id = int(folder name), e.g. "00000042" -> 42.
"""
from pathlib import Path
import numpy as np


def list_trips(root):
    """Sorted list of int trip ids found under root (train/ or student_test/)."""
    return sorted(int(p.name) for p in Path(root).iterdir() if p.is_dir())


def _folder(root, trip_id):
    return Path(root) / f"{int(trip_id):08d}"


def load_track(root, trip_id):
    """(n,3) float64 array: lon, lat, timestamp(s). Files use CRLF + tabs."""
    f = _folder(root, trip_id) / f"{int(trip_id):08d}.track"
    return np.loadtxt(f, dtype=np.float64, ndmin=2)


def load_nodes(root, trip_id):
    """(n_nodes,2) float64: lon, lat. Row index = node_id."""
    f = _folder(root, trip_id) / f"{int(trip_id):08d}.nodes"
    return np.loadtxt(f, dtype=np.float64, ndmin=2)


def load_arcs(root, trip_id):
    """(n_arcs,2) int64: from_node, to_node. Row index = arc_id."""
    f = _folder(root, trip_id) / f"{int(trip_id):08d}.arcs"
    return np.loadtxt(f, dtype=np.int64, ndmin=2)


def load_route(root, trip_id):
    """(m,) int64 true arc_ids in travel order. None if the file is absent."""
    f = _folder(root, trip_id) / f"{int(trip_id):08d}.route"
    if not f.exists():
        return None
    return np.loadtxt(f, dtype=np.int64, ndmin=1)


def load_trip(root, trip_id):
    """Everything for one trip, as the dict that build_features() expects.
    Note: every trip has its OWN road network, arc_ids are only valid per trip."""
    return {
        "trip_id": int(trip_id),
        "track": load_track(root, trip_id),
        "nodes": load_nodes(root, trip_id),
        "arcs": load_arcs(root, trip_id),
        "route": load_route(root, trip_id),
    }
