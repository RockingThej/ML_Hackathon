from pathlib import Path
import numpy as np

def list_trips(root):
    return sorted((int(p.name) for p in Path(root).iterdir() if p.is_dir()))

def _folder(root, trip_id):
    return Path(root) / f'{int(trip_id):08d}'

def load_track(root, trip_id):
    f = _folder(root, trip_id) / f'{int(trip_id):08d}.track'
    return np.loadtxt(f, dtype=np.float64, ndmin=2)

def load_nodes(root, trip_id):
    f = _folder(root, trip_id) / f'{int(trip_id):08d}.nodes'
    return np.loadtxt(f, dtype=np.float64, ndmin=2)

def load_arcs(root, trip_id):
    f = _folder(root, trip_id) / f'{int(trip_id):08d}.arcs'
    return np.loadtxt(f, dtype=np.int64, ndmin=2)

def load_route(root, trip_id):
    f = _folder(root, trip_id) / f'{int(trip_id):08d}.route'
    if not f.exists():
        return None
    return np.loadtxt(f, dtype=np.int64, ndmin=1)

def load_trip(root, trip_id):
    return {'trip_id': int(trip_id), 'track': load_track(root, trip_id), 'nodes': load_nodes(root, trip_id), 'arcs': load_arcs(root, trip_id), 'route': load_route(root, trip_id)}
