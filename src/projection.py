"""
projection.py - Metric Coordinate Reference System (CRS) projection for PathSnap.

Converts WGS84 angular coordinates (lon, lat in degrees) to a local metric projection
(UTM - Universal Transverse Mercator) to ensure Euclidean distances represent
true physical distances in meters with sub-millimeter precision and negligible scale distortion (<0.04%).
"""

from typing import Tuple, Union
import numpy as np
from pyproj import CRS, Transformer


class MetricProjector:
    """
    Handles forward and inverse coordinate transformations between
    geographic coordinates (EPSG:4326 WGS84 degrees) and a local
    metric projection (UTM meters).
    """

    def __init__(self, ref_lon: float, ref_lat: float):
        """
        Initialize projector based on trajectory reference point (centroid).
        Dynamically selects the appropriate UTM zone for minimal distortion.
        """
        self.ref_lon = float(ref_lon)
        self.ref_lat = float(ref_lat)
        self.utm_zone = int((self.ref_lon + 180.0) / 6.0) + 1
        self.is_north = (self.ref_lat >= 0.0)
        self.epsg_code = (32600 + self.utm_zone) if self.is_north else (32700 + self.utm_zone)
        self.crs_name = f"WGS 84 / UTM Zone {self.utm_zone}{'N' if self.is_north else 'S'} (EPSG:{self.epsg_code})"

        # Initialize pyproj transformers
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_metric = CRS.from_epsg(self.epsg_code)
        
        # always_xy=True ensures (lon, lat) -> (x, y) order regardless of CRS conventions
        self.forward_transformer = Transformer.from_crs(self.crs_wgs84, self.crs_metric, always_xy=True)
        self.inverse_transformer = Transformer.from_crs(self.crs_metric, self.crs_wgs84, always_xy=True)

    def transform(self, lon: Union[float, np.ndarray], lat: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Forward projection: (longitude, latitude) in degrees -> (x, y) in meters.
        """
        return self.forward_transformer.transform(lon, lat)

    def inverse(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Inverse projection: (x, y) in meters -> (longitude, latitude) in degrees.
        """
        return self.inverse_transformer.transform(x, y)


def get_projector_for_trip(trip_track_lon: np.ndarray, trip_track_lat: np.ndarray) -> MetricProjector:
    """Convenience constructor to get the optimal MetricProjector from trip coordinates."""
    mean_lon = float(np.mean(trip_track_lon))
    mean_lat = float(np.mean(trip_track_lat))
    return MetricProjector(ref_lon=mean_lon, ref_lat=mean_lat)
