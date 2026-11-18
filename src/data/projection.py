"""
src/data/projection.py - Metric Coordinate Reference System (CRS) projection.

Converts lon/lat (WGS84 degrees) to metric CRS using EPSG:32632 (UTM Zone 32N)
as specified in the project guidelines (lon ~11, lat ~47.6).
All distance and geometry calculations must use metric coordinates in meters.
"""

from typing import Tuple, Union
import numpy as np
from pyproj import CRS, Transformer

DEFAULT_EPSG = 32632  # WGS 84 / UTM zone 32N


class MetricProjector:
    """
    Transforms coordinates between EPSG:4326 (WGS84 degrees) and EPSG:32632 (UTM 32N meters).
    """

    def __init__(self, epsg_code: int = DEFAULT_EPSG):
        self.epsg_code = epsg_code
        self.crs_wgs84 = CRS.from_epsg(4326)
        self.crs_metric = CRS.from_epsg(epsg_code)
        
        # always_xy=True ensures (lon, lat) -> (x, y) order
        self.forward_transformer = Transformer.from_crs(self.crs_wgs84, self.crs_metric, always_xy=True)
        self.inverse_transformer = Transformer.from_crs(self.crs_metric, self.crs_wgs84, always_xy=True)

    def transform(
        self,
        lon: Union[float, np.ndarray],
        lat: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert (lon, lat) in degrees -> (x, y) in meters."""
        return self.forward_transformer.transform(lon, lat)

    def inverse(
        self,
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Convert (x, y) in meters -> (lon, lat) in degrees."""
        return self.inverse_transformer.transform(x, y)


_global_projector = MetricProjector(DEFAULT_EPSG)


def project_coordinates(
    lon: Union[float, np.ndarray],
    lat: Union[float, np.ndarray],
    epsg: int = DEFAULT_EPSG
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convenience function to project lon/lat arrays to metric x/y arrays."""
    if epsg == DEFAULT_EPSG:
        return _global_projector.transform(lon, lat)
    return MetricProjector(epsg).transform(lon, lat)
