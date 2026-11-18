import numpy as np
from pyproj import Transformer

def utm_epsg(lon, lat):
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone

def make_transformer(lon_mean, lat_mean):
    epsg = utm_epsg(lon_mean, lat_mean)
    return (Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True), epsg)

def project(transformer, lonlat):
    x, y = transformer.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])
