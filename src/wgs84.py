from __future__ import annotations

import math
import numpy as np

A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = A / math.sqrt(1.0 - E2 * sin_lat * sin_lat)

    x = (n + alt_m) * cos_lat * math.cos(lon)
    y = (n + alt_m) * cos_lat * math.sin(lon)
    z = (n * (1.0 - E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    # Bowring-style iterative conversion.
    lon = math.atan2(y, x)
    p = math.sqrt(x * x + y * y)
    lat = math.atan2(z, p * (1.0 - E2))

    for _ in range(8):
        sin_lat = math.sin(lat)
        n = A / math.sqrt(1.0 - E2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - E2 * n / (n + alt)))

    sin_lat = math.sin(lat)
    n = A / math.sqrt(1.0 - E2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(lon), alt


def ecef_to_enu_matrix(lat0_deg: float, lon0_deg: float) -> np.ndarray:
    lat = math.radians(lat0_deg)
    lon = math.radians(lon0_deg)
    return np.array(
        [
            [-math.sin(lon), math.cos(lon), 0.0],
            [-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat)],
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
        ],
        dtype=float,
    )


def geodetic_to_enu(lat: float, lon: float, alt: float, origin_lat: float, origin_lon: float, origin_alt: float) -> np.ndarray:
    origin = geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
    point = geodetic_to_ecef(lat, lon, alt)
    rot = ecef_to_enu_matrix(origin_lat, origin_lon)
    return rot @ (point - origin)


def enu_to_geodetic(enu: np.ndarray, origin_lat: float, origin_lon: float, origin_alt: float) -> tuple[float, float, float]:
    origin = geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
    rot = ecef_to_enu_matrix(origin_lat, origin_lon)
    ecef = origin + rot.T @ enu
    return ecef_to_geodetic(float(ecef[0]), float(ecef[1]), float(ecef[2]))
