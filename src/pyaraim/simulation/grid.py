"""User grids on the WGS-84 ellipsoid with local East-North-Up frames."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..constants import DEG, WGS84_A, WGS84_E2

__all__ = ["UserGrid", "REGIONS", "geodetic_to_ecef", "enu_basis"]

#: Approximate bounding boxes ``(lat_min, lat_max, lon_min, lon_max)`` in degrees. They stand in
#: for the polygon files (``usrconus.dat`` ...) of MAAST, which are not redistributed here.
REGIONS: Dict[str, Tuple[float, float, float, float]] = {
    "world": (-90.0, 90.0, -180.0, 180.0),
    "conus": (24.0, 50.0, -125.0, -66.0),
    "alaska": (51.0, 72.0, -170.0, -130.0),
    "canada": (42.0, 70.0, -141.0, -52.0),
    "mexico": (14.0, 33.0, -118.0, -86.0),
    "north_america": (14.0, 72.0, -170.0, -52.0),
    "europe": (35.0, 72.0, -12.0, 40.0),
    "japan": (24.0, 46.0, 123.0, 146.0),
    "brazil": (-34.0, 5.0, -74.0, -34.0),
}


def geodetic_to_ecef(lat_deg, lon_deg, h=0.0) -> np.ndarray:
    """WGS-84 geodetic (deg, deg, m) -> ECEF [m]."""
    lat, lon = np.asarray(lat_deg, float) * DEG, np.asarray(lon_deg, float) * DEG
    N = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1.0 - WGS84_E2) + h) * np.sin(lat)], axis=-1)


def enu_basis(lat_deg, lon_deg) -> np.ndarray:
    """Rows East, North, Up (ECEF components) of the local frame: shape ``(..., 3, 3)``."""
    lat, lon = np.asarray(lat_deg, float) * DEG, np.asarray(lon_deg, float) * DEG
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    e = np.stack([-so, co, np.zeros_like(so)], axis=-1)
    n = np.stack([-sl * co, -sl * so, cl], axis=-1)
    u = np.stack([cl * co, cl * so, sl], axis=-1)
    return np.stack([e, n, u], axis=-2)


@dataclass
class UserGrid:
    """A set of ground (or airborne) users.

    Attributes
    ----------
    lat, lon : (n_users,) degrees
    height : (n_users,) metres above the ellipsoid
    """

    lat: np.ndarray
    lon: np.ndarray
    height: np.ndarray

    def __post_init__(self):
        self.lat = np.asarray(self.lat, float)
        self.lon = np.asarray(self.lon, float)
        self.height = np.broadcast_to(np.asarray(self.height, float), self.lat.shape).copy()
        self.ecef = geodetic_to_ecef(self.lat, self.lon, self.height)
        self.basis = enu_basis(self.lat, self.lon)

    def __len__(self) -> int:
        return len(self.lat)

    @property
    def weight(self) -> np.ndarray:
        """Area weight ``cos(lat)`` of the cell around each grid point (as in ``coverage.m``)."""
        return np.cos(self.lat * DEG)

    @classmethod
    def regular(cls, lat_step: float = 10.0, lon_step: float = 10.0,
                box: Tuple[float, float, float, float] = REGIONS["world"], height: float = 0.0,
                drop_poles: bool = True) -> "UserGrid":
        """Regular lat/lon grid inside ``box = (lat_min, lat_max, lon_min, lon_max)``."""
        la0, la1, lo0, lo1 = box
        lats = np.arange(la0, la1 + 1e-9, lat_step)
        if drop_poles:
            lats = lats[np.abs(lats) < 90.0 - 1e-9]
        lons = np.arange(lo0, lo1 - 1e-9 if lo1 - lo0 >= 360 else lo1 + 1e-9, lon_step)
        LA, LO = np.meshgrid(lats, lons, indexing="ij")
        return cls(LA.ravel(), LO.ravel(), height)

    @classmethod
    def world(cls, lat_step: float = 10.0, lon_step: float = 10.0, height: float = 0.0) -> "UserGrid":
        return cls.regular(lat_step, lon_step, REGIONS["world"], height)

    @classmethod
    def region(cls, name: str, lat_step: float = 2.0, lon_step: float = 2.0, height: float = 0.0):
        if name not in REGIONS:
            raise KeyError(f"unknown region {name!r}; choose from {sorted(REGIONS)}")
        return cls.regular(lat_step, lon_step, REGIONS[name], height)

    @classmethod
    def points(cls, lat, lon, height=0.0) -> "UserGrid":
        return cls(np.atleast_1d(lat), np.atleast_1d(lon), height)

    def los_enu(self, user: int, sat_ecef: np.ndarray):
        """ENU unit line-of-sight vectors from ``user`` to the satellites, plus ranges [m]."""
        d = sat_ecef - self.ecef[user]
        rng = np.linalg.norm(d, axis=-1)
        return (d / rng[..., None]) @ self.basis[user].T, rng
