"""Almanac reading and satellite orbit propagation.

Reads almanacs in the Yuma-like text format shipped with MAAST/MAAST-for-ARAIM::

    ******** Week  703 almanac for PRN-01 ********
    ID:                         01
    Health:                     000
    Eccentricity:               0.0
    Time of Applicability(s):   344063.0000
    Orbital Inclination(rad):   0.9599310886
    Rate of Right Ascen(r/s):   0.0
    SQRT(A)  (m 1/2):           5153.620087
    Right Ascen at TOA(rad):    0.4762078504E+001
    Argument of Perigee(rad):   0.0
    Mean Anom(rad):             0.4679681510E+001
    Af0(s):  Af1(s/s):  week:

and propagates the Keplerian orbit to ECEF with the algorithm of IS-GPS-200 (Table 20-IV).
Satellite time is measured from the almanac time of applicability (``t = 0`` is ``toa``), as in
the MAAST scripts where the simulation starts at ``TStart = 0``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np

from ..constants import MU_EARTH, OMEGA_EARTH

__all__ = ["Almanac", "AlmanacEntry", "read_almanac", "bundled_almanac", "BUNDLED_ALMANACS",
           "constellation_from_almanacs"]

#: Almanacs shipped with the package (the WG-C scenarios of the MAAST-for-ARAIM repository).
BUNDLED_ALMANACS = {
    "gps_24": ("GPS", "gps_24.alm", "GPS 24-slot nominal constellation (WG-C)"),
    "gps_24_minus1": ("GPS", "gps_24_minus1.alm", "GPS 24 with one satellite out"),
    "gps_24_plus3": ("GPS", "gps_24_plus3.alm", "GPS 24 + 3 expandable slots"),
    "gal_24": ("Galileo", "gal_24.alm", "Galileo 24 satellites (WG-C)"),
    "gal_24_minus1": ("Galileo", "gal_24_minus1.alm", "Galileo 24 with one satellite out"),
    "gal_24_plus3": ("Galileo", "gal_24_plus3.alm", "Galileo 24 + 3 spares"),
    "gal_24_alt": ("Galileo", "gal_24_alt.alm", "Alternative Galileo 24 file of the repository"),
}

_FIELDS = {
    "ID": "prn",
    "Health": "health",
    "Eccentricity": "ecc",
    "Time of Applicability(s)": "toa",
    "Orbital Inclination(rad)": "inc",
    "Rate of Right Ascen(r/s)": "omega_dot",
    "SQRT(A)  (m 1/2)": "sqrt_a",
    "Right Ascen at TOA(rad)": "omega0",
    "Right Ascen at Week(rad)": "omega0",   # standard Yuma spelling
    "Argument of Perigee(rad)": "argp",
    "Mean Anom(rad)": "m0",
    "Af0(s)": "af0",
    "Af1(s/s)": "af1",
    "week": "week",
}


@dataclass(frozen=True)
class AlmanacEntry:
    prn: int
    health: int
    ecc: float
    toa: float
    inc: float
    omega_dot: float
    sqrt_a: float
    omega0: float
    argp: float
    m0: float
    af0: float = 0.0
    af1: float = 0.0
    week: int = 0


class Almanac:
    """A set of satellites of one constellation."""

    def __init__(self, entries: Sequence[AlmanacEntry], constellation: str = "GPS"):
        self.entries = list(entries)
        self.constellation = constellation

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def prns(self) -> List[int]:
        return [e.prn for e in self.entries]

    def healthy(self) -> "Almanac":
        return Almanac([e for e in self.entries if e.health == 0], self.constellation)

    def without(self, prns: Iterable[int]) -> "Almanac":
        """Almanac with some satellites removed (outage scenarios)."""
        bad = set(prns)
        return Almanac([e for e in self.entries if e.prn not in bad], self.constellation)

    # -------------------------------------------------------------- propagation
    def positions_ecef(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """ECEF positions [m] at time(s) ``t`` [s since the almanac ``toa``].

        Returns ``(n_sat, 3)`` for scalar ``t`` and ``(n_t, n_sat, 3)`` for an array.
        """
        t_arr = np.atleast_1d(np.asarray(t, dtype=float))
        e = self.entries
        ecc = np.array([x.ecc for x in e])
        a = np.array([x.sqrt_a for x in e]) ** 2
        inc = np.array([x.inc for x in e])
        om0 = np.array([x.omega0 for x in e])
        omd = np.array([x.omega_dot for x in e])
        w = np.array([x.argp for x in e])
        m0 = np.array([x.m0 for x in e])
        toa = np.array([x.toa for x in e])
        tk = t_arr[:, None]                                         # time from toa
        n = np.sqrt(MU_EARTH / a**3)
        M = m0 + n * tk
        E = M.copy()
        for _ in range(12):                                         # Kepler's equation
            E = E - (E - ecc * np.sin(E) - M) / (1.0 - ecc * np.cos(E))
        nu = np.arctan2(np.sqrt(1 - ecc**2) * np.sin(E), np.cos(E) - ecc)
        phi = nu + w
        r = a * (1.0 - ecc * np.cos(E))
        xo, yo = r * np.cos(phi), r * np.sin(phi)
        om = om0 + (omd - OMEGA_EARTH) * tk - OMEGA_EARTH * toa
        ci, si = np.cos(inc), np.sin(inc)
        x = xo * np.cos(om) - yo * ci * np.sin(om)
        y = xo * np.sin(om) + yo * ci * np.cos(om)
        z = yo * si
        out = np.stack([x, y, z], axis=-1)
        return out[0] if np.isscalar(t) or np.ndim(t) == 0 else out


def read_almanac(path: Union[str, Path], constellation: str = "GPS") -> Almanac:
    """Parse an almanac file (see module docstring for the format)."""
    text = Path(path).read_text().replace("\r", "")
    blocks = re.split(r"\*{4,}\s*Week.*?almanac for.*?\*{4,}", text)[1:]
    entries = []
    for blk in blocks:
        vals: Dict[str, float] = {}
        for line in blk.splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            for k, name in _FIELDS.items():
                if key == k.strip():
                    try:
                        vals[name] = float(val.split()[0])
                    except (IndexError, ValueError):
                        pass
        if "sqrt_a" not in vals:
            continue
        vals["prn"] = int(vals.get("prn", len(entries) + 1))
        vals["health"] = int(vals.get("health", 0))
        vals["week"] = int(vals.get("week", 0))
        entries.append(AlmanacEntry(**vals))
    if not entries:
        raise ValueError(f"no almanac entries found in {path}")
    return Almanac(entries, constellation)


def bundled_almanac(name: str) -> Almanac:
    """Load one of the almanacs shipped with the package (see :data:`BUNDLED_ALMANACS`)."""
    if name not in BUNDLED_ALMANACS:
        raise KeyError(f"unknown almanac {name!r}; choose from {sorted(BUNDLED_ALMANACS)}")
    const, fname, _ = BUNDLED_ALMANACS[name]
    ref = resources.files("pyaraim") / "data" / "almanacs" / fname
    with resources.as_file(ref) as p:
        return read_almanac(p, const)


def constellation_from_almanacs(*names: str) -> List[Almanac]:
    """Convenience: ``constellation_from_almanacs("gps_24", "gal_24")``."""
    return [bundled_almanac(n) for n in names]
