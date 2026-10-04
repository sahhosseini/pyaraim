"""Physical and signal constants used throughout :mod:`pyaraim`.

The numerical values follow ``init_const.m`` of the MAAST-for-ARAIM reference
repository (WGS-84 / GPS ICD values), so that results are directly comparable.
"""
from __future__ import annotations

import math

#: Speed of light [m/s]
C_LIGHT = 299_792_458.0

#: L1/E1, L2 and L5/E5a carrier frequencies [Hz]
F_L1 = 1575.42e6
F_L2 = 1227.60e6
F_L5 = 1176.45e6

#: Variance amplification of the L1/L5 iono-free pseudorange combination for
#: uncorrelated, equal-variance noise on both frequencies (``GAMMA_L1_L5`` in
#: MAAST):  (f1^4 + f5^4) / (f1^2 - f5^2)^2.
GAMMA_L1L5_NOISE = (F_L1**4 + F_L5**4) / (F_L1**2 - F_L5**2) ** 2

#: Ratio (f1/f5)^2, the first-order ionospheric scale factor between L1 and L5.
GAMMA_L1L5_IONO = (F_L1 / F_L5) ** 2

#: Earth gravitational parameter [m^3/s^2] (WGS-84, as in MAAST)
MU_EARTH = 3.986004418e14
#: Earth rotation rate [rad/s] (GPS ICD value, as in MAAST)
OMEGA_EARTH = 7.292115e-5
#: WGS-84 semi-major axis [m] and flattening
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

DEG = math.pi / 180.0
SECONDS_PER_HOUR = 3600.0
