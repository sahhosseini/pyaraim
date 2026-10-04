"""The numerical example of Appendix J of Blanch et al., IEEE TAES 51(1), 2015.

Ten satellites in two constellations with the geometry matrix of Eq. (128), ``sigma_URA = 0.75 m``,
``sigma_URE = 0.5 m``, ``P_sat = 1e-4``, ``P_const = 1e-5`` and ``b_nom = 0.5 m``. The paper reports
``VPL = 19.10 m``, ``HPL = 14.1 m``, ``EMT = 8.3 m`` and ``sigma_v,acc = 1.47 m``.
"""
from __future__ import annotations

import numpy as np

from .config import ARAIMConfig
from .geometry import SatelliteSet

#: Geometry matrix of Eq. (128)
G = np.array([
    [0.0225, 0.9951, -0.0966, 1, 0], [0.6750, -0.6900, -0.2612, 1, 0],
    [0.0723, -0.6601, -0.7477, 1, 0], [-0.9398, 0.2553, -0.2269, 1, 0],
    [-0.5907, -0.7539, -0.2877, 1, 0], [-0.3236, -0.0354, -0.9455, 0, 1],
    [-0.6748, 0.4356, -0.5957, 0, 1], [0.0938, -0.7004, -0.7075, 0, 1],
    [0.5571, 0.3088, -0.7709, 0, 1], [0.6622, 0.6958, -0.2780, 0, 1]])

#: Diagonals of ``C_int`` and ``C_acc`` given in Eq. (131) (they include the Appendix A user error and troposphere)
C_INT = np.array([3.8865, 1.4377, 0.8604, 1.6383, 1.3229, 0.8434, 0.8963, 0.8669, 0.8573, 1.3616])
C_ACC = np.array([3.5740, 1.1252, 0.5479, 1.3258, 1.0104, 0.5309, 0.5838, 0.5544, 0.5448, 1.0491])

#: Published results
PUBLISHED = {"vpl": 19.10, "hpl": 14.1, "emt": 8.3, "sigma_acc": 1.47, "k_fa3": 5.3953,
             "sigma3_const": (2.5760, 2.5577), "sigma_ss3_const": (1.5307, 1.5292)}


def satellite_set() -> SatelliteSet:
    return SatelliteSet.from_geometry_matrix(G, C_INT, C_ACC, b_int=0.5, b_acc=0.0, p_sat=1e-4,
                                             p_const=1e-5)


def config() -> ARAIMConfig:
    """Constants of Table II (``P_FA_VERT = 3.9e-6``, ``P_FA_HOR = 9e-8`` ...), fault detection only."""
    return ARAIMConfig.baseline_2015()
