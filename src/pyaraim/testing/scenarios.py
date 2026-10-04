"""Generation of test geometries and measurement epochs.

* :func:`random_sky` - random but realistic satellite constellations (uniform in azimuth, ``sin(el)``
  distributed above the mask), for property-based and stress tests;
* :func:`sky_from_almanac` - the satellites seen by one user at one epoch from real almanacs;
* :class:`Epoch` / :func:`make_epoch` - a satellite set together with simulated pseudorange residuals
  (nominal noise plus optional faults), the unit exchanged with the monitor;
* :func:`outage_cases` - single/double satellite and constellation outages of a nominal sky.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constants import DEG
from ..errormodels import ED259A, UserErrorModel
from ..geometry import SatelliteSet
from ..isd import ISD
from ..simulation.almanac import Almanac
from ..simulation.availability import Scenario, epoch_satellite_sets
from ..simulation.grid import UserGrid
from .faults import FaultSpec, apply_faults

__all__ = ["random_sky", "sky_from_almanac", "Epoch", "make_epoch", "outage_cases"]


def random_sky(n_per_const: Dict[str, int], isd: ISD, rng: np.random.Generator | int | None = None,
               mask_deg: float = 5.0, error_model: UserErrorModel | None = None) -> SatelliteSet:
    """Random satellites above ``mask_deg``.

    Elevations follow the distribution of satellites seen from the ground (``sin(el)`` uniform
    between ``sin(mask)`` and ``1`` is a good proxy for a uniform sky), azimuths are uniform.
    """
    rng = np.random.default_rng(rng)
    los, names = [], []
    for name, n in n_per_const.items():
        s = rng.uniform(np.sin(mask_deg * DEG), 1.0, n)
        az = rng.uniform(0, 2 * np.pi, n)
        c = np.sqrt(1 - s**2)
        los.append(np.column_stack([c * np.sin(az), c * np.cos(az), s]))
        names += [name] * n
    return SatelliteSet.from_los(np.vstack(los), names, isd, error_model or ED259A())


def sky_from_almanac(almanacs: Sequence[Almanac], isd: ISD, lat: float, lon: float, t: float = 0.0,
                     mask_deg: float = 5.0, error_model: UserErrorModel | None = None) -> SatelliteSet:
    """Satellites visible at ``(lat, lon)`` at time ``t`` [s since toa] (raises if none)."""
    scn = Scenario(list(almanacs), isd, None, UserGrid.points(lat, lon), np.array([t]), mask_deg,
                   error_model or ED259A())
    for _, sats in epoch_satellite_sets(scn, t):
        return sats
    raise ValueError("no satellite visible")


@dataclass
class Epoch:
    """One processing epoch: geometry, simulated residuals and the ground truth."""

    sats: SatelliteSet
    y: np.ndarray                         # pseudorange residuals
    position_error: np.ndarray            # true error of the all-in-view WLS solution (E, N, U)
    faults: List[FaultSpec] = field(default_factory=list)
    truth_faulted: np.ndarray = None      # boolean mask of the satellites affected by a fault


def make_epoch(sats: SatelliteSet, faults: Optional[Sequence[FaultSpec]] = None,
               rng: np.random.Generator | int | None = None, noise: str = "acc",
               nominal_bias: str = "none") -> Epoch:
    """Simulate residuals ``y = noise + nominal bias + faults``.

    Parameters
    ----------
    noise:
        ``"acc"`` draws noise from ``C_acc`` (the realistic accuracy/continuity model, used to
        study false alerts), ``"int"`` from ``C_int`` (the conservative integrity model, used to
        check integrity bounds).
    nominal_bias:
        ``"none"`` or ``"worst"`` (bias of magnitude ``b_nom`` and random sign per satellite).
    """
    rng = np.random.default_rng(rng)
    var = sats.var_acc if noise == "acc" else sats.var_int
    y = rng.normal(size=sats.n) * np.sqrt(var)
    if nominal_bias == "worst":
        y = y + rng.choice([-1.0, 1.0], sats.n) * (sats.b_int if noise == "int" else sats.b_acc)
    faults = list(faults or [])
    y, mask = apply_faults(sats, y, faults)
    # true error of the all-in-view solution: x_hat - x = S y (the truth is x = 0)
    from ..solution import wls_coefficients
    s, _ = wls_coefficients(sats.G, 1.0 / sats.var_int, np.ones((1, sats.n), bool))
    return Epoch(sats, y, s[0] @ y, faults, mask)


def outage_cases(sats: SatelliteSet, max_out: int = 1) -> Dict[str, SatelliteSet]:
    """Nominal sky plus every single satellite outage (``max_out=1``) or every pair (``max_out=2``)."""
    from itertools import combinations
    cases = {"nominal": sats}
    for k in range(1, max_out + 1):
        for comb in combinations(range(sats.n), k):
            cases["-".join(sats.ids[i] for i in comb) + " out"] = sats.drop(comb)
    return cases
