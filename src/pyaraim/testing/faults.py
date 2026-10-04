"""Fault models for test-data generation.

A fault adds a bias vector to the pseudorange residuals. Faults are described by
:class:`FaultSpec` and applied with :func:`apply_faults`. :func:`critical_fault` builds, for a
given set of faulted measurements, the fault with the *largest position error for a given
solution-separation statistic* - i.e. the worst case that the solution-separation monitor has to
withstand (it is the direction of ``S^(0)_q`` restricted to the faulted set, cf. Appendix G of
the TAES paper).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from ..geometry import SatelliteSet
from ..solution import wls_coefficients

__all__ = ["FaultSpec", "apply_faults", "satellite_fault", "constellation_fault", "ramp_fault",
           "critical_fault"]


@dataclass(frozen=True)
class FaultSpec:
    """A fault on a set of satellites.

    Attributes
    ----------
    satellites : tuple of int
        Indices (into the :class:`SatelliteSet`) of the faulted measurements.
    bias : tuple of float
        Bias [m] added to each faulted measurement (same length as ``satellites``).
    label : str
        Free text.
    """

    satellites: Tuple[int, ...]
    bias: Tuple[float, ...]
    label: str = ""

    def __post_init__(self):
        if len(self.satellites) != len(self.bias):
            raise ValueError("satellites and bias must have the same length")


def satellite_fault(sat: int, bias: float) -> FaultSpec:
    """Constant bias on one satellite."""
    return FaultSpec((int(sat),), (float(bias),), f"sat{sat}:{bias:+.1f}m")


def constellation_fault(sats: SatelliteSet, constellation: str | int, bias: float,
                        spread: float = 0.0, rng: np.random.Generator | int | None = None) -> FaultSpec:
    """Fault on *all* satellites of a constellation: common bias plus optional random spread [m]."""
    j = sats.const_names.index(constellation) if isinstance(constellation, str) else int(constellation)
    idx = np.where(sats.constellation == j)[0]
    rng = np.random.default_rng(rng)
    b = bias + spread * rng.normal(size=len(idx))
    return FaultSpec(tuple(int(i) for i in idx), tuple(float(x) for x in b),
                     f"{sats.const_names[j]}-const:{bias:+.1f}m")


def ramp_fault(sat: int, rate: float, t: float) -> FaultSpec:
    """Ramp ``rate * t`` [m] on one satellite at time ``t`` [s] (rate in m/s)."""
    return satellite_fault(sat, rate * t)


def apply_faults(sats: SatelliteSet, y: np.ndarray, faults: Sequence[FaultSpec]):
    """Add the faults to ``y`` and return ``(y_faulty, boolean mask of faulted measurements)``."""
    y = np.array(y, float, copy=True)
    mask = np.zeros(sats.n, bool)
    for f in faults:
        for i, b in zip(f.satellites, f.bias):
            y[i] += b
            mask[i] = True
    return y, mask


def critical_fault(sats: SatelliteSet, faulted: Sequence[int], error: float, axis: int = 2) -> FaultSpec:
    """Fault on ``faulted`` that produces a noise-free all-in-view position error ``error`` on ``axis``
    with the smallest possible solution-separation statistic.

    With ``S0`` the all-in-view coefficients, the bias is ``f = error * S0[axis, F] / |S0[axis, F]|^2``
    so that ``S0[axis] f = error``. The fault is invisible to the subset that excludes ``F``, hence
    the solution separation equals the position error (``|x^(0) - x^(k)| = error``).
    """
    s, ok = wls_coefficients(sats.G, 1.0 / sats.var_int, np.ones((1, sats.n), bool))
    if not ok[0]:
        raise ValueError("all-in-view solution unobservable")
    idx = np.asarray(list(faulted), int)
    v = s[0, axis, idx]
    f = error * v / np.sum(v**2)
    return FaultSpec(tuple(int(i) for i in idx), tuple(float(x) for x in f),
                     f"critical({','.join(sats.ids[i] for i in idx)}):{error:.1f}m@{'ENU'[axis]}")
