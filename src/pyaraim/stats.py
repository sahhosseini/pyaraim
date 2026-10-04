"""Gaussian tail functions used by the ARAIM protection-level equations.

Notation follows Blanch et al. (ION ITM 2022), Eqs. (29)-(30)::

    Q(u)    = 1/sqrt(2 pi) * int_u^inf exp(-t^2/2) dt
    Qbar(u) = Q(u) for u > 0, and 1 for u <= 0

``Qbar`` guarantees that a *smaller* standard deviation never yields a *larger*
integrity-risk contribution (it makes the PL equation monotone in sigma).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st
from scipy.special import ndtr, ndtri

__all__ = ["Q", "Qinv", "Qbar", "chi2_threshold"]


def Q(u):
    """Upper tail of the standard normal distribution, Eq. (29)."""
    return ndtr(-np.asarray(u, dtype=float))


def Qinv(p):
    """Inverse of :func:`Q`: the ``(1 - p)`` quantile of N(0, 1)."""
    return -ndtri(np.asarray(p, dtype=float))


def Qbar(u):
    """Modified tail function of Eq. (30): ``Q(u)`` for ``u > 0`` else ``1``."""
    u = np.asarray(u, dtype=float)
    return np.where(u > 0.0, ndtr(-u), 1.0)


def chi2_threshold(p_fa: float, dof: int) -> float:
    """Threshold ``T`` such that ``P(chi2_dof > T) = p_fa`` (Eq. (23) of the TAES paper).

    Returns ``inf`` when ``dof <= 0`` (no redundancy: nothing can be tested).
    """
    if dof <= 0:
        return float("inf")
    return float(_st.chi2.isf(p_fa, dof))
