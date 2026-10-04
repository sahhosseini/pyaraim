"""Nominal pseudorange error models (the diagonal of ``C_int`` and ``C_acc``).

For satellite *i*, Eq. (2) of the TAES paper gives::

    C_int(i,i) = sigma_URA,i^2 + sigma_tropo,i^2 + sigma_user,i^2
    C_acc(i,i) = sigma_URE,i^2 + sigma_tropo,i^2 + sigma_user,i^2

``sigma_tropo`` is common to all models (Eq. 58 of the TAES paper). The user
term (airborne multipath + noise, plus residual ionosphere of the dual-frequency
combination) is selectable:

* :class:`ED259A` - the model implemented in MAAST (DFMC SBAS MOPS ED-259A);
* :class:`BaselineAppendixA` - Appendix A of the TAES paper (GPS "Model A" and the
  tabulated Galileo values).

All elevation inputs are in **radians**.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .constants import DEG, GAMMA_L1L5_NOISE
from .isd import ISD

__all__ = [
    "sigma_tropo",
    "UserErrorModel",
    "ED259A",
    "BaselineAppendixA",
    "nominal_error_terms",
]


def sigma_tropo(el):
    """Tropospheric residual std-dev [m], Eq. (58): ``0.12 * 1.001 / sqrt(0.002001 + sin^2(el))``."""
    el = np.asarray(el, dtype=float)
    return 0.12 * 1.001 / np.sqrt(0.002001 + np.sin(el) ** 2)


class UserErrorModel(Protocol):
    """Interface of the user-error term (variance [m^2] of the iono-free pseudorange)."""

    def variance(self, el: np.ndarray, constellation: str) -> np.ndarray:  # pragma: no cover
        ...


@dataclass(frozen=True)
class ED259A:
    """L1/L5 iono-free user error as coded in ``usrprocess_araim.m`` / ``af_cnmp_259a.m``.

    ``var = 0.4^2 + (0.34 + 0.40 exp(-el/14 deg))^2 + sigma_uire^2`` with the
    residual ionospheric error ``sigma_uire = 40/(261 + el_deg^2) + 0.018`` [m].
    The same model is used for every constellation (as in MAAST).
    """

    def variance(self, el, constellation: str = "GPS"):
        el = np.asarray(el, dtype=float)
        eld = el / DEG
        cnmp = 0.4**2 + (0.34 + 0.40 * np.exp(-eld / 14.0)) ** 2
        sig_uire = 40.0 / (261.0 + eld**2) + 0.018
        return cnmp + sig_uire**2


# Table A.I of the TAES paper: Galileo elevation-dependent user error [m]
_GAL_EL = np.arange(5.0, 95.0, 5.0)
_GAL_SIG = np.array([0.4529, 0.3553, 0.3063, 0.2638, 0.2593, 0.2555, 0.2504, 0.2438, 0.2396,
                     0.2359, 0.2339, 0.2302, 0.2295, 0.2278, 0.2297, 0.2310, 0.2274, 0.2277])


@dataclass(frozen=True)
class BaselineAppendixA:
    """Appendix A of Blanch et al. (TAES 2015).

    GPS (Eq. 57): ``sigma = sqrt((f1^4+f5^4)/(f1^2-f5^2)^2) * sqrt(sigma_MP^2 + sigma_noise^2)`` with
    ``sigma_MP = 0.13 + 0.53 exp(-el/10)`` and ``sigma_noise = 0.15 + 0.43 exp(-el/6.9)``
    (``el`` in degrees). Galileo: Table A.I, linearly interpolated.
    Other constellations use the GPS model.
    """

    def variance(self, el, constellation: str = "GPS"):
        el = np.asarray(el, dtype=float)
        eld = el / DEG
        if constellation.lower().startswith("gal"):
            return np.interp(eld, _GAL_EL, _GAL_SIG) ** 2
        s_mp = 0.13 + 0.53 * np.exp(-eld / 10.0)
        s_no = 0.15 + 0.43 * np.exp(-eld / 6.9)
        return GAMMA_L1L5_NOISE * (s_mp**2 + s_no**2)


def nominal_error_terms(
    isd: ISD,
    constellation: Sequence[str],
    el: np.ndarray,
    model: UserErrorModel | None = None,
    sigma_ura: np.ndarray | None = None,
    sigma_ure: np.ndarray | None = None,
):
    """Assemble the per-satellite nominal-error terms.

    Parameters
    ----------
    isd:
        Integrity support data.
    constellation:
        Constellation name of every satellite (length *n*).
    el:
        Elevation angles [rad] (length *n*).
    model:
        User-error model (default :class:`ED259A`).
    sigma_ura, sigma_ure:
        Optional per-satellite overrides (e.g. a broadcast URA); ``NaN`` entries
        fall back to the constellation ISD value.

    Returns
    -------
    var_int, var_acc, b_int, b_acc : ndarray
        ``C_int`` and ``C_acc`` diagonals [m^2] and the nominal biases [m].
    """
    model = model or ED259A()
    names = np.asarray(list(constellation))
    n = len(names)
    el = np.asarray(el, dtype=float)
    ura = np.empty(n)
    ure = np.empty(n)
    b_int = np.empty(n)
    b_acc = np.empty(n)
    var_user = np.empty(n)
    for cname in dict.fromkeys(names.tolist()):
        sel = names == cname
        c = isd[cname]
        ura[sel] = c.sigma_ura
        ure[sel] = c.sigma_ure
        b_int[sel] = c.b_nom
        b_acc[sel] = c.b_nom_cont
        var_user[sel] = model.variance(el[sel], cname)
    if sigma_ura is not None:
        s = np.asarray(sigma_ura, float)
        ura = np.where(np.isnan(s), ura, s)
    if sigma_ure is not None:
        s = np.asarray(sigma_ure, float)
        ure = np.where(np.isnan(s), ure, s)
    common = sigma_tropo(el) ** 2 + var_user
    return ura**2 + common, ure**2 + common, b_int, b_acc
