"""Protection-level equations (Eqs. 21-28 of Blanch et al., ION ITM 2022).

For each coordinate ``q`` the protection level is the solution ``PL`` of::

    2 Qbar((PL - b_q^(0))/sigma_q^(0)) + sum_k p_k Qbar((PL - T_k,q - b_q^(k))/sigma_q^(k)) = target

with the integrity risk ``target`` allocated to that coordinate (Eqs. 21 and 23)::

    vertical  : rho_j * PHMI_VERT / N_ES        * (1 - P_nm / (PHMI_VERT + PHMI_HOR))
    horizontal: rho_j * PHMI_HOR / (2 N_ES)     * (1 - P_nm / (PHMI_VERT + PHMI_HOR))   (per axis)

``solve_protection_level`` finds the root by bisection between the closed-form lower and
upper bounds of Appendix B (Eqs. 62-63) of the TAES paper.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import ARAIMConfig
from .stats import Qbar, Qinv, chi2_threshold

__all__ = ["solve_protection_level", "false_alert_containment", "integrity_targets"]


def solve_protection_level(
    sigma: np.ndarray,
    offset: np.ndarray,
    coef: np.ndarray,
    target: float,
    alloc_max: Optional[np.ndarray] = None,
    tol: float = 1e-2,
    pl_target: Optional[float] = None,
    max_iter: int = 80,
) -> float:
    """Solve ``sum_k coef_k * min(Qbar((PL - offset_k)/sigma_k), alloc_max_k) = target``.

    Parameters
    ----------
    sigma : (M,) std-devs ``sigma^(k)`` of the position error of each term (row 0 = fault-free).
    offset : (M,) ``b^(k) + T_k`` (row 0: just ``b^(0)``).
    coef : (M,) weights of the terms: ``2`` for the fault-free term, ``p_fault,k`` otherwise.
    target : integrity risk allocated to this coordinate.
    alloc_max : optional (M,) cap on the tail probability of every term (the refinement of
        Blanch & Walter, ION GNSS+ 2023); ``None`` for no cap.
    tol : bracket width at which the search stops [m]; the **upper** end is returned, so the
        result never underestimates the exact root.
    pl_target : if given, the search stops as soon as the upper bracket is below it
        (the user is known to be available; the returned value is then an upper bound).

    Returns ``inf`` if ``target <= 0``.
    """
    if not target > 0.0:
        return float("inf")
    sigma = np.asarray(sigma, float)
    offset = np.asarray(offset, float)
    coef = np.asarray(coef, float)
    if alloc_max is None:
        alloc_max = np.ones_like(sigma)
    m = len(sigma)

    # lower bracket: every single term must stay below the target on its own (Appendix B, Eq. 62)
    ratio = target / coef
    binding = ratio < np.minimum(alloc_max, 1.0)
    with np.errstate(invalid="ignore"):
        lo_k = np.where(binding, offset + Qinv(np.where(binding, ratio, 0.5)) * sigma, -np.inf)
    lo = float(np.max(lo_k))
    lo = max(lo, 0.0) if np.isfinite(lo) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        k_hi = np.maximum(0.0, Qinv(np.minimum(1.0, target / (m * coef))))
    hi = float(np.max(offset + k_hi * sigma))
    if hi < lo:
        hi = lo

    def total(pl: float) -> float:
        return float(np.sum(coef * np.minimum(Qbar((pl - offset) / sigma), alloc_max)))

    if total(lo) <= target:          # already satisfied at the lower bound
        return float(lo)
    it = 0
    while hi - lo > tol and it < max_iter:
        if pl_target is not None and hi <= pl_target:
            break
        mid = 0.5 * (lo + hi)
        if total(mid) > target:
            lo = mid
        else:
            hi = mid
        it += 1
    return float(hi)


def false_alert_containment(n_modes: int, cfg: ARAIMConfig, n_sat_ref: int, n_clock_ref: int):
    """Containment factors ``K_fa,q`` of the solution-separation thresholds (Eq. 28).

    ``K_fa,1 = K_fa,2 = Qinv(P_FA_HOR / (4 N_modes N_ES,CONT))`` and
    ``K_fa,3 = Qinv(P_FA_VERT / (2 N_modes N_ES,CONT))``.
    With ``threshold_method="chi2"`` a common factor ``sqrt(T_chi2)`` is returned, where
    ``T_chi2`` is the chi-square threshold for ``P_FA / N_ES,CONT`` and ``n - 3 - N_const``
    degrees of freedom (Eqs. 36, 55 of the TAES paper).

    Returns an array of three factors (East, North, Up).
    """
    if cfg.threshold_method == "chi2":
        dof = n_sat_ref - 3 - n_clock_ref
        k = np.sqrt(chi2_threshold(cfg.pfa / cfg.n_es_cont, dof))
        return np.full(3, k)
    n_modes = max(int(n_modes), 1)
    k_h = float(Qinv(cfg.pfa_hor / (4.0 * n_modes * cfg.n_es_cont)))
    k_v = float(Qinv(cfg.pfa_vert / (2.0 * n_modes * cfg.n_es_cont)))
    return np.array([k_h, k_h, k_v])


def integrity_targets(cfg: ARAIMConfig, rho: float, p_not_monitored: float):
    """Right-hand sides of Eqs. (21) and (23).

    Returns ``(target_vertical, target_horizontal_per_axis)``; both are ``0`` if the
    probability of the non-monitored modes exhausts the integrity budget.
    """
    factor = 1.0 - p_not_monitored / (cfg.phmi_vert + cfg.phmi_hor)
    if factor <= 0.0:
        return 0.0, 0.0
    t_v = rho * cfg.phmi_vert * factor / cfg.n_es_int
    t_h = rho * 0.5 * cfg.phmi_hor * factor / cfg.n_es_int
    return t_v, t_h
