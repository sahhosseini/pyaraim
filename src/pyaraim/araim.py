"""The ARAIM user algorithm: protection levels, EMT and accuracy.

This module is the heart of the package. It implements, for a given
:class:`~pyaraim.geometry.SatelliteSet` (geometry + nominal error model + ISD priors),

1. the selection of the fault modes to monitor (:mod:`pyaraim.faultmodes`);
2. the fault-tolerant subset solutions and their statistics (:mod:`pyaraim.solution`);
3. the solution-separation thresholds (Eqs. 26-28);
4. the protection-level equations (Eqs. 21-25) with the modified tail function;
5. the Effective Monitor Threshold and accuracy figures (TAES Eqs. 27-30);
6. the integrity allocation over the exclusion options (Eqs. 19, 24, 25).

The geometry-only entry point is :meth:`ARAIM.protection_levels`, which gives the *worst-case*
protection levels over all pre-selected exclusion options, i.e. the quantity used in availability
simulations. The measurement-domain entry point (tests, exclusion, PL after exclusion) is
:class:`pyaraim.monitor.ARAIMMonitor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import ARAIMConfig, Criteria
from .faultmodes import FaultModeList, determine_fault_modes, unique_masks
from .geometry import SatelliteSet
from .protection import false_alert_containment, integrity_targets, solve_protection_level
from .solution import SubsetSolutions, dual_fault_bound, subset_solutions, _statistics

__all__ = ["ARAIM", "PLResult", "ReferenceTerms", "exclusion_candidates", "build_terms"]

INF = float("inf")


# --------------------------------------------------------------------------- #
@dataclass
class PLResult:
    """Output of the protection-level computation.

    All levels are in metres; ``inf`` means *not available* (no valid solution / the
    integrity budget is exhausted by non-monitorable modes).
    """

    vpl: float = INF
    hpl: float = INF
    hpl_east: float = INF
    hpl_north: float = INF
    emt: float = INF
    sigma_acc: float = INF
    p_not_monitored: float = 0.0
    n_modes: int = 0
    n_sat: int = 0
    k_fa: Tuple[float, float, float] = (np.nan, np.nan, np.nan)
    worst_candidate: str = "all-in-view"
    per_candidate: Optional[dict] = None

    @property
    def accuracy_95(self) -> float:
        return 1.96 * self.sigma_acc

    def available(self, crit: Criteria) -> bool:
        """``True`` if VPL, HPL, EMT and sigma_acc meet the criteria ``crit``."""
        return bool(self.vpl <= crit.val and self.hpl <= crit.hal
                    and self.emt <= crit.emt and self.sigma_acc <= crit.sigma_acc)

    def as_array(self) -> np.ndarray:
        """``[vpl, hpl, emt, sigma_acc]`` (the layout of the MAAST output matrices)."""
        return np.array([self.vpl, self.hpl, self.emt, self.sigma_acc])


@dataclass
class ReferenceTerms:
    """Terms of the PL equations for one reference ("all-in-view") set."""

    sigma: np.ndarray        # (M, 3)  rows: [reference, modes...]
    offset: np.ndarray       # (M, 3)  b^(k) + T_k  (row 0: b^(0))
    coef: np.ndarray         # (M,)    2 for the reference, p_fault otherwise
    alloc_max: np.ndarray    # (M,)
    p_not_monitored: float
    emt: float
    sigma_acc: float
    n_modes: int
    k_fa: np.ndarray
    thresholds: np.ndarray   # (M, 3) T_k,q (row 0: 0)
    solutions: Optional[SubsetSolutions] = None
    s_rows: Optional[np.ndarray] = None      # (M, 3, n_ref) position coefficients aligned with sigma rows
    ref_mask: Optional[np.ndarray] = None
    mode_masks: Optional[np.ndarray] = None  # (M, n_ref) subset masks aligned with sigma rows
    labels: Optional[List[str]] = None
    p_inst: Optional[np.ndarray] = None
    valid: bool = True

    def solve(self, cfg: ARAIMConfig, rho: float = 1.0, pl_target: Optional[Tuple[float, float, float]] = None):
        """Solve Eqs. (21), (23): returns ``(vpl, hpl_east, hpl_north)``."""
        if not self.valid:
            return INF, INF, INF
        t_v, t_h = integrity_targets(cfg, rho, self.p_not_monitored)
        out = []
        for q, tgt in enumerate((t_h, t_h, t_v)):
            pt = None if pl_target is None else pl_target[q]
            out.append(solve_protection_level(self.sigma[:, q], self.offset[:, q], self.coef, tgt,
                                              self.alloc_max, cfg.pl_tol, pt))
        hpl_e, hpl_n, vpl = out
        return vpl, hpl_e, hpl_n


# --------------------------------------------------------------------------- #
def exclusion_candidates(sats: SatelliteSet, cfg: ARAIMConfig) -> List[Tuple[str, np.ndarray]]:
    """Pre-selected exclusion options ``J`` (without the all-in-view).

    All single-satellite exclusions, plus the exclusion of every constellation whose prior
    ``P_const`` is at least ``cfg.exclusion_const_threshold`` (Section "Integrity allocation
    across exclusion options"). Each candidate is ``(label, mask_of_retained_satellites)``.
    """
    if not cfg.exclusion:
        return []
    out = []
    for i in range(sats.n):
        m = np.ones(sats.n, bool)
        m[i] = False
        out.append((sats.ids[i], m))
    if cfg.exclusion_const_threshold is not None:
        for j in range(sats.n_const):
            if sats.p_const[j] >= cfg.exclusion_const_threshold and sats.n_const > 1:
                out.append((f"{sats.const_names[j]}-const", sats.constellation != j))
    return out


# --------------------------------------------------------------------------- #
def _reduce(sats: SatelliteSet, ref: np.ndarray):
    """Geometry matrix restricted to the reference satellites, dropping empty clock columns."""
    G = sats.G[ref]
    keep = [0, 1, 2] + [3 + j for j in range(sats.n_const) if (G[:, 3 + j] != 0).any()]
    return G[:, keep]


def _adjust_position(sol: SubsetSolutions, p_inst: np.ndarray, var_int, var_acc, b_int, b_acc,
                     cfg: ARAIMConfig) -> SubsetSolutions:
    """Integrity-optimised position estimator (Blanch et al., TAES 51(3), 2015).

    For each coordinate, the reference coefficients are moved a fraction ``t`` towards those of the
    weakest monitored subset, as far as the accuracy constraint ``sigma_acc <= sig_acc_max`` allows
    (port of ``compute_adjusted_position_1D.m``).
    """
    s = sol.s.copy()
    limits = (cfg.sig_acc_max_hor1, cfg.sig_acc_max_hor2, cfg.sig_acc_max_vert)
    cand = np.where((p_inst >= 1e-11) & np.isfinite(sol.sigma[:, 0]) & np.isfinite(sol.sigma[:, 1])
                    & np.isfinite(sol.sigma[:, 2]))[0]
    for q in range(3):
        if cand.size == 0:
            break
        weak = cand[np.argmax(sol.sigma[cand, q])]
        s0 = s[0, q]
        dsw = s[weak, q] - s0
        a = float(np.sum(dsw**2 * var_acc))
        b = float(np.sum(s0 * dsw * var_acc))
        c = float(np.sum(s0**2 * var_acc)) - (limits[q] - 1e-5) ** 2
        if c < 0 and a > 0:
            t = min((-b + np.sqrt(max(b * b - a * c, 0.0))) / a, 1.0)
            s[0, q] = s0 + t * dsw
    return _statistics(s, sol.observable, sol.masks, var_int, var_acc, b_int, b_acc)


def build_terms(
    sats: SatelliteSet,
    ref: np.ndarray,
    modes: FaultModeList,
    cfg: ARAIMConfig,
    p_not_monitored_in: Optional[float] = None,
    adjust: bool = False,
    keep_solutions: bool = False,
    use_bound: Optional[bool] = None,
) -> ReferenceTerms:
    """Assemble the PL-equation terms for the reference set ``ref`` and the fault ``modes``.

    The fault modes are intersected with the reference set (Eq. 17) and identical subsets are
    merged (Eq. 18); modes that become identical to the reference are dropped.
    """
    ref = np.asarray(ref, bool)
    nref = int(ref.sum())
    G = _reduce(sats, ref)
    var_int, var_acc = sats.var_int[ref], sats.var_acc[ref]
    b_int, b_acc = sats.b_int[ref], sats.b_acc[ref]
    n_unk = G.shape[1]

    p_nm = modes.p_not_monitored if p_not_monitored_in is None else p_not_monitored_in
    bad = ReferenceTerms(np.full((1, 3), INF), np.zeros((1, 3)), np.array([2.0]), np.ones(1),
                         p_nm, INF, INF, 0, np.full(3, np.nan), np.zeros((1, 3)), valid=False)
    if nref < n_unk + 1 and nref < 4:
        return bad

    # ---- intersect modes with the reference, merge duplicates (Eq. 17-18) ----
    if len(modes):
        mm = modes.masks[:, ref]
        different = ~mm.all(axis=1)
        mm, pi, pe = mm[different], modes.p_inst[different], modes.p_exp[different]
        labels = [l for l, d in zip(modes.labels, different) if d]
        if len(mm):
            mm, pi, pe, first, _ = unique_masks(mm, pi, pe)
            labels = [labels[i] for i in first]
    else:
        mm = np.zeros((0, nref), bool)
        pi = pe = np.zeros(0)
        labels = []
    masks = np.vstack([np.ones((1, nref), bool), mm])
    p_inst = np.concatenate([[1.0], pi])
    p_exp = np.concatenate([[1.0], pe])
    labels = ["reference"] + labels

    # ---- solutions ---------------------------------------------------------------
    use_bound = (cfg.fast if use_bound is None else use_bound) and not adjust and len(mm) > 0
    removed = (~masks).sum(axis=1)
    two_out = (removed == 2) & (np.arange(len(masks)) > 0) if use_bound else np.zeros(len(masks), bool)
    bound = None
    if two_out.any():
        pair_idx = np.array([np.where(~m)[0] for m in masks[two_out]])
        bound = dual_fault_bound(G, var_int, var_acc, b_int, b_acc, pair_idx, cfg.cond_limit)
        if bound is None:
            two_out[:] = False
        else:
            cover = np.zeros(len(masks), bool)
            cover[np.where(two_out)[0][bound.valid]] = True
            two_out = cover
    exact = ~two_out
    sol = subset_solutions(G, var_int, var_acc, b_int, b_acc, masks[exact], cfg.cond_limit)
    if not sol.observable[0]:
        return bad
    if adjust:
        sol = _adjust_position(sol, p_inst[exact], var_int, var_acc, b_int, b_acc, cfg)

    sigma = sol.sigma.copy()
    sigma_ss = sol.sigma_ss.copy()
    bias = sol.bias.copy()
    bias_ss = sol.bias_ss.copy()
    pi_e, pe_e = p_inst[exact], p_exp[exact]
    nmodes_row = np.ones(len(sigma))
    lab_e = [l for l, e in zip(labels, exact) if e]
    if bound is not None and two_out.any():
        sigma = np.vstack([sigma, bound.sigma])
        sigma_ss = np.vstack([sigma_ss, bound.sigma_ss])
        bias = np.vstack([bias, bound.bias])
        bias_ss = np.vstack([bias_ss, bound.bias_ss])
        pi_e = np.append(pi_e, p_inst[two_out].sum())
        pe_e = np.append(pe_e, p_exp[two_out].sum())
        nmodes_row = np.append(nmodes_row, two_out.sum())
        lab_e = lab_e + ["dual-fault bound"]

    sigma_acc = float(np.sqrt(np.sum(sol.s[0, 2] ** 2 * var_acc)))

    # ---- unobservable modes ---------------------------------------------------
    finite = np.isfinite(sigma).all(axis=1)
    finite[0] = True
    alloc_cap = np.ones(len(sigma))
    if cfg.cap_by_exposure:
        with np.errstate(divide="ignore", invalid="ignore"):
            alloc_cap = np.minimum(pe_e / pi_e / cfg.n_es_int, 1.0)
        alloc_cap[0] = 1.0
    unobs = ~finite
    if unobs.any():
        # Eq. (2): a mode that cannot be monitored is charged its full exposure probability
        p_nm = p_nm + float(pe_e[unobs].sum())
    keep = finite
    sigma, sigma_ss, bias, bias_ss = sigma[keep], sigma_ss[keep], bias[keep], bias_ss[keep]
    pi_e, pe_e, nmodes_row, alloc_cap = pi_e[keep], pe_e[keep], nmodes_row[keep], alloc_cap[keep]
    lab_e = [l for l, k in zip(lab_e, keep) if k]

    n_modes = int(nmodes_row[1:].sum())
    k_fa = false_alert_containment(n_modes, cfg, nref, n_unk - 3)
    T = k_fa[None, :] * sigma_ss + bias_ss + np.finfo(float).eps
    T[0] = 0.0

    coef = pi_e.copy()
    coef[0] = 2.0
    offset = bias + T
    emt = float(np.max(T[1:, 2][pi_e[1:] >= cfg.p_emt])) if (pi_e[1:] >= cfg.p_emt).any() else 0.0
    out = ReferenceTerms(sigma, offset, coef, alloc_cap, p_nm, emt, sigma_acc, n_modes,
                         k_fa, T, None, None, ref, None, lab_e, pi_e)
    if keep_solutions:
        if bound is not None and two_out.any():
            raise ValueError("keep_solutions requires cfg.fast=False")
        out.s_rows = sol.s[keep]
        out.mode_masks = masks[exact][keep]
        out.solutions = sol
    return out


# --------------------------------------------------------------------------- #
class ARAIM:
    """ARAIM user algorithm (geometry-domain figures of merit).

    Parameters
    ----------
    cfg:
        Algorithm parameters (default: :meth:`ARAIMConfig.lpv200`).

    Examples
    --------
    >>> from pyaraim import ARAIM, ARAIMConfig
    >>> araim = ARAIM(ARAIMConfig.lpv200())            # doctest: +SKIP
    >>> res = araim.protection_levels(sats)            # doctest: +SKIP
    >>> res.vpl, res.hpl, res.emt, res.sigma_acc       # doctest: +SKIP
    """

    def __init__(self, cfg: ARAIMConfig | None = None):
        self.cfg = cfg or ARAIMConfig.lpv200()

    # ----------------------------------------------------------------- helpers
    def fault_modes(self, sats: SatelliteSet) -> FaultModeList:
        """Fault modes monitored for this geometry (Section "Determination of fault modes")."""
        return determine_fault_modes(sats.pruned(), self.cfg) if sats.n_const else FaultModeList(
            np.zeros((0, 0), bool), np.zeros(0), np.zeros(0))

    def _fd_only(self, sats, modes, adjust=False, keep_solutions=False, rho=1.0) -> Tuple[ReferenceTerms, tuple]:
        terms = build_terms(sats, np.ones(sats.n, bool), modes, self.cfg, adjust=adjust,
                            keep_solutions=keep_solutions)
        return terms, terms.solve(self.cfg, rho)

    # ------------------------------------------------------------ main entry
    def protection_levels(self, sats: SatelliteSet, criteria: Criteria | None = None,
                          detail: bool = False) -> PLResult:
        """Worst-case VPL/HPL/EMT/accuracy over the all-in-view and every exclusion option.

        With ``cfg.exclusion = False`` this is the fault-detection-only result.
        With ``cfg.fast`` and ``criteria`` given, a conservative bound on all dual-fault subsets is
        tried first (Eqs. 36-41); the exact levels are only computed if the bound does not already
        meet ``criteria``. Without ``criteria`` and with ``cfg.fast`` the bound-based levels are
        returned (valid but conservative). If ``cfg.optimize_position`` is set, the optimised
        estimator is also evaluated and used when it lowers the VPL or HPL.
        """
        cfg = self.cfg
        sats = sats.pruned() if sats.n else sats
        if sats.n < 4:
            return PLResult(n_sat=sats.n)
        if cfg.fast and criteria is not None and not detail:
            # screening: the dual-fault bound is conservative, so "available" is certain;
            # otherwise recompute exactly (as in mhss_raim_baseline_v5.m)
            res = self._evaluate(sats, adjust=False, detail=detail, use_bound=True)
            if res.available(criteria):
                return res
        res = self._evaluate(sats, adjust=False, detail=detail,
                             use_bound=(True if cfg.fast and criteria is None else False))
        if cfg.optimize_position and res.sigma_acc < cfg.sig_acc_max_vert and \
                (criteria is None or not res.available(criteria)):
            res_opt = self._evaluate(sats, adjust=True, detail=detail, use_bound=False)
            if res_opt.vpl < res.vpl or res_opt.hpl < res.hpl:
                res = res_opt
        return res

    def _evaluate(self, sats: SatelliteSet, adjust: bool, detail: bool, use_bound: bool = False) -> PLResult:
        cfg = self.cfg
        modes = self.fault_modes(sats)
        cands = exclusion_candidates(sats, cfg)
        n_exc = len(cands)
        rho = 1.0 / (n_exc + 1)
        joint = cfg.exclusion and cfg.exclusion_allocation == "joint" and n_exc > 0

        refs = [("all-in-view", np.ones(sats.n, bool))] + cands
        terms_list: List[ReferenceTerms] = []
        for k, (label, ref) in enumerate(refs):
            t = build_terms(sats, ref, modes, cfg, p_not_monitored_in=modes.p_not_monitored,
                            adjust=adjust, use_bound=use_bound)
            terms_list.append(t)

        t0 = terms_list[0]
        if joint:
            return self._joint(sats, refs, terms_list, modes)
        per = {}
        vpl = hpl = emt = sacc = 0.0
        he = hn = 0.0
        worst = "all-in-view"
        for (label, _), t in zip(refs, terms_list):
            v, e, n = t.solve(cfg, rho)
            h = float(np.hypot(e, n))
            per[label] = (v, h)
            if v > vpl:
                worst = label
            vpl, hpl = max(vpl, v), max(hpl, h)
            he, hn = max(he, e), max(hn, n)
            emt = max(emt, t.emt)
            sacc = max(sacc, t.sigma_acc)
        return PLResult(vpl, hpl, he, hn, emt, sacc, t0.p_not_monitored, t0.n_modes, sats.n,
                        tuple(t0.k_fa), worst, per if detail else None)

    def _joint(self, sats, refs, terms_list, modes) -> PLResult:
        """Eq. (25): a single PL such that the summed risk over all exclusion options meets the budget."""
        cfg = self.cfg
        t0 = terms_list[0]
        if any(not t.valid for t in terms_list):
            return PLResult(p_not_monitored=t0.p_not_monitored, n_sat=sats.n)
        sig = np.vstack([t.sigma for t in terms_list])
        off = np.vstack([t.offset for t in terms_list])
        coef = np.concatenate([t.coef for t in terms_list])
        cap = np.concatenate([t.alloc_max for t in terms_list])
        # the unmonitored mass is the worst over the options (conservative common allocation)
        p_nm = max(t.p_not_monitored for t in terms_list)
        t_v, t_h = integrity_targets(cfg, 1.0, p_nm)
        out = []
        for q, tgt in enumerate((t_h, t_h, t_v)):
            out.append(solve_protection_level(sig[:, q], off[:, q], coef, tgt, cap, cfg.pl_tol))
        he, hn, v = out
        return PLResult(v, float(np.hypot(he, hn)), he, hn, max(t.emt for t in terms_list),
                        max(t.sigma_acc for t in terms_list), t0.p_not_monitored, t0.n_modes, sats.n,
                        tuple(t0.k_fa), "joint")
