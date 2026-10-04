"""Measurement-domain ARAIM: consistency tests, exclusion and protection levels after exclusion.

:class:`ARAIMMonitor` processes one epoch of pseudorange residuals ``y`` (measurement minus
expected range at the linearisation point, after dual-frequency, tropospheric and smoothing
corrections) following Sections "Exclusion function" and "Protection level equation" of
Blanch et al. (ION ITM 2022):

1. the all-in-view solution and the solution-separation tests ``|x^(k) - x^(0)| <= T_k`` of every
   monitored fault mode (Eqs. 18, 21 of the TAES paper);
2. if every test passes, the protection levels are those of the all-in-view set, with the
   integrity allocation ``rho_0``;
3. otherwise each pre-selected exclusion candidate ``j`` is examined (ordered by number of
   excluded satellites and then by increasing chi-square statistic, Eq. 31 of the TAES paper; the
   2022 update allows any consistent candidate of ``J`` to be chosen).
   The first candidate whose own solution-separation tests pass is selected, and the protection
   levels are those of the reduced set with allocation ``rho_j`` (Eqs. 21-24);
4. if no candidate is consistent, an alert is raised.

Because the integrity allocation ``rho_j`` does not depend on the measurements, the protection
levels returned after exclusion are (by construction) valid whichever candidate was selected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .araim import ARAIM, ReferenceTerms, build_terms, exclusion_candidates
from .config import ARAIMConfig
from .geometry import SatelliteSet
from .solution import weighted_ls
from .stats import chi2_threshold

__all__ = ["ARAIMMonitor", "MonitorResult"]


@dataclass
class MonitorResult:
    """Outcome of processing one epoch."""

    status: str                         # "ok", "excluded", "alert"
    position: np.ndarray                # (3,) E, N, U estimate relative to the linearisation point
    excluded: Optional[str] = None
    vpl: float = np.inf
    hpl: float = np.inf
    emt: float = np.inf
    sigma_acc: float = np.inf
    tau_max: float = 0.0                # max_k,q |dx|/T for the final reference
    chi2: float = np.nan
    chi2_threshold: float = np.nan
    chi2_ok: bool = True
    n_candidates_tried: int = 0
    reason: str = ""

    @property
    def alert(self) -> bool:
        return self.status == "alert"


class ARAIMMonitor:
    """ARAIM fault detection and exclusion on pseudorange measurements."""

    def __init__(self, cfg: ARAIMConfig | None = None):
        cfg = cfg or ARAIMConfig.lpv200()
        # the position solutions of each subset are needed explicitly
        self.cfg = cfg.replace(fast=False, optimize_position=False)
        self._araim = ARAIM(self.cfg)

    # ------------------------------------------------------------------ pieces
    @staticmethod
    def _chi2(sats: SatelliteSet, ref: np.ndarray, y: np.ndarray):
        """Chi-square statistic with ``C_acc`` of the reference set (Eq. 22 / 32 of the TAES paper)."""
        _, chi2, dof = weighted_ls(sats.G, 1.0 / sats.var_acc, ref, y)
        return chi2, dof

    def _tests(self, terms: ReferenceTerms, y: np.ndarray, sats: SatelliteSet):
        """Test ratios ``tau_k,q`` (M-1, 3) of every monitored mode against the reference."""
        yr = y[terms.ref_mask]
        dx = (terms.s_rows[1:] - terms.s_rows[0:1]) @ yr                  # (M-1, 3)
        return np.abs(dx) / terms.thresholds[1:]

    def _passes(self, terms: ReferenceTerms, y, sats):
        cfg = self.cfg
        if cfg.threshold_method == "chi2":
            chi2, dof = self._chi2(sats, terms.ref_mask, y)
            thr = chi2_threshold(cfg.pfa / cfg.n_es_cont, dof)
            return chi2 <= thr, (chi2 / thr if np.isfinite(thr) else 0.0)
        tau = self._tests(terms, y, sats)
        tmax = float(tau.max()) if tau.size else 0.0
        return tmax <= 1.0, tmax

    # --------------------------------------------------------------------- main
    def process(self, sats: SatelliteSet, y: np.ndarray) -> MonitorResult:
        """Run detection, exclusion and PL computation for the residual vector ``y``."""
        cfg = self.cfg
        y = np.asarray(y, float)
        if y.shape != (sats.n,):
            raise ValueError("y must have one entry per satellite")
        sats = sats.pruned()
        modes = self._araim.fault_modes(sats)
        cands = exclusion_candidates(sats, cfg)
        rho = 1.0 / (len(cands) + 1)

        all_in = np.ones(sats.n, bool)
        t0 = build_terms(sats, all_in, modes, cfg, modes.p_not_monitored, False, True)
        if not t0.valid:
            return MonitorResult("alert", np.full(3, np.nan), reason="all-in-view solution unobservable")

        chi2, dof = self._chi2(sats, all_in, y)
        chi2_thr = chi2_threshold(cfg.pfa_chi2, dof)
        ok0, tau0 = self._passes(t0, y, sats)
        x0 = t0.s_rows[0] @ y

        if ok0:
            if cfg.chi2_sanity_check and chi2 > chi2_thr:
                return MonitorResult("alert", x0, chi2=chi2, chi2_threshold=chi2_thr, chi2_ok=False,
                                     tau_max=tau0, reason="chi-square sanity check failed")
            v, he, hn = t0.solve(cfg, rho)
            return self._result("ok", x0, None, v, he, hn, t0, tau0, chi2, chi2_thr, 0)

        # ---- exclusion ------------------------------------------------------------------
        # 2022 update: any pre-selected candidate j in J that passes the consistency checks may be
        # chosen, because the integrity allocation rho_j does not depend on the choice.
        order = []
        for label, ref in cands:
            c2, _ = self._chi2(sats, ref, y)
            order.append((int((~ref).sum()), c2, label, ref))
        order.sort(key=lambda t: (t[0], t[1]))
        tried = 0
        for _, _, label, ref in order:
            tried += 1
            t = build_terms(sats, ref, modes, cfg, modes.p_not_monitored, False, True)
            if not t.valid:
                continue
            ok, tau_j = self._passes(t, y, sats)
            if not ok:
                continue
            x = t.s_rows[0] @ y[ref]
            v, he, hn = t.solve(cfg, rho)
            c2, _ = self._chi2(sats, ref, y)
            return self._result("excluded", x, label, v, he, hn, t, tau_j, c2, chi2_thr, tried)
        return MonitorResult("alert", x0, chi2=chi2, chi2_threshold=chi2_thr, tau_max=tau0,
                             n_candidates_tried=tried, reason="no consistent exclusion candidate")

    @staticmethod
    def _result(status, x, label, v, he, hn, terms, tau, chi2, chi2_thr, tried) -> MonitorResult:
        return MonitorResult(status, np.asarray(x[:3]), label, v, float(np.hypot(he, hn)), terms.emt,
                             terms.sigma_acc, tau, chi2, chi2_thr, chi2 <= chi2_thr, tried)
