"""Monte-Carlo tools to *verify* an ARAIM implementation against its own claims.

The protection-level equation is built on a per-mode probability inequality. For a fault ``k`` that
affects only the measurements outside the subset ``idx_k`` and any level ``L``::

    P(  x_hat^(0)_q - x_q  > L  and  |x_hat^(k)_q - x_hat^(0)_q| <= T )  <=  Qbar((L - T - b^(k)_q) / sigma^(k)_q)   (*)

(one-sided, which is the form of the terms ``p_fault,k * Qbar(.)`` of the PL equation; the
two-sided event is bounded by twice the right-hand side).

(Appendix K of the TAES paper, with the position error split as ``x^(k) - x`` plus the separation).
:func:`verify_mode_bound` checks (*) empirically for a chosen mode and fault magnitude, and
:func:`false_alert_rate`, :func:`detection_exclusion_stats` exercise the monitor end to end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats as st

from ..araim import ARAIM, build_terms
from ..config import ARAIMConfig
from ..geometry import SatelliteSet
from ..monitor import ARAIMMonitor
from ..stats import Qbar
from .faults import FaultSpec, critical_fault
from .scenarios import make_epoch

__all__ = ["BoundCheck", "verify_mode_bound", "false_alert_rate", "DetectionStats",
           "detection_exclusion_stats", "clopper_pearson_upper"]


def clopper_pearson_upper(k: int, n: int, conf: float = 0.999) -> float:
    """One-sided upper confidence bound of a binomial proportion."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(st.beta.ppf(conf, k + 1, n - k))


@dataclass
class BoundCheck:
    """Result of :func:`verify_mode_bound`."""

    mode: str
    axis: int
    level: float
    threshold: float
    fault_bias: float
    n_trials: int
    n_hits: int
    empirical: float
    empirical_upper: float
    bound: float

    @property
    def empirical_lower(self) -> float:
        """One-sided 99.9 % lower confidence bound of the empirical probability."""
        if self.n_hits == 0:
            return 0.0
        return float(st.beta.ppf(0.001, self.n_hits, self.n_trials - self.n_hits + 1))

    @property
    def holds(self) -> bool:
        """``True`` unless the data *contradict* the analytic bound with 99.9 % confidence."""
        return not (self.empirical_lower > self.bound)


def verify_mode_bound(sats: SatelliteSet, cfg: ARAIMConfig, mode: int = 0, axis: int = 2,
                      level: Optional[float] = None, fault_bias: float = 0.0, n_trials: int = 20000,
                      rng: np.random.Generator | int | None = None, two_sided: bool = False) -> BoundCheck:
    """Empirically verify inequality (*) for fault mode ``mode`` (index into the monitored list).

    The fault is the critical one of the mode (``fault_bias`` is the noise-free position error it
    produces on ``axis``), noise is drawn from the integrity covariance ``C_int`` with worst-case
    nominal biases (random signs would be weaker; the bound uses ``b^(k)``, so zero nominal bias is
    simulated unless ``sats.b_int`` is non-zero, in which case the worst-sign bias is applied).
    ``level`` defaults to ``2 sigma^(k) + T``.
    """
    rng = np.random.default_rng(rng)
    modes = ARAIM(cfg).fault_modes(sats)
    terms = build_terms(sats, np.ones(sats.n, bool), modes, cfg.replace(fast=False), keep_solutions=True)
    k = mode + 1
    if k >= len(terms.sigma):
        raise IndexError("mode index out of range")
    s0, sk = terms.s_rows[0, axis], terms.s_rows[k, axis]
    mask = terms.mode_masks[k]
    thr = terms.thresholds[k, axis]
    sig_k = terms.sigma[k, axis]
    bias_k = terms.offset[k, axis] - thr
    L = level if level is not None else thr + bias_k + 2.0 * sig_k
    f = np.zeros(sats.n)
    faulted = np.where(~mask)[0]
    if fault_bias != 0.0:
        spec = critical_fault(sats, faulted, fault_bias, axis)
        f[list(spec.satellites)] = spec.bias
    # nominal worst-case bias: signs aligned with s^(k) (maximises the error of subset k)
    nb = np.sign(sk) * sats.b_int
    eps = rng.normal(size=(n_trials, sats.n)) * np.sqrt(sats.var_int)
    y = eps + f + nb
    err = y @ s0                            # all-in-view error on this axis
    sep = y @ (sk - s0)                     # solution separation
    exceed = (np.abs(err) > L) if two_sided else (err > L)
    hits = int(np.sum(exceed & (np.abs(sep) <= thr)))
    bound = float(Qbar((L - thr - bias_k) / sig_k)) * (2.0 if two_sided else 1.0)
    emp = hits / n_trials
    return BoundCheck(modes.labels[mode], axis, float(L), float(thr), fault_bias, n_trials, hits, emp,
                      clopper_pearson_upper(hits, n_trials), bound)


def false_alert_rate(sats: SatelliteSet, cfg: ARAIMConfig, n_trials: int = 20000,
                     rng: np.random.Generator | int | None = None):
    """Fault-free probability that at least one solution-separation test fails.

    Returns ``(empirical, upper_confidence_bound, analytic_bound)``; the analytic bound is the
    union bound ``sum_k sum_q 2 Q(K_fa,q) = P_FA / N_ES,CONT`` of Eq. (27) (an upper bound
    because the tests are correlated).
    """
    rng = np.random.default_rng(rng)
    modes = ARAIM(cfg).fault_modes(sats)
    terms = build_terms(sats, np.ones(sats.n, bool), modes, cfg.replace(fast=False), keep_solutions=True)
    y = rng.normal(size=(n_trials, sats.n)) * np.sqrt(sats.var_acc)
    dx = np.einsum("kqn,tn->tkq", terms.s_rows[1:] - terms.s_rows[0:1], y)
    fail = (np.abs(dx) > terms.thresholds[None, 1:, :]).any(axis=(1, 2))
    n = int(fail.sum())
    kfa = terms.k_fa
    analytic = float(terms.n_modes * (2 * 2 * Qbar(kfa[0]) + 2 * Qbar(kfa[2])))
    return n / n_trials, clopper_pearson_upper(n, n_trials), analytic


@dataclass
class DetectionStats:
    n_trials: int
    detected: float            # fraction of epochs in which the all-in-view tests failed
    correct_exclusion: float   # fraction excluded exactly the faulted satellites
    wrong_exclusion: float
    alerts: float
    hmi: float                 # fraction with a position error above the PL and no alert
    mean_vpl: float


def detection_exclusion_stats(sats: SatelliteSet, cfg: ARAIMConfig, fault: FaultSpec | Sequence[FaultSpec],
                              n_trials: int = 200, rng: np.random.Generator | int | None = None,
                              noise: str = "acc") -> DetectionStats:
    """Run the monitor ``n_trials`` times with the given fault(s) and summarise the outcome."""
    rng = np.random.default_rng(rng)
    faults = [fault] if isinstance(fault, FaultSpec) else list(fault)
    mon = ARAIMMonitor(cfg)
    n_det = n_ok = n_wrong = n_alert = n_hmi = 0
    vpls = []
    faulted_ids = {sats.ids[i] for f in faults for i in f.satellites}
    for _ in range(n_trials):
        ep = make_epoch(sats, faults, rng, noise=noise)
        r = mon.process(sats, ep.y)
        if r.alert:
            n_alert += 1
            continue
        if r.status == "excluded":
            n_det += 1
            if r.excluded in faulted_ids or (r.excluded and r.excluded.endswith("-const") and len(faulted_ids) > 1):
                n_ok += 1
            else:
                n_wrong += 1
        err = r.position - 0.0
        # the monitor returns the position estimate of the (possibly reduced) solution
        if abs(err[2]) > r.vpl or np.hypot(err[0], err[1]) > r.hpl:
            n_hmi += 1
        vpls.append(r.vpl)
    return DetectionStats(n_trials, n_det / n_trials, n_ok / n_trials, n_wrong / n_trials,
                          n_alert / n_trials, n_hmi / n_trials, float(np.mean(vpls)) if vpls else np.inf)
