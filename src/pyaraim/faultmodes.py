"""Determination of the fault modes to be monitored (Sections "Upper bound on the
probability of a fault affecting a user over a given interval" and
"Determination of fault modes" of Blanch et al., ION ITM 2022).

Primary events
--------------
Every satellite and every constellation is an independent *primary event* with
instantaneous prior ``P_event`` (Eq. 5) and prior over the exposure window
(Eq. 9)::

    P_event(T) = (1 + T / MFD_event) * P_event

A *fault mode* is a combination of primary events. Its prior is the probability
that exactly these events (and no other) are present, Eq. (10)::

    p_fault,k = prod_{i in k} P_i * prod_{i not in k} (1 - P_i)

Each mode *k* is represented by the boolean vector ``masks[k]`` of the
measurements that are **not** affected (``idx_k`` in the paper).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import ARAIMConfig
from .geometry import SatelliteSet

__all__ = [
    "FaultModeList",
    "p_multiple_bound",
    "determine_nmax",
    "determine_fault_modes",
    "unique_masks",
]


# --------------------------------------------------------------------------- #
# Probability of r or more simultaneous events (Appendix C of the TAES paper)  #
# --------------------------------------------------------------------------- #
def p_multiple_bound(p_event: np.ndarray, r: int) -> float:
    """Upper bound on the probability that **r or more** independent events occur.

    Exact for ``r <= 3`` (Eqs. 70-72 of the TAES paper), and the bound
    ``(sum p)^r / r!`` of Eq. (74) for ``r >= 4``. Events with probability one
    are removed beforehand and reduce ``r`` accordingly.
    """
    p = np.asarray(p_event, dtype=float)
    certain = p >= 1.0
    r = r - int(certain.sum())
    p = p[~certain]
    if r <= 0:
        return 1.0
    p_nofault = float(np.prod(1.0 - p))
    x = p / (1.0 - p)
    if r == 1:
        return 1.0 - p_nofault
    if r == 2:
        return 1.0 - p_nofault * (1.0 + x.sum())
    if r == 3:
        return (1.0 - p_nofault * (1.0 + x.sum())
                - 0.5 * p_nofault * (x.sum() ** 2 - (x**2).sum()))
    fact = float(np.prod(np.arange(1, r + 1)))
    return float(p.sum() ** r / fact)


def determine_nmax(p_event: np.ndarray, p_thres: float) -> int:
    """Largest number of simultaneous events that must be monitored (Eqs. 7, 75-78).

    It is the smallest ``N`` such that the probability of ``N + 1`` or more
    simultaneous events is at most ``p_thres``.
    """
    n = len(p_event)
    r = 0
    p_nm = 1.0
    while p_nm > p_thres and r < n + 1:
        r += 1
        p_nm = p_multiple_bound(p_event, r)
    return r - 1


# --------------------------------------------------------------------------- #
# Data container                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class FaultModeList:
    """List of monitored fault modes.

    Attributes
    ----------
    masks : (K, n) bool array
        ``True`` where the measurement is used by the fault-tolerant subset ``k``
        (i.e. it is *not* affected by fault mode ``k``). The fault-free
        (all-in-view) mode is **not** part of the list.
    p_inst : (K,) array
        Instantaneous prior ``p_fault,k`` of every mode.
    p_exp : (K,) array
        Upper bound ``pbar_fault,k(T_EXP)`` of the probability of the mode at any
        time during the exposure window (Eq. 10).
    p_not_monitored : float
        Bound of the probability of the modes that are **not** monitored (Eq. 14).
    labels : list of str
        Human-readable description of each mode.
    n_fault_max : int
        Maximum number of simultaneous primary events that was enumerated.
    """

    masks: np.ndarray
    p_inst: np.ndarray
    p_exp: np.ndarray
    p_not_monitored: float = 0.0
    labels: List[str] = field(default_factory=list)
    n_fault_max: int = 0
    p_nofault_exp: float = 1.0

    def __post_init__(self) -> None:
        self.masks = np.atleast_2d(np.asarray(self.masks, dtype=bool))
        self.p_inst = np.asarray(self.p_inst, dtype=float).ravel()
        self.p_exp = np.asarray(self.p_exp if self.p_exp is not None else self.p_inst,
                                dtype=float).ravel()
        if not self.labels:
            self.labels = [f"mode{k}" for k in range(len(self.p_inst))]

    def __len__(self) -> int:
        return int(self.masks.shape[0])

    @classmethod
    def from_masks(cls, masks, p_inst, p_exp=None, p_not_monitored: float = 0.0,
                   labels: Optional[Sequence[str]] = None) -> "FaultModeList":
        """Build a list from user-provided subsets (e.g. to reproduce a published example)."""
        masks = np.atleast_2d(np.asarray(masks, bool))
        p_inst = np.broadcast_to(np.asarray(p_inst, float), (masks.shape[0],)).copy()
        p_exp = p_inst.copy() if p_exp is None else np.broadcast_to(np.asarray(p_exp, float),
                                                                    (masks.shape[0],)).copy()
        return cls(masks, p_inst, p_exp, p_not_monitored, list(labels) if labels else [])


def unique_masks(masks: np.ndarray, *weights: np.ndarray):
    """Group identical rows, summing the associated ``weights`` (Eq. 18).

    Returns ``(unique_masks, summed_weights..., first_index, inverse)`` where the
    unique rows keep the order of their first occurrence.
    """
    masks = np.asarray(masks, bool)
    if masks.shape[0] == 0:
        return (masks, *[np.zeros(0) for _ in weights], np.zeros(0, int), np.zeros(0, int))
    _, first, inv = np.unique(masks, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).ravel()
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    inv = rank[inv]
    first = first[order]
    out = [masks[first]]
    for w in weights:
        out.append(np.bincount(inv, weights=np.asarray(w, float), minlength=len(first)))
    out.extend([first, inv])
    return tuple(out)


# --------------------------------------------------------------------------- #
# Main routine                                                                #
# --------------------------------------------------------------------------- #
def _event_tables(sats: SatelliteSet, t_exp: float):
    """Prior of every primary event (satellites then constellations), Eqs. (5), (9)."""
    p_inst = np.concatenate([sats.p_sat, sats.p_const])
    mfd = np.concatenate([sats.mfd_sat, sats.mfd_const])
    with np.errstate(divide="ignore"):
        tau = np.where(mfd > 0, t_exp / mfd, 0.0)
    p_exp = np.minimum(p_inst * (1.0 + tau), 1.0 - 1e-12)
    # membership: event e removes satellite s
    n, nc = sats.n, sats.n_const
    member = np.zeros((n + nc, n), dtype=bool)
    member[np.arange(n), np.arange(n)] = True
    for j in range(nc):
        member[n + j] = sats.constellation == j
    return p_inst, p_exp, member


def _event_name(sats: SatelliteSet, e: int) -> str:
    return sats.ids[e] if e < sats.n else f"{sats.const_names[e - sats.n]}-const"


def determine_fault_modes(sats: SatelliteSet, cfg: ARAIMConfig) -> FaultModeList:
    """Select the fault modes to monitor and their priors for a given geometry.

    Implements, in this order: enumeration of all combinations of up to
    ``N_fault,max`` primary events (Eq. 7 / Appendix C), probabilities of
    Eqs. (9)-(10), merging of identical subsets (Eq. 18), fault consolidation
    (Eqs. 15-16), and the ordered selection of Eqs. (11)-(14) (or one of the
    alternatives selected by ``cfg.subset_selection``).
    """
    n, nc = sats.n, sats.n_const
    p_ev_inst, p_ev_exp, member = _event_tables(sats, cfg.t_exp)
    n_events = n + nc

    if cfg.prior_form == "product":
        p_nofault_inst = p_nofault_exp = 1.0
        r_inst, r_exp = p_ev_inst, p_ev_exp
        p_nofault_exp_total = float(np.prod(1.0 - p_ev_exp))
    else:
        p_nofault_inst = float(np.prod(1.0 - p_ev_inst))
        p_nofault_exp = float(np.prod(1.0 - p_ev_exp))
        r_inst = p_ev_inst / (1.0 - p_ev_inst)
        r_exp = p_ev_exp / (1.0 - p_ev_exp)
        p_nofault_exp_total = p_nofault_exp

    nmax = max(determine_nmax(p_ev_exp, cfg.p_thres), cfg.min_fault_order)
    nmax = min(nmax, n_events)

    # ---- enumerate combinations of events ---------------------------------
    removed_blocks, pinst_blocks, pexp_blocks, deg_blocks, combo_blocks = [], [], [], [], []
    for k in range(1, nmax + 1):
        combos = np.array(list(combinations(range(n_events), k)), dtype=int)
        removed_blocks.append(member[combos].any(axis=1))
        pinst_blocks.append(p_nofault_inst * np.prod(r_inst[combos], axis=1))
        pexp_blocks.append(p_nofault_exp * np.prod(r_exp[combos], axis=1))
        deg_blocks.append(np.full(len(combos), k))
        combo_blocks.append(combos)
    removed = np.vstack(removed_blocks)
    p_inst = np.concatenate(pinst_blocks)
    p_exp = np.concatenate(pexp_blocks)
    degree = np.concatenate(deg_blocks)
    # representative combination (first / lowest degree) per unique subset
    combos_flat: List[Tuple[int, ...]] = [tuple(c) for blk in combo_blocks for c in blk.tolist()]

    # ---- merge identical subsets (Eq. 18) ---------------------------------
    masks, p_inst, p_exp, first, inv = unique_masks(~removed, p_inst, p_exp)
    degree_u = np.full(len(first), 99)
    np.minimum.at(degree_u, inv, degree)
    rep = [combos_flat[i] for i in first]
    labels = ["+".join(_event_name(sats, e) for e in c) for c in rep]
    degree = degree_u

    # ---- fault consolidation (Eqs. 15-16) ---------------------------------
    if cfg.consolidation != "none" and cfg.fc_thres > 0 and nc >= 1:
        masks, p_inst, p_exp, labels, degree = _consolidate(
            sats, masks, p_inst, p_exp, labels, degree, cfg.fc_thres, cfg.consolidation)

    # ---- selection --------------------------------------------------------
    sel = cfg.subset_selection
    if sel == "all":
        order = np.arange(len(p_exp))
        p_nm = p_multiple_bound(p_ev_exp, nmax + 1)
        chosen = order
    elif sel == "quasi_fixed":
        chosen, p_nm = _select_quasi_fixed(sats, masks, p_exp, degree, cfg, p_nofault_exp_total)
    else:
        chosen, p_nm = _select_default(masks, p_exp, degree, cfg, p_nofault_exp_total)

    return FaultModeList(masks[chosen], p_inst[chosen], p_exp[chosen],
                         p_not_monitored=float(max(p_nm, 0.0)),
                         labels=[labels[i] for i in chosen], n_fault_max=nmax,
                         p_nofault_exp=p_nofault_exp_total)


# --------------------------------------------------------------------------- #
def _consolidate(sats, masks, p_inst, p_exp, labels, degree, fc, rule):
    """Absorb weak multi-satellite modes of one constellation into its constellation-wide mode."""
    n = sats.n
    drop = np.zeros(len(p_inst), bool)
    p_inst = p_inst.copy()
    p_exp = p_exp.copy()
    removed = ~masks
    for j in range(sats.n_const):
        members = sats.constellation == j
        if n - members.sum() < 4:      # constellation mode would not be monitorable
            continue
        kj = np.where((removed == members).all(axis=1))[0]
        if kj.size == 0:
            continue
        kj = int(kj[0])
        # modes made only of satellites of constellation j (strict subsets of the constellation)
        inside = (removed & ~members).sum(axis=1) == 0
        inside &= removed.any(axis=1)
        inside[kj] = False
        inside &= ~drop
        cand = np.where(inside)[0]
        if cand.size == 0:
            continue
        if rule == "group":
            if p_inst[cand].sum() <= fc * p_inst[kj]:
                sel = cand
            else:
                continue
        else:  # individual
            sel = cand[p_inst[cand] < fc * p_inst[kj]]
            if sel.size == 0:
                continue
        p_inst[kj] += p_inst[sel].sum()
        p_exp[kj] += p_exp[sel].sum()
        drop[sel] = True
    keep = ~drop
    return (masks[keep], p_inst[keep], p_exp[keep],
            [l for l, k in zip(labels, keep) if k], degree[keep])


def _select_default(masks, p_exp, degree, cfg, p_nofault_exp):
    """Eqs. (11)-(14): mandatory modes first, then fill by decreasing geometry strength."""
    phmi = cfg.phmi
    large = np.where(p_exp > phmi)[0]
    p_nm = 1.0 - p_nofault_exp - p_exp[large].sum()
    other = np.setdiff1d(np.arange(len(p_exp)), large, assume_unique=True)
    sats_in = masks[other].sum(axis=1)
    if cfg.mode_ordering == "degree":
        key = np.lexsort((-p_exp[other], -sats_in, degree[other]))
    else:  # geometry: most satellites remaining first (== fewest satellites out)
        key = np.lexsort((-p_exp[other], degree[other], -sats_in))
    other = other[key]
    added = []
    for idx in other:
        if p_nm <= cfg.p_thres:
            break
        if p_exp[idx] > 0:
            p_nm -= p_exp[idx]
            added.append(idx)
    chosen = np.concatenate([large, np.array(added, dtype=int)]).astype(int)
    return chosen, p_nm


def _select_quasi_fixed(sats, masks, p_exp, degree, cfg, p_nofault_exp):
    """Quasi-fixed subset list: (1) p > PHMI, (2) dual faults from two constellations,
    (3) constellation + satellite faults; same-constellation duals are grouped in the
    constellation fault (they are consolidated unconditionally)."""
    phmi = cfg.phmi
    removed = ~masks
    large = np.where(p_exp > phmi)[0]
    p_nm = 1.0 - p_nofault_exp - p_exp[large].sum()
    # number of distinct constellations touched by the removed satellites
    touched = np.stack([(removed & (sats.constellation == j)).any(axis=1)
                        for j in range(sats.n_const)], axis=1)
    n_touch = touched.sum(axis=1)
    # whole-constellation removals
    whole = np.stack([((removed & (sats.constellation == j)).sum(axis=1) == (sats.constellation == j).sum())
                      for j in range(sats.n_const)], axis=1).any(axis=1)
    rem_count = removed.sum(axis=1)
    cross_dual = (degree == 2) & (n_touch >= 2) & (~whole)
    const_sat = (degree == 2) & whole & (n_touch >= 2)
    pool = [np.where(cross_dual)[0], np.where(const_sat)[0]]
    chosen = list(large)
    for grp in pool:
        grp = np.setdiff1d(grp, large)
        grp = grp[np.argsort(rem_count[grp], kind="stable")]  # strongest geometry first
        for idx in grp:
            if p_nm <= cfg.p_thres:
                break
            p_nm -= p_exp[idx]
            chosen.append(idx)
    return np.array(chosen, dtype=int), p_nm
