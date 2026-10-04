"""Fault-tolerant position solutions and their statistics (Eqs. 12-17 of the TAES paper).

For a subset of measurements ``idx_k`` the weighted least-squares coefficients are::

    S^(k) = (G' W^(k) G)^-1 G' W^(k),    W^(k) = diag(w_i for i in idx_k, 0 otherwise)

and, for each coordinate ``q`` (East, North, Up)::

    sigma_q^(k)^2       = e_q' S^(k) C_int S^(k)' e_q                    (Eq. 15)
    b_q^(k)             = sum_i |S^(k)_qi| b_nom,i                       (Eq. 16)
    sigma_ss,q^(k)^2    = e_q' (S^(k)-S^(0)) C_acc (S^(k)-S^(0))' e_q    (Eq. 17)

The clock column of a constellation with no satellite in the subset is removed
(Eq. 13). Subsets that leave the position unobservable are flagged with
``observable = False`` and infinite standard deviations.

In addition to the direct batched solution this module implements the
matrix-inversion-lemma ("rank-one update") route of Appendix I and Eq. (41),
which obtains the solution of every subset with one or two satellites removed
from by-products of the all-in-view solution, and the bound on the dual-fault
subsets of the 2022 paper (Eqs. 36-41).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "SubsetSolutions",
    "wls_coefficients",
    "subset_solutions",
    "downdate_coefficients",
    "DualFaultBound",
    "dual_fault_bound",
    "weighted_ls",
]


@dataclass
class SubsetSolutions:
    """Statistics of a reference solution (row 0) and of a list of subsets.

    Row 0 is the *reference* ("all-in-view") solution: for the fault-free case it
    uses every satellite, after an exclusion it uses the retained satellites.
    All other rows are fault-tolerant subsets of the reference.
    """

    masks: np.ndarray          # (M, n) bool
    s: np.ndarray              # (M, 3, n) position coefficients (zeros if unobservable)
    sigma: np.ndarray          # (M, 3)  std-dev of the position error (integrity model)
    sigma_ss: np.ndarray       # (M, 3)  std-dev of the solution separation (accuracy model)
    bias: np.ndarray           # (M, 3)  worst-case nominal bias in position
    bias_ss: np.ndarray        # (M, 3)  worst-case nominal bias of the separation
    observable: np.ndarray     # (M,) bool

    def __len__(self) -> int:
        return int(self.masks.shape[0])


# --------------------------------------------------------------------------- #
def wls_coefficients(G: np.ndarray, w: np.ndarray, masks: np.ndarray,
                     cond_limit: float = 1e10, n_pos: int = 3):
    """Batched weighted-least-squares coefficients ``S^(k)`` for many subsets.

    Parameters
    ----------
    G : (n, 3 + n_const) geometry matrix
    w : (n,) weights (``1 / var_int``)
    masks : (K, n) boolean, ``True`` for the measurements used by subset ``k``
    cond_limit : subsets whose normal matrix has a larger condition number are
        declared unobservable.

    Returns
    -------
    s : (K, 3, n) rows of ``S`` for East, North, Up (0 for unobservable subsets)
    observable : (K,) bool
    """
    G = np.asarray(G, float)
    masks = np.atleast_2d(np.asarray(masks, bool))
    K, n = masks.shape
    nc = G.shape[1] - n_pos
    s = np.zeros((K, n_pos, n))
    observable = np.zeros(K, bool)
    if K == 0:
        return s, observable
    clk = G[:, n_pos:] != 0
    present = (masks[:, :, None] & clk[None, :, :]).any(axis=1)          # (K, nc)
    patterns, inverse = np.unique(present, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    for pidx, pat in enumerate(patterns):
        rows = np.where(inverse == pidx)[0]
        cols = list(range(n_pos)) + [n_pos + j for j in range(nc) if pat[j]]
        m = len(cols)
        Gp = G[:, cols]
        Wk = w[None, :] * masks[rows]                                    # (R, n)
        enough = masks[rows].sum(axis=1) >= m
        rows, Wk = rows[enough], Wk[enough]
        if rows.size == 0:
            continue
        A = np.einsum("rn,ni,nj->rij", Wk, Gp, Gp)                      # (R, m, m)
        with np.errstate(all="ignore"):
            cond = np.linalg.cond(A)
        ok = np.isfinite(cond) & (cond < cond_limit)
        if not ok.any():
            continue
        rows, Wk, A = rows[ok], Wk[ok], A[ok]
        B = np.einsum("ni,rn->rin", Gp, Wk)                              # (R, m, n)
        X = np.linalg.solve(A, B)
        s[rows] = X[:, :n_pos, :]
        observable[rows] = True
    return s, observable


def subset_solutions(
    G: np.ndarray,
    var_int: np.ndarray,
    var_acc: np.ndarray,
    b_int: np.ndarray,
    b_acc: np.ndarray,
    masks: np.ndarray,
    cond_limit: float = 1e10,
) -> SubsetSolutions:
    """Solutions and statistics of every row of ``masks``; **row 0 is the reference**.

    The statistics of Eqs. (15)-(17) are returned for all rows (the entries for row 0
    of ``sigma_ss`` / ``bias_ss`` are zero by construction).
    """
    masks = np.atleast_2d(np.asarray(masks, bool))
    w = 1.0 / np.asarray(var_int, float)
    s, obs = wls_coefficients(G, w, masks, cond_limit)
    return _statistics(s, obs, masks, var_int, var_acc, b_int, b_acc)


def _statistics(s, obs, masks, var_int, var_acc, b_int, b_acc) -> SubsetSolutions:
    var_int = np.asarray(var_int, float)
    var_acc = np.asarray(var_acc, float)
    b_int = np.broadcast_to(np.asarray(b_int, float), var_int.shape)
    b_acc = np.broadcast_to(np.asarray(b_acc, float), var_int.shape)
    sigma = np.sqrt(np.einsum("kqn,n->kq", s**2, var_int))
    bias = np.einsum("kqn,n->kq", np.abs(s), b_int)
    ds = s - s[0:1]
    sigma_ss = np.sqrt(np.einsum("kqn,n->kq", ds**2, var_acc))
    bias_ss = np.einsum("kqn,n->kq", np.abs(ds), b_acc)
    bad = ~obs
    sigma[bad] = np.inf
    sigma_ss[bad] = np.inf
    bias[bad] = np.inf
    bias_ss[bad] = np.inf
    return SubsetSolutions(masks, s, sigma, sigma_ss, bias, bias_ss, obs)


# --------------------------------------------------------------------------- #
# Appendix I / Eq. (41): rank-one (and rank-two) updates                       #
# --------------------------------------------------------------------------- #
def downdate_coefficients(G: np.ndarray, w: np.ndarray, removed_sets: np.ndarray,
                          cond_limit: float = 1e10):
    """Coefficients of the subsets obtained by removing a few satellites from the *full* set.

    Uses the matrix inversion lemma (Appendix I, Eq. 123 and Eq. 41): with
    ``S0 = (G'WG)^-1 G'W`` and ``P = W - W G S0``,::

        S^(J) = S0_{:,in} + S0_{:,J} (P_JJ)^-1 S0_{:,J}' * (G'W)_{:,in}

    for the set ``J`` of removed measurements, so that no ``(3+n_c)``-square matrix has to be
    inverted per subset. Valid when every constellation keeps at least one satellite
    (otherwise ``P_JJ`` is singular and the caller must use :func:`wls_coefficients`).

    Parameters
    ----------
    removed_sets : (K, r) int array of removed satellite indices (r = 1 or 2)

    Returns
    -------
    s : (K, 3, n); observable : (K,) bool
    """
    G = np.asarray(G, float)
    n = G.shape[0]
    removed_sets = np.atleast_2d(np.asarray(removed_sets, int))
    K, r = removed_sets.shape
    s = np.zeros((K, 3, n))
    obs = np.zeros(K, bool)
    GtW = G.T * w[None, :]
    A0 = GtW @ G
    if np.linalg.cond(A0) > cond_limit:
        return s, obs
    cov0 = np.linalg.inv(A0)
    S0 = cov0 @ GtW                                    # (m, n)
    P = np.diag(w) - GtW.T @ S0                        # (n, n)
    S0_3 = S0[:3]
    for k in range(K):
        J = removed_sets[k]
        PJJ = P[np.ix_(J, J)]
        if np.linalg.cond(PJJ) > cond_limit:
            continue
        keep = np.ones(n, bool)
        keep[J] = False
        SJ = S0[:, J]                                  # (m, r)
        delta = SJ @ np.linalg.solve(PJJ, SJ.T)        # (m, m)  == cov_k - cov_0
        Sk = np.zeros((3, n))
        Sk[:, keep] = S0_3[:, keep] + delta[:3] @ GtW[:, keep]
        s[k] = Sk
        obs[k] = True
    return s, obs


# --------------------------------------------------------------------------- #
# Dual-fault bound (Section "Reducing the computational load")                #
# --------------------------------------------------------------------------- #
@dataclass
class DualFaultBound:
    """Upper bounds covering *all* the (valid) dual-satellite-fault subsets of a list."""

    sigma: np.ndarray       # (3,) bound on sigma_q^(J)
    sigma_ss: np.ndarray    # (3,) bound on sigma_ss,q^(J) under C_acc
    bias: np.ndarray        # (3,)
    bias_ss: np.ndarray     # (3,)
    p_norm_max: float
    valid: np.ndarray       # (n_pairs,) bool: pairs covered by the bound


def dual_fault_bound(G: np.ndarray, var_int: np.ndarray, var_acc: np.ndarray,
                     b_int: np.ndarray, b_acc: np.ndarray, pairs: np.ndarray,
                     cond_limit: float = 1e10) -> Optional[DualFaultBound]:
    """Bound the statistics of every dual-satellite-fault subset in ``pairs``.

    Implements Eqs. (37)-(41) of the ION ITM 2022 paper. With
    ``P = W - W G (G'WG)^-1 G'W``, its normalisation ``P_norm`` (unit diagonal) and the
    normalised coefficients ``s_norm = s / sqrt(P_ii)``::

        sigma_ss,q^(J)^2 = s_norm,J P_norm,JJ^-1 s_norm,J'
                         <= (sum of the two largest s_norm,i^2) / (1 - max_{i!=j} |p_norm,ij|)

    where the maxima are taken over the satellites / pairs that actually appear in
    ``pairs`` (a tighter, still rigorous, version of the bound of the paper). Pairs for
    which ``P_JJ`` is (numerically) singular are *not* covered (``valid == False``): their
    subsets are unobservable or must be treated exactly by the caller.
    The effect of unequal integrity/accuracy covariances is covered by the largest ratio
    ``var_acc / var_int`` and the nominal biases by the Cauchy-Schwarz inequality.

    Returns ``None`` if the all-in-view solution is not observable or no pair is valid.
    """
    G = np.asarray(G, float)
    var_int = np.asarray(var_int, float)
    w = 1.0 / var_int
    GtW = G.T * w[None, :]
    A0 = GtW @ G
    if np.linalg.cond(A0) > cond_limit:
        return None
    cov0 = np.linalg.inv(A0)
    S0 = cov0 @ GtW
    P = np.diag(w) - GtW.T @ S0
    pii = np.diag(P)
    pairs = np.atleast_2d(np.asarray(pairs, int))
    i, j = pairs[:, 0], pairs[:, 1]
    scale = np.sqrt(np.clip(pii[i], 0, None) * np.clip(pii[j], 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        pn = np.abs(P[i, j]) / scale
    valid = (pii[i] > 1e-12 * w[i]) & (pii[j] > 1e-12 * w[j]) & np.isfinite(pn) & (pn < 1.0 - 1e-9)
    if not valid.any():
        return None
    pv = pairs[valid]
    p_norm_max = float(pn[valid].max())
    used = np.unique(pv)
    sn2 = (S0[:3, :] ** 2) / np.clip(pii, 1e-300, None)[None, :]       # (3, n)
    top2 = np.sort(sn2[:, used], axis=1)[:, -2:].sum(axis=1)
    sigma_ss_int2 = top2 / (1.0 - p_norm_max)
    sigma = np.sqrt(np.diag(cov0)[:3] + sigma_ss_int2)
    ratio = float(np.max(np.asarray(var_acc, float) / var_int))
    sigma_ss = np.sqrt(ratio * sigma_ss_int2)
    b_int = np.broadcast_to(np.asarray(b_int, float), var_int.shape)
    b_acc = np.broadcast_to(np.asarray(b_acc, float), var_int.shape)
    B_int = float(np.sqrt(np.sum((b_int / np.sqrt(var_int)) ** 2)))
    B_acc = float(np.sqrt(np.sum((b_acc / np.sqrt(var_int)) ** 2)))
    return DualFaultBound(sigma, sigma_ss, sigma * B_int, np.sqrt(sigma_ss_int2) * B_acc,
                          p_norm_max, valid)


# --------------------------------------------------------------------------- #
def weighted_ls(G: np.ndarray, w: np.ndarray, mask: np.ndarray, y: np.ndarray,
                cond_limit: float = 1e10):
    """Weighted LS state estimate, chi-square statistic and degrees of freedom of one subset.

    Returns ``(x, chi2, dof)``; ``x`` has one entry per column of ``G`` (columns whose
    constellation is absent from the subset are returned as ``nan``). ``x``/``chi2`` are
    ``None``/``inf`` if the subset is unobservable. Eq. (22)/(32) of the TAES paper.
    """
    mask = np.asarray(mask, bool)
    nc = G.shape[1] - 3
    cols = [0, 1, 2] + [3 + j for j in range(nc) if (G[mask, 3 + j] != 0).any()]
    Gs, ws, ys = G[np.ix_(mask, cols)], w[mask], y[mask]
    A = Gs.T @ (ws[:, None] * Gs)
    if Gs.shape[0] < len(cols) or np.linalg.cond(A) > cond_limit:
        return None, np.inf, Gs.shape[0] - len(cols)
    xs = np.linalg.solve(A, Gs.T @ (ws * ys))
    r = ys - Gs @ xs
    x = np.full(G.shape[1], np.nan)
    x[cols] = xs
    return x, float(np.sum(ws * r * r)), Gs.shape[0] - len(cols)
