"""Snapshot of the measurements available to the user at one epoch.

A :class:`SatelliteSet` carries everything the user algorithm needs *before*
looking at the pseudorange values themselves: the line-of-sight geometry, the
constellation membership, the nominal error model (``C_int``, ``C_acc``, nominal
biases) and the fault priors from the ISD.

Geometry convention
-------------------
Following Appendix E of DO-229 (and ``usr_vhpl_araim.m``) the geometry matrix is
expressed in East-North-Up coordinates with **negated** line-of-sight vectors and
one clock column per constellation (Eq. 4 of the TAES paper)::

    G[i, :3]    = -los_enu[i]            (user -> satellite unit vector, negated)
    G[i, 3 + j] = 1  if satellite i belongs to constellation j, else 0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from .errormodels import UserErrorModel, nominal_error_terms
from .isd import ISD

__all__ = ["SatelliteSet"]


@dataclass
class SatelliteSet:
    """Geometry, nominal error model and priors for *n* satellites in view.

    Attributes
    ----------
    los_enu : (n, 3) array
        Unit line-of-sight vectors from the user to the satellites in ENU.
    constellation : (n,) int array
        Index (0 .. n_const-1) of the constellation of every satellite.
    const_names : tuple of str
        Names of the constellations (clock columns of ``G``).
    var_int, var_acc : (n,) arrays
        Diagonals of ``C_int`` and ``C_acc`` [m^2].
    b_int, b_acc : (n,) arrays
        Maximum nominal biases [m] for integrity and for accuracy/continuity.
    p_sat, mfd_sat : (n,) arrays
        Prior probability of a single-satellite fault and mean fault duration [h].
    p_const, mfd_const : (n_const,) arrays
        Same for constellation-wide faults.
    ids : tuple of str
        Human-readable satellite labels (e.g. ``"GPS-07"``).
    """

    los_enu: np.ndarray
    constellation: np.ndarray
    const_names: Tuple[str, ...]
    var_int: np.ndarray
    var_acc: np.ndarray
    b_int: np.ndarray
    b_acc: np.ndarray
    p_sat: np.ndarray
    mfd_sat: np.ndarray
    p_const: np.ndarray
    mfd_const: np.ndarray
    ids: Tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        self.los_enu = np.atleast_2d(np.asarray(self.los_enu, dtype=float))
        n = self.los_enu.shape[0]
        self.constellation = np.asarray(self.constellation, dtype=int)
        for name in ("var_int", "var_acc", "b_int", "b_acc", "p_sat", "mfd_sat"):
            arr = np.broadcast_to(np.asarray(getattr(self, name), dtype=float), (n,)).copy()
            setattr(self, name, arr)
        nc = len(self.const_names)
        for name in ("p_const", "mfd_const"):
            arr = np.broadcast_to(np.asarray(getattr(self, name), dtype=float), (nc,)).copy()
            setattr(self, name, arr)
        if self.constellation.shape != (n,):
            raise ValueError("constellation must have one entry per satellite")
        if n and (self.constellation.min() < 0 or self.constellation.max() >= nc):
            raise ValueError("constellation index out of range")
        if not self.ids:
            self.ids = tuple(f"{self.const_names[c]}-{i}" for i, c in enumerate(self.constellation))
        if len(self.ids) != n:
            raise ValueError("ids must have one entry per satellite")
        if np.any(self.var_int <= 0) or np.any(self.var_acc <= 0):
            raise ValueError("pseudorange variances must be strictly positive")

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        """Number of satellites."""
        return self.los_enu.shape[0]

    @property
    def n_const(self) -> int:
        """Number of constellations (clock columns of ``G``)."""
        return len(self.const_names)

    @property
    def G(self) -> np.ndarray:
        """Geometry matrix (n, 3 + n_const), Eq. (4)."""
        clk = np.zeros((self.n, self.n_const))
        clk[np.arange(self.n), self.constellation] = 1.0
        return np.hstack([-self.los_enu, clk])

    @property
    def weights_int(self) -> np.ndarray:
        """Diagonal of ``W = C_int^-1``, Eq. (5)."""
        return 1.0 / self.var_int

    def const_members(self, j: int) -> np.ndarray:
        """Boolean mask of the satellites of constellation ``j``."""
        return self.constellation == j

    def elevation(self) -> np.ndarray:
        """Elevation of each satellite [rad] (from the ENU line of sight)."""
        return np.arcsin(np.clip(self.los_enu[:, 2], -1.0, 1.0))

    def drop(self, idx: Sequence[int]) -> "SatelliteSet":
        """Return a new set without the satellites in ``idx`` (outage)."""
        keep = np.ones(self.n, bool)
        keep[np.asarray(list(idx), dtype=int)] = False
        return self.select(keep)

    def select(self, mask: np.ndarray) -> "SatelliteSet":
        """Return a new set restricted to ``mask`` (constellations with no satellite are removed)."""
        mask = np.asarray(mask, bool)
        used = np.unique(self.constellation[mask])
        remap = {int(c): k for k, c in enumerate(used)}
        return SatelliteSet(
            los_enu=self.los_enu[mask],
            constellation=np.array([remap[int(c)] for c in self.constellation[mask]], dtype=int),
            const_names=tuple(self.const_names[c] for c in used),
            var_int=self.var_int[mask], var_acc=self.var_acc[mask],
            b_int=self.b_int[mask], b_acc=self.b_acc[mask],
            p_sat=self.p_sat[mask], mfd_sat=self.mfd_sat[mask],
            p_const=self.p_const[used], mfd_const=self.mfd_const[used],
            ids=tuple(i for i, m in zip(self.ids, mask) if m),
        )

    def pruned(self) -> "SatelliteSet":
        """Remove constellations that have no satellite in view."""
        return self.select(np.ones(self.n, bool))

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_los(
        cls,
        los_enu: np.ndarray,
        constellation: Sequence[str],
        isd: ISD,
        error_model: UserErrorModel | None = None,
        ids: Optional[Sequence[str]] = None,
        sigma_ura: Optional[np.ndarray] = None,
        sigma_ure: Optional[np.ndarray] = None,
    ) -> "SatelliteSet":
        """Build a set from ENU lines of sight, constellation names and the ISD.

        The elevation needed by the error models is derived from ``los_enu``.
        """
        los = np.atleast_2d(np.asarray(los_enu, dtype=float))
        names = list(constellation)
        el = np.arcsin(np.clip(los[:, 2], -1, 1))
        var_int, var_acc, b_int, b_acc = nominal_error_terms(isd, names, el, error_model,
                                                             sigma_ura, sigma_ure)
        uniq = list(dict.fromkeys(names))
        cidx = np.array([uniq.index(c) for c in names], dtype=int)
        p_sat = np.array([isd[c].p_sat for c in names])
        mfd_sat = np.array([isd[c].mfd_sat for c in names])
        p_const = np.array([isd[c].p_const for c in uniq])
        mfd_const = np.array([isd[c].mfd_const for c in uniq])
        return cls(los, cidx, tuple(uniq), var_int, var_acc, b_int, b_acc,
                   p_sat, mfd_sat, p_const, mfd_const,
                   tuple(ids) if ids is not None else ())

    @classmethod
    def from_geometry_matrix(
        cls,
        G: np.ndarray,
        var_int,
        var_acc,
        b_int=0.0,
        b_acc=0.0,
        p_sat=1e-5,
        p_const=1e-8,
        mfd_sat=1.0,
        mfd_const=1.0,
        const_names: Optional[Sequence[str]] = None,
    ) -> "SatelliteSet":
        """Build a set directly from a geometry matrix ``G`` (n x (3 + n_const)).

        This is how the numerical example of Appendix J of the TAES paper is
        reproduced. ``G`` must follow the convention described in the module
        docstring (negated LOS, one clock column per constellation).
        """
        G = np.asarray(G, dtype=float)
        nc = G.shape[1] - 3
        cidx = np.argmax(G[:, 3:], axis=1)
        names = tuple(const_names) if const_names is not None else tuple(f"C{j + 1}" for j in range(nc))
        return cls(-G[:, :3], cidx, names, var_int, var_acc, b_int, b_acc,
                   p_sat, mfd_sat, p_const, mfd_const)
