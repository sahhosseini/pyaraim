"""Availability / coverage simulation (the Python counterpart of ``svmrun_araim.m``).

For every epoch and every user of a :class:`~pyaraim.simulation.grid.UserGrid` the satellites above
the mask angle are selected, a :class:`~pyaraim.geometry.SatelliteSet` is built from the ISD and the
error model, and :meth:`pyaraim.araim.ARAIM.protection_levels` is evaluated. The results are stored
in an :class:`AvailabilityResult`, from which availability and coverage are derived exactly as in
``coverage.m`` (cos-latitude weighting of users).
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Union

import numpy as np

from ..araim import ARAIM
from ..config import ARAIMConfig, Criteria
from ..constants import DEG
from ..errormodels import ED259A, UserErrorModel
from ..geometry import SatelliteSet
from ..isd import ISD
from .almanac import Almanac
from .grid import UserGrid

__all__ = ["Scenario", "AvailabilityResult", "run_availability", "epoch_satellite_sets"]

QUANT = ("vpl", "hpl", "emt", "sigma_acc")


@dataclass
class Scenario:
    """Everything that defines an availability simulation.

    Parameters
    ----------
    almanacs:
        One :class:`Almanac` per constellation (its ``constellation`` name must exist in ``isd``).
    isd:
        Integrity support data.
    cfg:
        Algorithm configuration.
    grid:
        User grid.
    times:
        Epochs [s since the almanac time of applicability]; default one day at 300 s.
    mask_deg:
        Elevation mask [deg], a scalar or a ``{constellation: mask}`` mapping.
    error_model:
        User error model (default :class:`~pyaraim.errormodels.ED259A`).
    """

    almanacs: Sequence[Almanac]
    isd: ISD
    cfg: ARAIMConfig
    grid: UserGrid
    times: np.ndarray = field(default_factory=lambda: np.arange(0.0, 86400.0, 300.0))
    mask_deg: Union[float, Dict[str, float]] = 5.0
    error_model: UserErrorModel = field(default_factory=ED259A)
    criteria: Optional[Criteria] = None

    def mask_for(self, name: str) -> float:
        return self.mask_deg[name] if isinstance(self.mask_deg, dict) else float(self.mask_deg)


# --------------------------------------------------------------------------- #
def epoch_satellite_sets(scn: Scenario, t: float, users: Optional[Sequence[int]] = None):
    """Yield ``(user_index, SatelliteSet)`` for the users that see at least one satellite at time ``t``."""
    pos = [a.positions_ecef(t) for a in scn.almanacs]
    prn = [[e.prn for e in a.entries] for a in scn.almanacs]
    healthy = [np.array([e.health == 0 for e in a.entries]) for a in scn.almanacs]
    names = [a.constellation for a in scn.almanacs]
    masks = [np.sin(scn.mask_for(n) * DEG) for n in names]
    users = range(len(scn.grid)) if users is None else users
    for u in users:
        los_all, cons_all, ids = [], [], []
        for c, name in enumerate(names):
            los, _ = scn.grid.los_enu(u, pos[c])
            ok = (los[:, 2] >= masks[c]) & healthy[c]
            if ok.any():
                los_all.append(los[ok])
                cons_all += [name] * int(ok.sum())
                ids += [f"{name}-{prn[c][i]:02d}" for i in np.where(ok)[0]]
        if not los_all:
            continue
        yield u, SatelliteSet.from_los(np.vstack(los_all), cons_all, scn.isd, scn.error_model, ids)


def _run_epoch(args):
    scn, t, users = args
    araim = ARAIM(scn.cfg)
    out = np.full((len(scn.grid), 5), np.inf)
    out[:, 4] = 0
    for u, sats in epoch_satellite_sets(scn, t, users):
        r = araim.protection_levels(sats, scn.criteria)
        out[u] = (r.vpl, r.hpl, r.emt, r.sigma_acc, sats.n)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class AvailabilityResult:
    """Protection levels etc. of every user at every epoch (shape ``(n_users, n_times)``)."""

    lat: np.ndarray
    lon: np.ndarray
    times: np.ndarray
    vpl: np.ndarray
    hpl: np.ndarray
    emt: np.ndarray
    sigma_acc: np.ndarray
    n_sat: np.ndarray
    meta: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------- derived data
    @property
    def weights(self) -> np.ndarray:
        return np.cos(self.lat * DEG)

    def available(self, crit: Criteria) -> np.ndarray:
        """Boolean ``(n_users, n_times)`` array: criteria met."""
        return ((self.vpl <= crit.val) & (self.hpl <= crit.hal) & (self.emt <= crit.emt)
                & (self.sigma_acc <= crit.sigma_acc))

    def availability(self, crit: Criteria) -> np.ndarray:
        """Fraction of epochs in which each user meets ``crit``."""
        return self.available(crit).mean(axis=1)

    def coverage(self, crit: Criteria, level: float = 0.999, lat_max_deg: float = 90.0) -> float:
        """Percentage of users (weighted by ``cos(lat)``) whose availability is at least ``level``.

        Same definition as ``coverage.m`` of MAAST.
        """
        w = self.weights * (np.abs(self.lat) <= lat_max_deg)
        av = self.availability(crit)
        return float(100.0 * np.sum((av >= level - 1e-12) * w) / np.sum(w))

    def percentile(self, name: str, p: float = 99.9) -> np.ndarray:
        """``p``-th percentile over time of ``vpl``/``hpl``/``emt``/``sigma_acc`` for each user."""
        if name not in QUANT:
            raise KeyError(name)
        # method="higher": never interpolates, so unavailable (inf) epochs propagate cleanly
        return np.percentile(getattr(self, name), p, axis=1, method="higher")

    # ---------------------------------------------------------------------- I/O
    def save(self, path: Union[str, Path]) -> None:
        np.savez_compressed(path, lat=self.lat, lon=self.lon, times=self.times, vpl=self.vpl, hpl=self.hpl,
                            emt=self.emt, sigma_acc=self.sigma_acc, n_sat=self.n_sat,
                            meta=np.array([f"{k}={v}" for k, v in self.meta.items()]))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AvailabilityResult":
        d = np.load(path, allow_pickle=False)
        meta = dict(s.split("=", 1) for s in d["meta"].tolist())
        return cls(d["lat"], d["lon"], d["times"], d["vpl"], d["hpl"], d["emt"], d["sigma_acc"],
                   d["n_sat"], meta)

    def to_csv(self, path: Union[str, Path], crit: Optional[Criteria] = None, percentile: float = 99.9) -> None:
        """One row per user: position, percentile PLs, and availability if ``crit`` is given."""
        cols = [self.lat, self.lon] + [self.percentile(q, percentile) for q in QUANT]
        head = ["lat", "lon"] + [f"{q}_p{percentile:g}" for q in QUANT]
        if crit is not None:
            cols.append(self.availability(crit))
            head.append("availability")
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=",".join(head), comments="")


def run_availability(scn: Scenario, n_jobs: int = 1, progress: Optional[Callable[[int, int], None]] = None,
                     verbose: bool = False) -> AvailabilityResult:
    """Run the availability simulation of ``scn``.

    Parameters
    ----------
    n_jobs:
        Number of worker processes (epochs are distributed among them).
    progress:
        Optional callback ``progress(done_epochs, total_epochs)``.
    """
    times = np.asarray(scn.times, float)
    nt, nu = len(times), len(scn.grid)
    arr = np.full((4, nu, nt), np.inf)
    nsat = np.zeros((nu, nt), int)
    jobs = [(scn, float(t), None) for t in times]
    t0 = time.time()

    def store(i, out):
        arr[:, :, i] = out[:, :4].T
        nsat[:, i] = out[:, 4].astype(int)
        if progress:
            progress(i + 1, nt)
        if verbose:
            print(f"epoch {i + 1}/{nt}  ({time.time() - t0:.0f}s)")

    if n_jobs == 1:
        for i, j in enumerate(jobs):
            store(i, _run_epoch(j))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, out in enumerate(ex.map(_run_epoch, jobs, chunksize=max(1, nt // (4 * n_jobs)))):
                store(i, out)
    meta = {"constellations": "+".join(a.constellation for a in scn.almanacs),
            "n_users": str(nu), "n_epochs": str(nt)}
    return AvailabilityResult(scn.grid.lat, scn.grid.lon, times, arr[0], arr[1], arr[2], arr[3], nsat, meta)
