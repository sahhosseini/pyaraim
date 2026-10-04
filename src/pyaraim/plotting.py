"""Matplotlib figures for availability studies (maps, time series, PL-vs-geometry scatter).

Matplotlib is imported lazily so the core package does not require it at import time.
The map colour scale of :func:`plot_availability_map` reproduces the legend of Figure 2 of the
ION ITM 2022 paper.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import Criteria
from .simulation.availability import AvailabilityResult

__all__ = ["plot_availability_map", "plot_pl_map", "plot_time_series", "AVAILABILITY_LEVELS"]

#: (threshold, colour) pairs of the availability legend (< 50 % is the first colour).
AVAILABILITY_LEVELS = [(0.0, "#f4a259"), (0.50, "#f6e05e"), (0.75, "#b7e4a1"), (0.85, "#48b96f"),
                       (0.90, "#4fd1d9"), (0.95, "#3a7bd5"), (0.99, "#1d3a8a"), (0.995, "#7b2fa6"),
                       (0.999, "#d62ee0")]


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _grid(res: AvailabilityResult, values: np.ndarray):
    lats, lons = np.unique(res.lat), np.unique(res.lon)
    Z = np.full((len(lats), len(lons)), np.nan)
    ilat = np.searchsorted(lats, res.lat)
    ilon = np.searchsorted(lons, res.lon)
    Z[ilat, ilon] = values
    return lats, lons, Z


def plot_availability_map(res: AvailabilityResult, crit: Criteria, ax=None, title: Optional[str] = None):
    """Map of the availability of ``crit`` with the paper's colour classes."""
    plt = _plt()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    ax = ax or plt.subplots(figsize=(9, 4.5))[1]
    lats, lons, Z = _grid(res, res.availability(crit))
    edges = [l for l, _ in AVAILABILITY_LEVELS] + [1.0 + 1e-9]
    cmap = ListedColormap([c for _, c in AVAILABILITY_LEVELS])
    norm = BoundaryNorm(edges, cmap.N)
    mesh = ax.pcolormesh(lons, lats, Z, cmap=cmap, norm=norm, shading="nearest")
    cb = ax.figure.colorbar(mesh, ax=ax, ticks=edges[:-1])
    cb.ax.set_yticklabels([f"{100 * e:g}%" for e in edges[:-1]])
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(alpha=0.3)
    cov = res.coverage(crit, 0.999)
    ax.set_title(title or f"{crit.name}: coverage(99.9 %) = {cov:.2f} %")
    return ax


def plot_pl_map(res: AvailabilityResult, quantity: str = "hpl", percentile: float = 99.9, ax=None,
                vmax: Optional[float] = None):
    """Map of the ``percentile`` of ``vpl``/``hpl``/``emt``/``sigma_acc`` over time."""
    plt = _plt()
    ax = ax or plt.subplots(figsize=(9, 4.5))[1]
    vals = res.percentile(quantity, percentile)
    vals = np.where(np.isfinite(vals), vals, np.nan)
    lats, lons, Z = _grid(res, vals)
    mesh = ax.pcolormesh(lons, lats, Z, shading="nearest", vmax=vmax, cmap="viridis")
    ax.figure.colorbar(mesh, ax=ax, label=f"{quantity.upper()} {percentile:g}% (m)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(alpha=0.3)
    return ax


def plot_time_series(res: AvailabilityResult, user: int, ax=None):
    """VPL/HPL and number of satellites of one user versus time."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3.5))
    t = res.times / 3600.0
    ax.semilogy(t, np.where(np.isfinite(res.vpl[user]), res.vpl[user], np.nan), label="VPL")
    ax.semilogy(t, np.where(np.isfinite(res.hpl[user]), res.hpl[user], np.nan), label="HPL")
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Protection level (m)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.set_title(f"user at ({res.lat[user]:.0f}, {res.lon[user]:.0f})")
    return ax
