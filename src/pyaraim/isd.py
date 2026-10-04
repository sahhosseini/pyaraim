"""Integrity Support Data (ISD).

Table 1 of Blanch et al. (ION ITM 2022) lists the parameters that describe the
nominal error model (``sigma_URA``, ``sigma_URE``, ``b_nom``) and the fault
modes (``R_sat``, ``R_const``, ``MFD_sat``, ``MFD_const``). The prior fault
probabilities follow from Eq. (5)::

    P_sat   = R_sat   * MFD_sat
    P_const = R_const * MFD_const
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Mapping

__all__ = ["ConstellationISD", "ISD"]


@dataclass(frozen=True)
class ConstellationISD:
    """ISD of one constellation (all satellites share the same values).

    Per-satellite values (e.g. a broadcast URA) can be injected when building a
    :class:`~pyaraim.geometry.SatelliteSet`.

    Parameters
    ----------
    name:
        Constellation label (``"GPS"``, ``"Galileo"``, ...).
    sigma_ura, sigma_ure:
        Std-dev [m] of the clock and ephemeris error used for **integrity**
        (URA) and for **accuracy / continuity** (URE).
    b_nom, b_nom_cont:
        Maximum nominal bias [m] for integrity / for continuity and accuracy.
    r_sat, r_const:
        Fault rates [1/h] of a single-satellite fault and of a fault affecting
        more than one satellite of the constellation.
    mfd_sat, mfd_const:
        Mean fault durations [h].
    """

    name: str
    sigma_ura: float
    sigma_ure: float
    b_nom: float = 0.0
    b_nom_cont: float = 0.0
    r_sat: float = 1e-5
    mfd_sat: float = 1.0
    r_const: float = 1e-8
    mfd_const: float = 1.0

    @property
    def p_sat(self) -> float:
        """Prior probability of a satellite fault at a given time, Eq. (5)."""
        return self.r_sat * self.mfd_sat

    @property
    def p_const(self) -> float:
        """Prior probability of a constellation-wide fault at a given time, Eq. (5)."""
        return self.r_const * self.mfd_const

    def with_priors(self, p_sat: float, p_const: float, mfd_sat: float | None = None,
                    mfd_const: float | None = None) -> "ConstellationISD":
        """Build a copy from *probabilities* instead of rates (rate := P / MFD)."""
        ms = self.mfd_sat if mfd_sat is None else mfd_sat
        mc = self.mfd_const if mfd_const is None else mfd_const
        return replace(self, r_sat=p_sat / ms, mfd_sat=ms, r_const=p_const / mc, mfd_const=mc)


class ISD:
    """A set of :class:`ConstellationISD`, indexed by constellation name."""

    def __init__(self, constellations: Iterable[ConstellationISD] | Mapping[str, ConstellationISD]):
        if isinstance(constellations, Mapping):
            constellations = constellations.values()
        self._c: Dict[str, ConstellationISD] = {c.name: c for c in constellations}

    def __getitem__(self, name: str) -> ConstellationISD:
        try:
            return self._c[name]
        except KeyError as exc:
            raise KeyError(f"no ISD for constellation {name!r}; have {list(self._c)}") from exc

    def __contains__(self, name: str) -> bool:
        return name in self._c

    def __iter__(self):
        return iter(self._c.values())

    def names(self):
        return list(self._c)

    def replace(self, name: str, **changes) -> "ISD":
        new = dict(self._c)
        new[name] = replace(self._c[name], **changes)
        return ISD(new)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "ISD(" + ", ".join(self._c) + ")"

    # ------------------------------------------------------------------ presets
    @classmethod
    def paper_2022(cls) -> "ISD":
        """ISD of the "Simulation settings" table of Blanch et al., ION ITM 2022.

        GPS: ``R_sat = 1e-5/h``, ``R_const = 1e-8/h``, MFD 1 h, URA 2.4 m.
        Galileo: ``P_sat = P_const = 1e-4`` (MFD 1 h), URA 6 m. The paper does not
        list the URE; the values of the MAAST scripts (2.4 m and 4 m) are assumed.
        """
        return cls([
            ConstellationISD("GPS", sigma_ura=2.4, sigma_ure=2.4, r_sat=1e-5, mfd_sat=1.0,
                             r_const=1e-8, mfd_const=1.0),
            ConstellationISD("Galileo", sigma_ura=6.0, sigma_ure=4.0, r_sat=1e-4, mfd_sat=1.0,
                             r_const=1e-4, mfd_const=1.0),
        ])

    @classmethod
    def icao_draft_2024(cls) -> "ISD":
        """Default ISD of ``main_maast_araim.m`` (draft ICAO Annex 10, June 2024)."""
        psat_gal, rsat_gal = 3e-5, 2e-5
        pconst_gal, rconst_gal = 2e-4, 1e-4
        return cls([
            ConstellationISD("GPS", 2.4, 2.4, r_sat=1e-5, mfd_sat=1.0, r_const=1e-8, mfd_const=1.0),
            ConstellationISD("Galileo", 6.0, 4.0, r_sat=rsat_gal, mfd_sat=psat_gal / rsat_gal,
                             r_const=rconst_gal, mfd_const=pconst_gal / rconst_gal),
            ConstellationISD("GLONASS", 9.0, 8.0, r_sat=3.4e-5, mfd_sat=3.0, r_const=1e-5, mfd_const=10.0),
            ConstellationISD("BeiDou", float("inf"), float("inf"), r_sat=1e-5, mfd_sat=1.0,
                             r_const=1e-4, mfd_const=1.0),
        ])

    @classmethod
    def baseline_2015_example(cls, n_const: int = 2) -> "ISD":
        """Appendix J of the TAES paper: URA 0.75 m, URE 0.5 m, b_nom 0.5 m, P_sat 1e-4, P_const 1e-5."""
        return cls([
            ConstellationISD(f"C{j + 1}", 0.75, 0.5, b_nom=0.5, b_nom_cont=0.0,
                             r_sat=1e-4, mfd_sat=1.0, r_const=1e-5, mfd_const=1.0)
            for j in range(n_const)
        ])
