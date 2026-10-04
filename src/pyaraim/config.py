"""Receiver-side algorithm parameters (the "constants" of the baseline algorithm).

:class:`ARAIMConfig` gathers every tunable of the user algorithm: the integrity
and continuity budgets, the temporal-exposure parameters introduced in the 2022
update (``n_es_int``, ``n_es_cont``, ``t_exp``), and the switches that select
between the algorithm variants described in the papers.

Three ready-made presets are provided:

* :meth:`ARAIMConfig.baseline_2015` - constants of Table II of Blanch et al.,
  IEEE TAES 51(1), 2015 (instantaneous risk, fault detection only);
* :meth:`ARAIMConfig.lpv200` - vertical ARAIM with exclusion and a 150 s
  exposure window (``N_ES = 25``), as used in the MAAST reference scripts;
* :meth:`ARAIMConfig.rnp_horizontal` - horizontal ARAIM for RNP operations with
  a 1 h exposure window (``N_ES = 360``), the scenario of Figure 2 of Blanch et
  al., ION ITM 2022.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Optional

__all__ = ["ARAIMConfig", "Criteria"]

ThresholdMethod = Literal["solution_separation", "chi2"]
Consolidation = Literal["group", "individual", "none"]
SubsetSelection = Literal["default", "quasi_fixed", "all"]
ModeOrdering = Literal["geometry", "degree"]
ExclusionAllocation = Literal["equal", "joint"]
PriorForm = Literal["exact", "product"]


@dataclass(frozen=True)
class ARAIMConfig:
    """Parameters of the ARAIM user algorithm.

    Attributes
    ----------
    phmi_vert, phmi_hor:
        Integrity budget allocated to the vertical / horizontal coordinates
        (``PHMI_VERT``, ``PHMI_HOR``). Their sum is the total ``PHMI``.
    p_thres:
        ``P_THRES``: maximum total probability of the fault modes that are *not*
        monitored (Eq. 11).
    pfa_vert, pfa_hor:
        Continuity budget (false alert) allocated to the vertical / horizontal
        solution-separation tests; ``P_FA = pfa_vert + pfa_hor`` (Eq. 28).
    p_emt:
        Prior-probability level used to define the Effective Monitor Threshold.
    pl_tol:
        Tolerance [m] of the protection-level search (``TOL_PL``).
    fc_thres:
        ``F_C`` of Eq. (15): fault-consolidation factor.
    n_es_int, n_es_cont:
        Effective number of independent samples over the exposure interval for
        integrity (``N_ES``) and continuity (``N_ES,CONT``), Eqs. (2)-(3).
    t_exp:
        Exposure window ``T_EXP`` [hours] used to bound the probability of
        faults that are *not* monitored and the fault-mode prior (Eqs. 6-10).
    exclusion:
        Compute worst-case protection levels over all pre-selected exclusion
        options (fault detection and exclusion, FDE). If ``False`` the algorithm
        is fault detection only and the integrity allocation is not split.
    exclusion_const_threshold:
        Constellation-wide faults whose prior ``P_const`` is at least this value
        are added to the exclusion candidates (``None``: satellites only).
    cap_by_exposure:
        Apply the refinement of Blanch & Walter (ION GNSS+ 2023) that caps the
        integrity-risk contribution of every mode by ``p_fault(T_EXP)/N_ES``
        (``max_pmd_flag`` in MAAST).
    fast:
        Bound all dual-satellite-fault subsets by a single term using the
        results of Section "Reducing the computational load" (Eqs. 36-41).
    threshold_method:
        ``"solution_separation"`` (one Gaussian threshold per mode, Eq. 28) or
        ``"chi2"`` (a single chi-square test, Eq. 36 / Eq. 55 of the TAES paper).
    consolidation:
        Fault-consolidation rule: ``"group"`` implements Eqs. (15)-(16) literally
        (all single-constellation multi-satellite modes are absorbed together when their
        summed prior is below ``fc_thres`` times the constellation prior),
        ``"individual"`` (default) mimics the MAAST code (each mode is absorbed
        independently when its prior is below ``fc_thres`` times the constellation prior), ``"none"`` disables it.
    subset_selection:
        ``"default"``: ordered list of Eqs. (11)-(14); ``"quasi_fixed"``: the
        fixed-order alternative of Section "Quasi-fixed subset list"; ``"all"``:
        the TAES-2015 rule, i.e. monitor *every* combination of at most
        ``N_fault,max`` primary events.
    mode_ordering:
        Ordering of the modes that are *not* mandatory when filling the monitored
        list: ``"geometry"`` (strongest remaining geometry first, as MAAST) or
        ``"degree"`` (lowest number of simultaneous primary events first).
    min_fault_order:
        Minimum number of simultaneous primary events enumerated (MAAST forces 2).
    prior_form:
        ``"exact"``: ``p_fault,k = prod_{i in k} P_i prod_{i not in k} (1 - P_i)`` (Eq. 10 of the
        2022 paper, used by MAAST); ``"product"``: the simple product ``prod_{i in k} P_i`` of
        Eq. (9) of the TAES paper (reproduces its Appendix J).
    exclusion_allocation:
        ``"equal"``: ``rho_j = 1/(N_exc+1)`` for every exclusion option (Eq. 24) and
        the worst-case PL over the options is reported; ``"joint"``: one PL that
        solves Eq. (25) (equal PL under every option).
    optimize_position, sig_acc_max_*:
        Use the integrity-optimised position estimator of Blanch et al. (TAES 51(3),
        2015) that trades accuracy (``sigma_acc <= sig_acc_max_*``) for lower PLs.
    chi2_sanity_check, pfa_chi2:
        Optional chi-square sanity test of the TAES paper (Eqs. 22-23).
    k_acc, k_ff:
        Number of sigmas for the 95 % accuracy and the 1e-7 fault-free bound.
    cond_limit:
        Largest accepted condition number of ``G' W G`` for a subset solution.
    """

    phmi_vert: float = 9.8e-8
    phmi_hor: float = 2.0e-9
    p_thres: float = 6.0e-8
    pfa_vert: float = 3.9e-6
    pfa_hor: float = 0.9e-6
    p_emt: float = 1.0e-5
    pl_tol: float = 1.0e-2
    fc_thres: float = 0.01
    n_es_int: float = 25.0
    n_es_cont: float = 25.0
    t_exp: float = 1.0 / 24.0
    exclusion: bool = True
    exclusion_const_threshold: Optional[float] = 1.0e-7
    cap_by_exposure: bool = False
    fast: bool = False
    threshold_method: ThresholdMethod = "solution_separation"
    consolidation: Consolidation = "individual"
    subset_selection: SubsetSelection = "default"
    mode_ordering: ModeOrdering = "geometry"
    min_fault_order: int = 2
    exclusion_allocation: ExclusionAllocation = "equal"
    prior_form: PriorForm = "exact"
    optimize_position: bool = False
    sig_acc_max_vert: float = 1.86
    sig_acc_max_hor1: float = 3.0
    sig_acc_max_hor2: float = 3.0
    chi2_sanity_check: bool = False
    pfa_chi2: float = 1.0e-8
    k_acc: float = 1.96
    k_ff: float = 5.33
    cond_limit: float = 1.0e10

    def __post_init__(self) -> None:
        for name in ("phmi_vert", "phmi_hor", "p_thres", "pfa_vert", "pfa_hor"):
            if not self.__dict__[name] >= 0:
                raise ValueError(f"{name} must be non-negative")
        if self.n_es_int < 1 or self.n_es_cont < 1:
            raise ValueError("n_es_int and n_es_cont must be >= 1")
        if self.t_exp < 0:
            raise ValueError("t_exp must be >= 0")
        if self.pl_tol <= 0:
            raise ValueError("pl_tol must be > 0")

    # ------------------------------------------------------------------ helpers
    @property
    def phmi(self) -> float:
        """Total integrity budget ``PHMI = PHMI_VERT + PHMI_HOR``."""
        return self.phmi_vert + self.phmi_hor

    @property
    def pfa(self) -> float:
        """Total false-alert budget ``P_FA = P_FA_VERT + P_FA_HOR``."""
        return self.pfa_vert + self.pfa_hor

    def replace(self, **changes) -> "ARAIMConfig":
        """Return a copy with some fields changed."""
        return replace(self, **changes)

    # ------------------------------------------------------------------ presets
    @classmethod
    def baseline_2015(cls, **changes) -> "ARAIMConfig":
        """Constants of Table II of the TAES 2015 paper (instantaneous, FD only)."""
        base = dict(
            phmi_vert=9e-8, phmi_hor=1e-8, p_thres=9e-8,
            pfa_vert=3.9e-6, pfa_hor=9e-8, pfa_chi2=1e-8,
            p_emt=1e-5, pl_tol=5e-2, fc_thres=0.0,
            n_es_int=1.0, n_es_cont=1.0, t_exp=0.0,
            exclusion=False, consolidation="none", chi2_sanity_check=True,
            min_fault_order=1, subset_selection="all", prior_form="product",
        )
        base.update(changes)
        return cls(**base)

    @classmethod
    def lpv200(cls, **changes) -> "ARAIMConfig":
        """Vertical ARAIM (LPV-200) with exclusion and a 150 s exposure window."""
        base = dict(
            phmi_vert=9.8e-8, phmi_hor=2e-9, p_thres=6e-8,
            pfa_vert=3.9e-6, pfa_hor=0.9e-6, p_emt=1e-5, pl_tol=1e-2,
            fc_thres=0.01, n_es_int=25, n_es_cont=25, t_exp=1 / 24,
            cap_by_exposure=True, exclusion=True,
        )
        base.update(changes)
        return cls(**base)

    @classmethod
    def rnp_horizontal(cls, **changes) -> "ARAIMConfig":
        """Horizontal ARAIM for RNP: 1 h exposure window, ``N_ES = 360``.

        Matches the "Simulation settings" of Figure 1 of the ION ITM 2022 paper
        (``P_HMI = 1e-7``, ``P_FA = 5e-7``, ``P_THRES = 8e-8``).
        """
        base = dict(
            phmi_vert=1e-9, phmi_hor=9.9e-8, p_thres=8e-8,
            pfa_vert=1e-9, pfa_hor=5e-7, p_emt=1e-5, pl_tol=1e-2,
            fc_thres=0.05, n_es_int=360, n_es_cont=360, t_exp=1.0,
            exclusion=True, cap_by_exposure=False,
        )
        base.update(changes)
        return cls(**base)


@dataclass(frozen=True)
class Criteria:
    """Operational availability criteria (alert limits and accuracy gates)."""

    val: float = float("inf")
    hal: float = float("inf")
    emt: float = float("inf")
    sigma_acc: float = float("inf")
    name: str = "custom"

    # canonical operations -----------------------------------------------------
    @classmethod
    def lpv200(cls) -> "Criteria":
        return cls(val=35.0, hal=40.0, emt=15.0, sigma_acc=1.87, name="LPV-200")

    @classmethod
    def lpv250(cls) -> "Criteria":
        return cls(val=50.0, hal=40.0, name="LPV-250")

    @classmethod
    def rnp(cls, nm: float) -> "Criteria":
        """RNP ``nm`` nautical-mile horizontal containment (HAL = 2 nm for RNP 1...)."""
        return cls(hal=1852.0 * nm, name=f"HAL {nm:g} NM")

    @classmethod
    def horizontal(cls, hal: float) -> "Criteria":
        return cls(hal=hal, name=f"HAL {hal:g} m")
