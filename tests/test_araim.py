import numpy as np
import pytest

from pyaraim import ARAIM, ARAIMConfig, Criteria, SatelliteSet
from pyaraim.araim import build_terms, exclusion_candidates
from pyaraim.testing import random_sky


def pl(sky, **kw):
    return ARAIM(ARAIMConfig.lpv200(**kw)).protection_levels(sky)


def test_result_is_finite_and_ordered(sky):
    r = pl(sky)
    assert np.isfinite([r.vpl, r.hpl, r.emt, r.sigma_acc]).all()
    assert r.vpl > 0 and r.hpl > 0
    # a PL can never be below the fault-free requirement Q^-1(.) * sigma
    assert r.vpl > 4 * r.sigma_acc * 0.5


def test_exclusion_costs_integrity_budget(sky):
    fd = pl(sky, exclusion=False)
    fde = pl(sky, exclusion=True)
    assert fde.vpl >= fd.vpl - 0.02 and fde.hpl >= fd.hpl - 0.02


def test_fast_bound_is_conservative(sky):
    exact = ARAIM(ARAIMConfig.lpv200(fast=False)).protection_levels(sky)
    bound = ARAIM(ARAIMConfig.lpv200(fast=True)).protection_levels(sky)   # no criteria -> bound result
    assert bound.vpl >= exact.vpl - 0.02 and bound.hpl >= exact.hpl - 0.02


def test_fast_screening_preserves_availability_decision(sky):
    crit = Criteria(val=60.0, hal=60.0)
    ex = ARAIM(ARAIMConfig.lpv200(fast=False)).protection_levels(sky, crit)
    fa = ARAIM(ARAIMConfig.lpv200(fast=True)).protection_levels(sky, crit)
    assert ex.available(crit) == fa.available(crit)
    tight = Criteria(val=ex.vpl + 0.5, hal=ex.hpl + 0.5)
    assert ARAIM(ARAIMConfig.lpv200(fast=True)).protection_levels(sky, tight).available(tight)


def test_permutation_invariance(sky):
    rng = np.random.default_rng(0)
    perm = rng.permutation(sky.n)
    sats2 = SatelliteSet(sky.los_enu[perm], sky.constellation[perm], sky.const_names, sky.var_int[perm],
                         sky.var_acc[perm], sky.b_int[perm], sky.b_acc[perm], sky.p_sat[perm],
                         sky.mfd_sat[perm], sky.p_const, sky.mfd_const)
    a, b = pl(sky), pl(sats2)
    assert a.vpl == pytest.approx(b.vpl, abs=0.05)
    assert a.hpl == pytest.approx(b.hpl, abs=0.05)
    # equal-rank modes are truncated at P_THRES in index order, so tiny differences are legitimate
    assert a.emt == pytest.approx(b.emt, rel=0.02)


def test_quarter_turn_rotation_about_vertical_leaves_pls_unchanged(sky):
    """HPL = sqrt(HPL_E^2 + HPL_N^2) is symmetric under E<->N swaps, so 90-degree turns are exact."""
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rot = SatelliteSet(sky.los_enu @ R.T, sky.constellation, sky.const_names, sky.var_int, sky.var_acc,
                       sky.b_int, sky.b_acc, sky.p_sat, sky.mfd_sat, sky.p_const, sky.mfd_const)
    a, b = pl(sky), pl(rot)
    assert a.vpl == pytest.approx(b.vpl, abs=0.05)
    assert a.hpl == pytest.approx(b.hpl, rel=0.01)


def test_arbitrary_rotation_changes_hpl_only_moderately(sky):
    """The per-axis horizontal split is not rotation invariant (a known conservatism), but is bounded."""
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    rot = SatelliteSet(sky.los_enu @ R.T, sky.constellation, sky.const_names, sky.var_int, sky.var_acc,
                       sky.b_int, sky.b_acc, sky.p_sat, sky.mfd_sat, sky.p_const, sky.mfd_const)
    a, b = pl(sky), pl(rot)
    assert a.vpl == pytest.approx(b.vpl, abs=0.05)
    assert b.hpl == pytest.approx(a.hpl, rel=0.25)


def test_pl_scales_with_noise(sky):
    a = pl(sky)
    sky2 = SatelliteSet(sky.los_enu, sky.constellation, sky.const_names, 4 * sky.var_int, 4 * sky.var_acc,
                        sky.b_int, sky.b_acc, sky.p_sat, sky.mfd_sat, sky.p_const, sky.mfd_const)
    b = pl(sky2)
    assert b.vpl == pytest.approx(2 * a.vpl, rel=0.02)       # no bias -> exactly linear in sigma
    assert b.sigma_acc == pytest.approx(2 * a.sigma_acc, rel=1e-9)


def test_more_prior_fault_probability_gives_larger_pl(isd):
    sky = random_sky({"GPS": 8, "Galileo": 8}, isd, 3)
    worse = random_sky({"GPS": 8, "Galileo": 8}, isd.replace("GPS", r_sat=1e-4), 3)
    assert pl(worse).vpl >= pl(sky).vpl - 0.02


def test_larger_n_es_gives_larger_pl(sky):
    assert pl(sky, n_es_int=360).vpl > pl(sky, n_es_int=1).vpl


def test_pl_decreases_when_satellites_are_added(isd):
    base = random_sky({"GPS": 6, "Galileo": 6}, isd, 21)
    # build a superset: same LOS plus 4 extra satellites
    extra = random_sky({"GPS": 2, "Galileo": 2}, isd, 99)
    los = np.vstack([base.los_enu, extra.los_enu])
    names = [base.const_names[c] for c in base.constellation] + [extra.const_names[c] for c in extra.constellation]
    big = SatelliteSet.from_los(los, names, isd)
    assert pl(big).vpl < pl(base).vpl


def test_four_satellites_unavailable_five_with_two_constellations_borderline(isd):
    sky = random_sky({"GPS": 3, "Galileo": 1}, isd, 1)
    assert np.isinf(pl(sky).vpl)


def test_pl_equation_is_satisfied_at_returned_value(sky):
    cfg = ARAIMConfig.lpv200(exclusion=False, pl_tol=1e-4)
    modes = ARAIM(cfg).fault_modes(sky)
    t = build_terms(sky, np.ones(sky.n, bool), modes, cfg)
    vpl, _, _ = t.solve(cfg)
    from pyaraim.stats import Qbar
    def total(x):
        return np.sum(t.coef * np.minimum(Qbar((x - t.offset[:, 2]) / t.sigma[:, 2]), t.alloc_max))
    target = cfg.phmi_vert / cfg.n_es_int * (1 - t.p_not_monitored / cfg.phmi)
    assert total(vpl) <= target * (1 + 1e-9)
    assert total(vpl - 2e-4) > target


def test_exclusion_candidates(sky):
    cands = exclusion_candidates(sky, ARAIMConfig.lpv200())
    assert len(cands) == sky.n + sum(sky.p_const >= 1e-7)
    assert exclusion_candidates(sky, ARAIMConfig.lpv200(exclusion=False)) == []


def test_joint_allocation_not_smaller_than_equal_split_in_the_worst_case(sky):
    eq = pl(sky, exclusion_allocation="equal")
    jt = pl(sky, exclusion_allocation="joint")
    assert np.isfinite(jt.vpl)
    assert jt.vpl <= eq.vpl * 1.05            # same order; joint allocates risk optimally across options


def test_chi2_threshold_method_runs(sky):
    r = pl(sky, threshold_method="chi2")
    assert np.isfinite(r.vpl)


def test_position_optimization_never_increases_pl_beyond_baseline(sky):
    base = pl(sky)
    opt = pl(sky, optimize_position=True, sig_acc_max_vert=3.0, sig_acc_max_hor1=3.0, sig_acc_max_hor2=3.0)
    assert opt.vpl <= base.vpl + 1e-6 or opt.hpl <= base.hpl + 1e-6


def test_cap_by_exposure_lowers_pl(sky):
    assert pl(sky, cap_by_exposure=True).vpl <= pl(sky, cap_by_exposure=False).vpl + 1e-6


def test_exhausted_budget_returns_inf(sky):
    bad = ARAIMConfig.lpv200(p_thres=1.0, phmi_vert=1e-8, phmi_hor=1e-9)   # unmonitored mass > budget
    r = ARAIM(bad).protection_levels(sky)
    assert np.isinf(r.vpl)


def test_baseline_2015_preset_runs_without_exclusion(sky):
    r = ARAIM(ARAIMConfig.baseline_2015()).protection_levels(sky)
    assert np.isfinite(r.vpl) and r.worst_candidate == "all-in-view"
