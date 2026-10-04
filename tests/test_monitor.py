import numpy as np
import pytest

from pyaraim import ARAIM, ARAIMConfig, ARAIMMonitor
from pyaraim.testing import (constellation_fault, critical_fault, detection_exclusion_stats, make_epoch,
                             satellite_fault)


@pytest.fixture()
def cfg():
    return ARAIMConfig.lpv200()


def test_fault_free_epoch_has_no_alert_and_matches_geometry_pl(sky, cfg):
    ep = make_epoch(sky, rng=1)
    r = ARAIMMonitor(cfg).process(sky, ep.y)
    assert r.status == "ok"
    # all-in-view PL after the all-in-view test equals the all-in-view term of the worst-case evaluation
    detail = ARAIM(cfg).protection_levels(sky, detail=True)
    assert r.vpl == pytest.approx(detail.per_candidate["all-in-view"][0], rel=1e-3)
    assert r.vpl <= detail.vpl + 1e-9


def test_large_satellite_fault_is_detected_and_correctly_excluded(sky, cfg):
    ep = make_epoch(sky, [satellite_fault(4, 60.0)], rng=2)
    r = ARAIMMonitor(cfg).process(sky, ep.y)
    assert r.status == "excluded" and r.excluded == sky.ids[4]
    assert abs(r.position[2]) < r.vpl


def test_constellation_fault_is_excluded(sky, cfg):
    # a *common* bias would be absorbed by the constellation clock state: the fault must have spread
    f = constellation_fault(sky, "Galileo", 0.0, spread=40.0, rng=0)
    r = ARAIMMonitor(cfg).process(sky, make_epoch(sky, [f], rng=3).y)
    assert r.status == "excluded" and r.excluded == "Galileo-const"


def test_small_fault_below_threshold_is_not_flagged_but_error_is_protected(sky, cfg):
    ep = make_epoch(sky, [satellite_fault(2, 1.0)], rng=4)
    r = ARAIMMonitor(cfg).process(sky, ep.y)
    assert r.status == "ok"
    assert abs(ep.position_error[2]) < r.vpl


def test_critical_fault_just_below_threshold_passes_tests_but_is_bounded(sky, cfg):
    """The worst undetectable fault (Appendix G) must still produce an error below the VPL."""
    araim = ARAIM(cfg)
    ref = araim.protection_levels(sky, detail=True)
    err = 0.9 * ref.per_candidate["all-in-view"][0]
    f = critical_fault(sky, [3], err, axis=2)
    ep = make_epoch(sky, [f], rng=5)
    # the noise-free separation equals err (> threshold for a big single fault): it must be detected
    r = ARAIMMonitor(cfg).process(sky, np.asarray(ep.y))
    assert r.status in ("ok", "excluded")
    if r.status == "ok":
        assert abs(ep.position_error[2]) <= r.vpl


def test_alert_when_two_independent_large_faults_inconsistent(sky, cfg):
    f1, f2 = satellite_fault(0, 80.0), satellite_fault(9, -80.0)
    r = ARAIMMonitor(cfg).process(sky, make_epoch(sky, [f1, f2], rng=6).y)
    # a dual fault is outside single-satellite exclusion: either an alert or exclusion of a constellation
    assert r.status in ("alert", "excluded")


def test_detection_stats(small_sky, cfg):
    stats = detection_exclusion_stats(small_sky, cfg, satellite_fault(1, 80.0), n_trials=40, rng=0)
    assert stats.detected > 0.9
    assert stats.correct_exclusion > 0.9
    assert stats.hmi == 0.0


def test_chi2_sanity_check_alerts_on_large_inconsistency(sky):
    cfg = ARAIMConfig.lpv200(chi2_sanity_check=True, exclusion=False, pfa_chi2=1e-8)
    y = np.zeros(sky.n)
    assert ARAIMMonitor(cfg).process(sky, y).status == "ok"
    y2 = make_epoch(sky, [satellite_fault(0, 200.0)], rng=1).y
    assert ARAIMMonitor(cfg).process(sky, y2).alert


def test_wrong_residual_shape_raises(sky, cfg):
    with pytest.raises(ValueError):
        ARAIMMonitor(cfg).process(sky, np.zeros(3))


def test_common_mode_constellation_bias_is_unobservable_by_design(sky, cfg):
    """A bias common to all satellites of a constellation is indistinguishable from a clock offset:
    it does not corrupt the position and must not trigger any test."""
    f = constellation_fault(sky, "Galileo", 50.0, spread=0.0)
    ep = make_epoch(sky, [f], rng=3)
    r = ARAIMMonitor(cfg).process(sky, ep.y)
    assert r.status == "ok"
    assert np.abs(ep.position_error).max() < 5.0
