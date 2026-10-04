import numpy as np
import pytest

from pyaraim import ARAIMConfig
from pyaraim.testing import false_alert_rate, verify_mode_bound


@pytest.mark.parametrize("mode", [0, 5, 18])
@pytest.mark.parametrize("fault", [0.0, 6.0, 15.0])
def test_per_mode_probability_bound_holds(sky, mode, fault):
    cfg = ARAIMConfig.lpv200(exclusion=False)
    c = verify_mode_bound(sky, cfg, mode=mode, fault_bias=fault, n_trials=30000, rng=mode * 7 + int(fault))
    assert c.holds, (c.mode, c.empirical, c.bound)


def test_bound_is_not_vacuous(sky):
    """For the worst fault magnitude the bound is attained to within a small factor (it is tight, not trivial)."""
    cfg = ARAIMConfig.lpv200(exclusion=False)
    ratios = []
    for fb in (3, 6, 9, 12, 15):
        c = verify_mode_bound(sky, cfg, mode=0, fault_bias=fb, n_trials=30000, rng=fb)
        assert c.holds
        ratios.append(c.empirical / c.bound)
    assert max(ratios) > 0.1


def test_two_sided_bound_holds(sky):
    cfg = ARAIMConfig.lpv200(exclusion=False)
    c = verify_mode_bound(sky, cfg, mode=3, fault_bias=0.0, n_trials=30000, rng=4, two_sided=True)
    assert c.holds


def test_false_alert_rate_below_budget_with_inflated_pfa(sky):
    cfg = ARAIMConfig.lpv200(exclusion=False, pfa_vert=0.02, pfa_hor=0.02, n_es_cont=1)
    emp, upper, analytic = false_alert_rate(sky, cfg, n_trials=20000, rng=0)
    assert emp <= analytic * 1.05 + 3e-3
    assert analytic == pytest.approx(0.04, rel=1e-6)
    assert emp > 0.0           # rate is observable at this budget


def test_false_alert_chi2_method_matches_budget(sky):
    cfg = ARAIMConfig.lpv200(exclusion=False, pfa_vert=0.02, pfa_hor=0.02, n_es_cont=1,
                             threshold_method="chi2")
    rng = np.random.default_rng(0)
    y = rng.normal(size=(40000, sky.n)) * np.sqrt(sky.var_acc)
    from pyaraim.stats import chi2_threshold
    G, w = sky.G, 1 / sky.var_acc
    W = np.diag(w)
    P = W - W @ G @ np.linalg.solve(G.T @ W @ G, G.T @ W)
    chi2 = np.einsum("ti,ij,tj->t", y, P, y)
    thr = chi2_threshold(cfg.pfa, sky.n - 3 - sky.n_const)
    assert (chi2 > thr).mean() == pytest.approx(cfg.pfa, abs=4e-3)
