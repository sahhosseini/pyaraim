import numpy as np
import pytest
from scipy import optimize

from pyaraim.protection import integrity_targets, solve_protection_level
from pyaraim.config import ARAIMConfig
from pyaraim.stats import Q, Qbar, Qinv, chi2_threshold


def test_q_function_values():
    assert Q(0.0) == pytest.approx(0.5)
    assert Q(1.959964) == pytest.approx(0.025, rel=1e-5)
    assert Qinv(0.025) == pytest.approx(1.959964, rel=1e-6)


def test_qinv_inverts_q_over_many_decades():
    p = np.logspace(-14, -0.1, 40)
    np.testing.assert_allclose(Q(Qinv(p)), p, rtol=1e-9)


def test_qbar_is_one_for_nonpositive_argument():
    assert Qbar(-3.0) == 1.0 and Qbar(0.0) == 1.0
    assert Qbar(2.0) == pytest.approx(Q(2.0))


def test_chi2_threshold_matches_tail_probability():
    from scipy.stats import chi2
    t = chi2_threshold(1e-8, 7)
    assert chi2.sf(t, 7) == pytest.approx(1e-8, rel=1e-6)
    assert chi2_threshold(1e-8, 0) == np.inf


def _brute(sigma, off, coef, target, cap):
    f = lambda pl: np.sum(coef * np.minimum(Qbar((pl - off) / sigma), cap)) - target
    return optimize.brentq(f, 0.0, 1e4, xtol=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_pl_solver_matches_root_finder(seed):
    rng = np.random.default_rng(seed)
    m = 12
    sigma = rng.uniform(0.5, 4.0, m)
    off = rng.uniform(0.0, 10.0, m)
    off[0] = rng.uniform(0, 1)
    coef = np.concatenate([[2.0], rng.uniform(1e-8, 1e-4, m - 1)])
    cap = np.ones(m)
    target = 1e-7
    exact = _brute(sigma, off, coef, target, cap)
    got = solve_protection_level(sigma, off, coef, target, cap, tol=1e-3)
    assert exact <= got <= exact + 1.1e-3          # the upper end of the bracket is returned


def test_pl_solver_with_caps():
    rng = np.random.default_rng(1)
    m = 8
    sigma = rng.uniform(1, 3, m)
    off = rng.uniform(0, 8, m)
    coef = np.concatenate([[2.0], rng.uniform(1e-6, 1e-4, m - 1)])
    cap = np.concatenate([[1.0], rng.uniform(1e-3, 1e-1, m - 1)])
    exact = _brute(sigma, off, coef, 1e-7, cap)
    got = solve_protection_level(sigma, off, coef, 1e-7, cap, tol=1e-4)
    assert exact <= got <= exact + 2e-4
    # caps can only lower the PL
    assert got <= solve_protection_level(sigma, off, coef, 1e-7, None, tol=1e-4) + 1e-9


def test_pl_monotone_in_sigma_and_target():
    sigma = np.array([1.0, 2.0, 2.5])
    off = np.array([0.0, 4.0, 5.0])
    coef = np.array([2.0, 1e-4, 1e-5])
    base = solve_protection_level(sigma, off, coef, 1e-7)
    assert solve_protection_level(sigma * 1.2, off, coef, 1e-7) > base
    assert solve_protection_level(sigma, off, coef, 1e-8) > base
    assert solve_protection_level(sigma, off, coef, 1e-6) < base


def test_pl_infinite_when_target_exhausted():
    assert solve_protection_level(np.ones(2), np.zeros(2), np.array([2.0, 1.0]), 0.0) == np.inf


def test_integrity_targets_follow_eq21_eq23():
    cfg = ARAIMConfig(phmi_vert=9e-8, phmi_hor=1e-8, n_es_int=10)
    tv, th = integrity_targets(cfg, rho=0.25, p_not_monitored=2e-8)
    assert tv == pytest.approx(0.25 * 9e-8 / 10 * (1 - 2e-8 / 1e-7))
    assert th == pytest.approx(0.25 * 1e-8 / 20 * (1 - 2e-8 / 1e-7))
    assert integrity_targets(cfg, 1.0, 2e-7) == (0.0, 0.0)
