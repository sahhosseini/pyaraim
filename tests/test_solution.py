from itertools import combinations

import numpy as np
import pytest

from pyaraim.solution import (downdate_coefficients, dual_fault_bound, subset_solutions, weighted_ls,
                              wls_coefficients)


def _problem(sky):
    return sky.G, sky.var_int, sky.var_acc, sky.b_int, sky.b_acc


def test_wls_matches_textbook_formula(sky):
    G, vi, *_ = _problem(sky)
    w = 1 / vi
    mask = np.ones((1, sky.n), bool)
    s, ok = wls_coefficients(G, w, mask)
    S = np.linalg.solve(G.T @ (w[:, None] * G), G.T * w)
    np.testing.assert_allclose(s[0], S[:3], atol=1e-12)
    assert ok[0]


def test_subset_removal_ignores_removed_measurements(sky):
    G, vi, *_ = _problem(sky)
    mask = np.ones((2, sky.n), bool)
    mask[1, [2, 9]] = False
    s, _ = wls_coefficients(G, 1 / vi, mask)
    assert np.all(s[1][:, [2, 9]] == 0)


def test_constellation_removal_drops_clock_column(sky):
    G, vi, *_ = _problem(sky)
    mask = np.ones((2, sky.n), bool)
    mask[1] = sky.constellation != 1
    s, ok = wls_coefficients(G, 1 / vi, mask)
    assert ok[1]
    # equals the solution of the problem built only with the remaining constellation
    sub = sky.select(sky.constellation != 1)
    s2, _ = wls_coefficients(sub.G, 1 / sub.var_int, np.ones((1, sub.n), bool))
    np.testing.assert_allclose(s[1][:, sky.constellation != 1], s2[0], atol=1e-10)


def test_unobservable_subset_flagged(small_sky):
    G, vi, va, bi, ba = _problem(small_sky)
    masks = np.ones((2, small_sky.n), bool)
    masks[1, :] = False
    masks[1, :3] = True
    sol = subset_solutions(G, vi, va, bi, ba, masks)
    assert sol.observable[0] and not sol.observable[1]
    assert np.isinf(sol.sigma[1]).all()


def test_eq37_sigma_ss_identity_when_models_coincide(sky):
    """With C_acc = C_int the separation variance is sigma_k^2 - sigma_0^2 (Eq. 37)."""
    G, vi, *_ = _problem(sky)
    masks = np.ones((1 + sky.n, sky.n), bool)
    for i in range(sky.n):
        masks[1 + i, i] = False
    sol = subset_solutions(G, vi, vi, 0.0, 0.0, masks)
    np.testing.assert_allclose(sol.sigma_ss[1:] ** 2, sol.sigma[1:] ** 2 - sol.sigma[0:1] ** 2, rtol=1e-9)


def test_nominal_bias_formula(sky):
    G, vi, va, bi, ba = _problem(sky)
    b = np.full(sky.n, 0.5)
    masks = np.ones((2, sky.n), bool)
    masks[1, 4] = False
    sol = subset_solutions(G, vi, va, b, b, masks)
    np.testing.assert_allclose(sol.bias, np.abs(sol.s).sum(axis=2) * 0.5, rtol=1e-12)


@pytest.mark.parametrize("r", [1, 2])
def test_rank_update_equals_direct_solution(sky, r):
    G, vi, *_ = _problem(sky)
    removed = np.array(list(combinations(range(sky.n), r)))
    masks = np.ones((len(removed), sky.n), bool)
    for k, J in enumerate(removed):
        masks[k, J] = False
    s_direct, _ = wls_coefficients(G, 1 / vi, masks)
    s_upd, ok = downdate_coefficients(G, 1 / vi, removed)
    assert ok.all()
    np.testing.assert_allclose(s_upd, s_direct, atol=1e-10)


def test_dual_fault_bound_is_rigorous(sky):
    G, vi, va, bi, ba = _problem(sky)
    bi = np.full(sky.n, 0.4)
    ba = np.full(sky.n, 0.2)
    pairs = np.array(list(combinations(range(sky.n), 2)))
    masks = np.ones((1 + len(pairs), sky.n), bool)
    for k, (i, j) in enumerate(pairs):
        masks[1 + k, [i, j]] = False
    sol = subset_solutions(G, vi, va, bi, ba, masks)
    b = dual_fault_bound(G, vi, va, bi, ba, pairs)
    assert b is not None and b.valid.all()
    ok = sol.observable[1:]
    assert (sol.sigma[1:][ok] <= b.sigma + 1e-9).all()
    assert (sol.sigma_ss[1:][ok] <= b.sigma_ss + 1e-9).all()
    assert (sol.bias[1:][ok] <= b.bias + 1e-9).all()
    assert (sol.bias_ss[1:][ok] <= b.bias_ss + 1e-9).all()


def test_weighted_ls_chi2_matches_projection_formula(sky):
    G, vi, va, *_ = _problem(sky)
    rng = np.random.default_rng(5)
    y = rng.normal(size=sky.n) * np.sqrt(va)
    w = 1 / va
    x, chi2, dof = weighted_ls(G, w, np.ones(sky.n, bool), y)
    W = np.diag(w)
    P = W - W @ G @ np.linalg.solve(G.T @ W @ G, G.T @ W)
    assert chi2 == pytest.approx(y @ P @ y, rel=1e-9)
    assert dof == sky.n - 3 - sky.n_const


def test_eq127_position_difference_for_one_satellite_out(sky):
    """Appendix I, Eq. (127): x - x_(-i) = (G'WG)^-1 g_i w_i (y_i - g_i' x) / (1 - g_i' w_i (G'WG)^-1 g_i)."""
    G, vi, *_ = _problem(sky)
    w = 1 / vi
    rng = np.random.default_rng(2)
    y = rng.normal(size=sky.n)
    i = 3
    mask = np.ones((2, sky.n), bool)
    mask[1, i] = False
    s, _ = wls_coefficients(G, w, mask)
    A = np.linalg.inv(G.T @ (w[:, None] * G))
    x = A @ G.T @ (w * y)
    full_pos = (s[0] @ y)
    sub_pos = s[1] @ y
    g = G[i]
    d = (A @ g * w[i] * (y[i] - g @ x)) / (1 - g @ A @ g * w[i])
    np.testing.assert_allclose(full_pos - sub_pos, d[:3], atol=1e-10)
