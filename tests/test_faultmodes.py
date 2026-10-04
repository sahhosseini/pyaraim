import numpy as np
import pytest

from pyaraim import ARAIMConfig
from pyaraim.faultmodes import (determine_fault_modes, determine_nmax, p_multiple_bound,
                                unique_masks)
from pyaraim import appendix_j


def test_appendix_c_worked_example():
    # 18 satellites + 2 constellations, all P = 1e-4: P(>=3 events) bound = 1.33e-9 (Eq. 10 of the TAES paper)
    p = np.full(20, 1e-4)
    assert (p.sum() ** 3 / 6) == pytest.approx(1.33e-9, rel=1e-2)
    assert p_multiple_bound(p, 3) < 1.4e-9
    assert determine_nmax(p, 9e-8) == 2


@pytest.mark.parametrize("pev,nev,expected", [
    (1e-5, 10, 1), (1e-5, 40, 1), (1e-4, 10, 2), (1e-4, 40, 2),
    (5e-4, 10, 2), (5e-4, 15, 2), (5e-4, 20, 3), (5e-4, 40, 3),
    (1e-3, 10, 3), (1e-3, 35, 3), (1e-3, 40, 4)])
def test_table_c1_nfault_max(pev, nev, expected):
    # Table C.I of the TAES paper (P_THRES = 9e-8)
    assert determine_nmax(np.full(nev, pev), 9e-8) == expected


def test_exact_probabilities_for_small_r():
    rng = np.random.default_rng(0)
    p = rng.uniform(1e-3, 5e-2, 6)
    # brute-force the exact probability of >= r events
    import itertools
    def exact(r):
        tot = 0.0
        for bits in itertools.product([0, 1], repeat=len(p)):
            if sum(bits) >= r:
                tot += np.prod([pi if b else 1 - pi for pi, b in zip(p, bits)])
        return tot
    for r in (1, 2, 3):
        assert p_multiple_bound(p, r) == pytest.approx(exact(r), rel=1e-9)
    assert p_multiple_bound(p, 4) >= exact(4)             # Eq. (74) is an upper bound


def test_unique_masks_merges_and_sums():
    m = np.array([[1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1]], bool)
    u, w, first, inv = unique_masks(m, np.array([1.0, 2.0, 4.0, 8.0]))
    assert u.shape == (3, 3)
    np.testing.assert_allclose(w, [5.0, 2.0, 8.0])
    np.testing.assert_array_equal(first, [0, 1, 3])


def test_modes_are_unique_and_probabilities_sensible(sky):
    cfg = ARAIMConfig.lpv200()
    fm = determine_fault_modes(sky, cfg)
    assert len(np.unique(fm.masks, axis=0)) == len(fm)
    assert np.all(fm.p_inst > 0) and np.all(fm.p_exp >= fm.p_inst - 1e-30)
    # every single satellite fault is monitored
    singles = {tuple(np.where(~m)[0]) for m in fm.masks if (~m).sum() == 1}
    assert len(singles) == sky.n
    assert 0 <= fm.p_not_monitored <= cfg.p_thres * 1.0000001 + 1.0


def test_unmonitored_probability_below_threshold(sky):
    cfg = ARAIMConfig.lpv200()
    fm = determine_fault_modes(sky, cfg)
    assert fm.p_not_monitored <= cfg.p_thres


def test_consolidation_reduces_number_of_modes(sky):
    n_none = len(determine_fault_modes(sky, ARAIMConfig.lpv200(consolidation="none")))
    n_ind = len(determine_fault_modes(sky, ARAIMConfig.lpv200(consolidation="individual", fc_thres=0.05)))
    assert n_ind <= n_none


def test_consolidation_preserves_total_probability(sky):
    """Absorbed modes are moved, not lost: sum of p_exp is unchanged by consolidation if all are kept."""
    cfg0 = ARAIMConfig.lpv200(consolidation="none", subset_selection="all", p_thres=0.0)
    cfg1 = ARAIMConfig.lpv200(consolidation="individual", subset_selection="all", p_thres=0.0, fc_thres=0.5)
    f0, f1 = determine_fault_modes(sky, cfg0), determine_fault_modes(sky, cfg1)
    assert f1.p_exp.sum() == pytest.approx(f0.p_exp.sum(), rel=1e-9)


def test_appendix_j_mode_count_with_all_selection():
    sats = appendix_j.satellite_set()
    fm = determine_fault_modes(sats, appendix_j.config())
    assert fm.n_fault_max == 2
    # 10 single + 45 dual + 2 constellations + 10 (one constellation + one satellite of the other) + 1 (both)
    assert len(fm) == 68


def test_degree_and_quasi_fixed_selection_run(sky):
    for sel in ("default", "quasi_fixed"):
        fm = determine_fault_modes(sky, ARAIMConfig.rnp_horizontal(subset_selection=sel))
        assert len(fm) > sky.n
    fm = determine_fault_modes(sky, ARAIMConfig.rnp_horizontal(mode_ordering="degree"))
    assert len(fm) > 0
