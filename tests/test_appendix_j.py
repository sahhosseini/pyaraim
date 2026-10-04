"""Regression against the numerical example of Appendix J of the TAES 2015 paper."""
import numpy as np
import pytest

from pyaraim import ARAIM, appendix_j as aj
from pyaraim.araim import build_terms
from pyaraim.faultmodes import FaultModeList


@pytest.fixture(scope="module")
def result():
    return ARAIM(aj.config()).protection_levels(aj.satellite_set())


def test_vpl(result):
    assert result.vpl == pytest.approx(aj.PUBLISHED["vpl"], abs=0.15)


def test_hpl(result):
    assert result.hpl == pytest.approx(aj.PUBLISHED["hpl"], abs=0.3)


def test_emt(result):
    assert result.emt == pytest.approx(aj.PUBLISHED["emt"], abs=0.1)


def test_accuracy_sigma(result):
    assert result.sigma_acc == pytest.approx(aj.PUBLISHED["sigma_acc"], abs=0.01)


def test_constellation_mode_statistics_match_paper_eq132():
    sats = aj.satellite_set()
    cfg = aj.config()
    ARAIM(cfg)
    modes = ARAIM(cfg).fault_modes(sats)
    t = build_terms(sats, np.ones(10, bool), modes, cfg, keep_solutions=True)
    sol = t.solutions
    for j, (sig, sig_ss) in enumerate(zip(aj.PUBLISHED["sigma3_const"], aj.PUBLISHED["sigma_ss3_const"])):
        mask = sats.constellation != j
        k = [i for i, m in enumerate(sol.masks) if np.array_equal(m, mask)][0]
        assert sol.sigma[k, 2] == pytest.approx(sig, abs=1e-3)
        assert sol.sigma_ss[k, 2] == pytest.approx(sig_ss, abs=1e-3)


def test_false_alert_containment_with_57_modes_matches_eq133():
    from pyaraim.stats import Qinv
    assert Qinv(3.9e-6 / (2 * 57)) == pytest.approx(aj.PUBLISHED["k_fa3"], abs=1e-3)


def test_paper_mode_list_gives_close_vpl():
    """The paper's list (n-1, n-2 and the two constellation modes: 57 modes) gives the same VPL within ~0.3 m."""
    sats, cfg = aj.satellite_set(), aj.config()
    modes = ARAIM(cfg).fault_modes(sats)
    rem = (~modes.masks).sum(axis=1)
    keep = [i for i, m in enumerate(modes.masks)
            if m.sum() >= 8 or (rem[i] == 5 and (m[:5].sum() == 0 or m[5:].sum() == 0))]
    assert len(keep) == 57
    fm = FaultModeList.from_masks(modes.masks[keep], modes.p_inst[keep], modes.p_exp[keep], 1.1e-8)
    t = build_terms(sats, np.ones(10, bool), fm, cfg)
    vpl, e, n = t.solve(cfg)
    assert t.n_modes == 57
    assert vpl == pytest.approx(19.10, abs=0.1)
