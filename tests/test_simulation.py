import numpy as np
import pytest

from pyaraim import ARAIMConfig, Criteria, ISD
from pyaraim.constants import MU_EARTH
from pyaraim.simulation import (AvailabilityResult, BUNDLED_ALMANACS, Scenario, UserGrid, bundled_almanac,
                                constellation_from_almanacs, epoch_satellite_sets, geodetic_to_ecef,
                                enu_basis, run_availability)
from pyaraim.simulation.availability import QUANT


@pytest.mark.parametrize("name", sorted(BUNDLED_ALMANACS))
def test_bundled_almanacs_parse(name):
    alm = bundled_almanac(name)
    assert 23 <= len(alm) <= 27
    assert all(e.sqrt_a > 5000 for e in alm.entries)


def test_orbit_geometry_is_physical():
    gps = bundled_almanac("gps_24")
    p = gps.positions_ecef(np.arange(0, 43200, 600.0))
    r = np.linalg.norm(p, axis=-1)
    assert r.min() > 2.6e7 and r.max() < 2.7e7
    # inertial period = 2 pi sqrt(a^3/mu)  ~ 11 h 58 min; ECEF radius is constant for circular orbits
    a = gps.entries[0].sqrt_a ** 2
    assert 2 * np.pi * np.sqrt(a**3 / MU_EARTH) == pytest.approx(43082, abs=60)
    assert np.ptp(r[:, 0]) < 1.0
    # latitude bounded by the inclination
    inc = gps.entries[0].inc
    lat = np.arcsin(p[..., 2] / r)
    assert np.abs(lat).max() <= inc + 1e-6


def test_walker_like_visibility_counts():
    sat = bundled_almanac("gps_24")
    grid = UserGrid.points([45.0], [10.0])
    scn = Scenario([sat], ISD.paper_2022(), None, grid, mask_deg=5.0)
    n = [s.n for t in range(0, 86400, 1800) for _, s in epoch_satellite_sets(scn, float(t))]
    assert 5 <= min(n) and max(n) <= 14           # GPS 24: 6-12 typical visible at 5 deg


def test_geodetic_conversions_and_enu_frame():
    p = geodetic_to_ecef(0.0, 0.0, 0.0)
    np.testing.assert_allclose(p, [6378137.0, 0, 0])
    b = enu_basis(0.0, 0.0)
    np.testing.assert_allclose(b, [[0, 1, 0], [0, 0, 1], [1, 0, 0]], atol=1e-15)
    pole = geodetic_to_ecef(90.0, 0.0, 0.0)
    assert pole[2] == pytest.approx(6356752.3142, abs=1e-3)
    R = enu_basis(37.0, -122.0)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_grid_weights_and_regions():
    g = UserGrid.world(30, 60)
    assert np.all(np.abs(g.lat) < 90)
    assert np.all(g.weight > 0)
    conus = UserGrid.region("conus", 5, 5)
    assert conus.lat.min() >= 24 and conus.lon.max() <= -66


def test_los_vectors_are_unit_and_above_horizon():
    g = UserGrid.points([34.0], [-118.0])
    sat = bundled_almanac("gps_24")
    pos = sat.positions_ecef(0.0)
    los, rng = g.los_enu(0, pos)
    np.testing.assert_allclose(np.linalg.norm(los, axis=1), 1.0, atol=1e-12)
    assert (rng > 1.9e7).all() and (rng < 5.3e7).all()


def _tiny_result():
    cfg = ARAIMConfig.lpv200(fast=True)
    scn = Scenario(constellation_from_almanacs("gps_24", "gal_24"), ISD.paper_2022(), cfg,
                   UserGrid.points([34.0, -20.0, 60.0], [-118.0, 20.0, 10.0]),
                   times=np.arange(0, 3600.0, 1800.0), criteria=Criteria.lpv200())
    return run_availability(scn)


def test_availability_run_shapes_and_values():
    res = _tiny_result()
    assert res.vpl.shape == (3, 2)
    assert np.isfinite(res.vpl).all() and (res.n_sat >= 10).all()
    assert (res.hpl < res.vpl * 3).all()
    crit = Criteria.lpv200()
    av = res.availability(crit)
    assert av.shape == (3,) and ((av >= 0) & (av <= 1)).all()


def test_result_roundtrip_and_csv(tmp_path):
    res = _tiny_result()
    f = tmp_path / "r.npz"
    res.save(f)
    back = AvailabilityResult.load(f)
    for q in QUANT:
        np.testing.assert_array_equal(getattr(back, q), getattr(res, q))
    assert back.meta["n_users"] == "3"
    res.to_csv(tmp_path / "r.csv", Criteria.lpv200())
    assert (tmp_path / "r.csv").read_text().splitlines()[0].startswith("lat,lon,vpl_p99.9")


def test_coverage_definition_cos_weighted():
    lat = np.array([0.0, 60.0])
    res = AvailabilityResult(lat, np.zeros(2), np.arange(4.0), np.array([[10.0] * 4, [10.0, 10, 10, 99]]),
                             np.full((2, 4), 5.0), np.zeros((2, 4)), np.zeros((2, 4)), np.full((2, 4), 8))
    crit = Criteria(val=35, hal=40)
    np.testing.assert_allclose(res.availability(crit), [1.0, 0.75])
    assert res.coverage(crit, 0.999) == pytest.approx(100 * 1.0 / (1.0 + 0.5))
    assert res.coverage(crit, 0.70) == pytest.approx(100.0)


def test_percentile_handles_infinite_epochs():
    vpl = np.array([[1.0, 2.0, np.inf, 3.0]])
    res = AvailabilityResult(np.zeros(1), np.zeros(1), np.arange(4.0), vpl, vpl, vpl, vpl, np.ones((1, 4)))
    assert res.percentile("vpl", 50)[0] in (2.0, 3.0)
    assert np.isinf(res.percentile("vpl", 99.9)[0])


def test_outage_degrades_or_equals_availability_metric():
    from pyaraim import ARAIM
    from pyaraim.testing import sky_from_almanac
    alm = constellation_from_almanacs("gps_24", "gal_24")
    sky = sky_from_almanac(alm, ISD.paper_2022(), 45.0, 10.0, 0.0)
    cfg = ARAIMConfig.lpv200()
    base = ARAIM(cfg).protection_levels(sky)
    out = ARAIM(cfg).protection_levels(sky.drop([0]))
    assert out.vpl >= base.vpl - 0.05
