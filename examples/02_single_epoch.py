"""Protection levels for the sky of one user at one epoch, built from the bundled almanacs."""
from pyaraim import ARAIM, ARAIMConfig, Criteria, ISD
from pyaraim.simulation import constellation_from_almanacs
from pyaraim.testing import sky_from_almanac

almanacs = constellation_from_almanacs("gps_24", "gal_24")
sky = sky_from_almanac(almanacs, ISD.paper_2022(), lat=34.05, lon=-118.25, t=3600.0, mask_deg=5.0)
counts = {name: int((sky.constellation == j).sum()) for j, name in enumerate(sky.const_names)}
print(f"{sky.n} satellites in view: {counts}")

cfg = ARAIMConfig.lpv200()
araim = ARAIM(cfg)
res = araim.protection_levels(sky, detail=True)
print(f"VPL {res.vpl:.1f} m  HPL {res.hpl:.1f} m  EMT {res.emt:.1f} m  sigma_acc {res.sigma_acc:.2f} m")
print(f"monitored fault modes: {res.n_modes};  P_not_monitored = {res.p_not_monitored:.2e}")
print(f"worst exclusion option: {res.worst_candidate}")
print("LPV-200 available:", res.available(Criteria.lpv200()))

# the fault modes themselves
modes = araim.fault_modes(sky)
for k in range(5):
    print(f"  mode {modes.labels[k]:28s} p_inst={modes.p_inst[k]:.2e}")
