"""Inject faults into simulated pseudoranges and watch detection, exclusion and the resulting PLs."""
import numpy as np

from pyaraim import ARAIMConfig, ARAIMMonitor, ISD
from pyaraim.testing import (constellation_fault, make_epoch, random_sky, satellite_fault,
                             detection_exclusion_stats)

sky = random_sky({"GPS": 8, "Galileo": 8}, ISD.paper_2022(), rng=3)
mon = ARAIMMonitor(ARAIMConfig.lpv200())
rng = np.random.default_rng(0)

cases = {
    "fault free": [],
    "5 m bias on one satellite": [satellite_fault(4, 5.0)],
    "60 m bias on one satellite": [satellite_fault(4, 60.0)],
    "Galileo constellation fault": [constellation_fault(sky, "Galileo", 0.0, spread=40.0, rng=1)],
}
for name, faults in cases.items():
    ep = make_epoch(sky, faults, rng)
    r = mon.process(sky, ep.y)
    print(f"{name:30s} -> {r.status:8s} excluded={r.excluded!s:14s} VPL={r.vpl:6.1f} "
          f"|true VPE|={abs(ep.position_error[2]):5.1f}")

stats = detection_exclusion_stats(sky, ARAIMConfig.lpv200(), satellite_fault(4, 60.0), n_trials=200, rng=1)
print(stats)
