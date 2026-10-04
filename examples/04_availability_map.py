"""Coarse availability map (GPS + Galileo, 24 + 24, HAL = 556 m): a scaled-down version of Fig. 2 of the
ION ITM 2022 paper. Increase the resolution (lat/lon step, t_step) and ``n_jobs`` for a production run."""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyaraim import ARAIMConfig, Criteria, ISD
from pyaraim.plotting import plot_availability_map
from pyaraim.simulation import Scenario, UserGrid, constellation_from_almanacs, run_availability

crit = Criteria.horizontal(556.0)
scn = Scenario(constellation_from_almanacs("gps_24", "gal_24"), ISD.paper_2022(),
               ARAIMConfig.rnp_horizontal(fast=True), UserGrid.regular(10, 30),
               times=np.arange(0, 86400, 1800.0), criteria=crit)
res = run_availability(scn, n_jobs=int(sys.argv[1]) if len(sys.argv) > 1 else 1, verbose=True)
res.save("availability_hal556.npz")
print(f"coverage(99.9 %) = {res.coverage(crit):.2f} %")
plot_availability_map(res, crit)
plt.savefig("availability_hal556.png", dpi=150, bbox_inches="tight")
