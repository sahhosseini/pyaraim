# Availability studies

```python
import numpy as np
from pyaraim import ARAIMConfig, Criteria, ISD
from pyaraim.simulation import Scenario, UserGrid, constellation_from_almanacs, run_availability

crit = Criteria.horizontal(556.0)
scn = Scenario(constellation_from_almanacs("gps_24", "gal_24"), ISD.paper_2022(),
               ARAIMConfig.rnp_horizontal(fast=True), UserGrid.regular(10, 10, box=(-80, 80, -180, 180)),
               times=np.arange(0, 86400, 300.0), criteria=crit)
res = run_availability(scn, n_jobs=4)
res.coverage(crit, 0.999)          # % of users (cos-lat weighted) with >= 99.9 % availability
res.save("run.npz"); res.to_csv("run.csv", crit)
```

* **Outage scenarios**: `Almanac.without([prns])`, or the bundled `*_minus1` almanacs.
* **Regions**: `UserGrid.region("conus" | "europe" | ... )` (lat/lon boxes), `UserGrid.points(lat, lon)`.
* **Criteria**: `Criteria.lpv200()`, `Criteria.rnp(nm)`, `Criteria.horizontal(hal)`, or custom.
* **Plots**: `pyaraim.plotting.plot_availability_map / plot_pl_map / plot_time_series`.
* **Speed**: `fast=True` + `criteria` enables the dual-fault bound screening; `n_jobs` distributes epochs.
