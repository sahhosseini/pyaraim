# Testing toolkit

`pyaraim.testing` is meant for people who want to test *an* ARAIM implementation (this one, MAAST, or their own).

## Scenarios

```python
from pyaraim.testing import random_sky, sky_from_almanac, outage_cases, make_epoch
sky = random_sky({"GPS": 8, "Galileo": 8}, isd, rng=3)           # stress / property tests
sky = sky_from_almanac(almanacs, isd, lat=45, lon=10, t=0.0)      # realistic geometry
cases = outage_cases(sky, max_out=2)                              # single and double outages
ep = make_epoch(sky, faults=[...], rng=0, noise="acc")            # simulated residuals + truth
```

## Fault injection

* `satellite_fault(i, bias)`, `constellation_fault(sky, "Galileo", bias, spread)`, `ramp_fault(i, rate, t)`.
* `critical_fault(sky, faulted_set, error, axis)` — the fault that yields a position error `error` with the smallest
  possible solution separation (direction `S⁰_q` restricted to the faulted set). It is the stress case of the
  solution-separation monitor.
* Note: a bias *common* to all satellites of a constellation is absorbed by its clock state and is neither
  harmful nor detectable; constellation faults need a spread across satellites.

## Verification by Monte Carlo

| Function | What is checked |
|---|---|
| `verify_mode_bound` | `P(err > L ∧ test passes | fault k) ≤ Q̄((L−T_k−b_k)/σ_k)` for a chosen mode, level and fault size |
| `false_alert_rate` | empirical false-alert rate vs. the union bound `P_FA/N_ES,CONT` |
| `detection_exclusion_stats` | detection, correct/wrong exclusion, alerts and HMI counts of the monitor |

At the true budgets (1e-7) events are unobservable by simulation; use inflated budgets
(`cfg.replace(pfa_vert=0.02, ...)`) to make the inequalities measurable.

## Reference vectors for other implementations

```bash
pyaraim ref-cases cases.json --n 50
```

produces JSON with `G`, covariance diagonals, biases, priors, the configuration and the expected
`VPL/HPL/EMT/σ_acc`, number of modes and `K_fa`. Load it from MATLAB/C++/Rust, rebuild the problem, and compare.
`pyaraim ref-cases cases.json --check` replays a file against this package.
