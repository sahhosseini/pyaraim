# pyaraim

Open-source reference implementation of the **Advanced RAIM baseline user algorithm**
(Blanch et al., IEEE TAES 2015; ION ITM 2022) plus the tools to test it.

* [Theory → code map](theory.md) – every equation of the papers and the function that implements it
* [Testing toolkit](testing.md) – scenarios, fault injection, Monte-Carlo verification, reference vectors
* [Availability studies](availability.md) – almanacs, grids, coverage, plots
* [Validation](validation.md) – what is reproduced from the papers and with what accuracy
* [MATLAB (MAAST) correspondence](maast.md) – file-by-file mapping and known differences

## Minimal example

```python
from pyaraim import ARAIM, ARAIMConfig, appendix_j
res = ARAIM(appendix_j.config()).protection_levels(appendix_j.satellite_set())
print(res.vpl, res.hpl, res.emt, res.sigma_acc)    # 19.06 14.2 8.30 1.470
```

## Data flow

```
ISD + error model + geometry ──► SatelliteSet ──► determine_fault_modes ──► FaultModeList
                                      │                                         │
                                      └──────────────► build_terms (per exclusion option j) ◄─┘
                                                           │  subset WLS, σ, σ_ss, b, T_k, p_not_monitored
                                                           ▼
                                      solve_protection_level (Eq. 21–25) ──► VPL, HPL, EMT, σ_acc
                                                           ▼
                           ARAIM.protection_levels (worst case over j)      ARAIMMonitor.process (measurements)
```
