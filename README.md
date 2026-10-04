# pyaraim

**An open-source reference implementation of the Advanced RAIM (ARAIM) baseline user algorithm**, with the
tools needed to *test* it: almanac-based constellation simulation, availability/coverage studies, fault
injection, Monte-Carlo verification of the integrity bounds, and language-neutral reference vectors.

It implements the algorithm of

* J. Blanch, T. Walter, C. Milner, M. Joerger, B. Pervan, D. Bouvet, *Baseline Advanced RAIM User Algorithm:
  Proposed Updates*, ION ITM 2022 — referred to as **[ITM22]** (primary reference), and
* J. Blanch, T. Walter, P. Enge, Y. Lee, B. Pervan, M. Rippl, A. Spletter, V. Kropp, *Baseline Advanced RAIM
  User Algorithm and Possible Improvements*, IEEE TAES 51(1), 2015 — **[TAES15]** (baseline, appendices),

and was cross-checked against the MATLAB `maast_for_araim_2` repository (the scripts that produced the
published results).

> **Status.** Research/teaching code, not certified avionics software. See [Validation](#validation) for what
> is and is not reproduced, and [Limitations](#limitations).

## Install

```bash
pip install -e .            # numpy + scipy
pip install -e .[plot,dev]  # + matplotlib, pytest, mkdocs
```

## 60-second tour

```python
from pyaraim import ARAIM, ARAIMConfig, Criteria, ISD
from pyaraim.simulation import constellation_from_almanacs
from pyaraim.testing import sky_from_almanac

sky = sky_from_almanac(constellation_from_almanacs("gps_24", "gal_24"), ISD.paper_2022(),
                       lat=34.05, lon=-118.25, t=3600.0)           # one user, one epoch
res = ARAIM(ARAIMConfig.lpv200()).protection_levels(sky)
print(res.vpl, res.hpl, res.emt, res.sigma_acc, res.available(Criteria.lpv200()))
```

Measurement domain (tests, exclusion, protection levels after exclusion):

```python
from pyaraim import ARAIMMonitor
from pyaraim.testing import make_epoch, satellite_fault

ep = make_epoch(sky, [satellite_fault(4, 60.0)], rng=0)             # 60 m fault on satellite 4
out = ARAIMMonitor(ARAIMConfig.lpv200()).process(sky, ep.y)
print(out.status, out.excluded, out.vpl, out.hpl)                   # excluded GPS-xx
```

Command line:

```bash
pyaraim appendix-j                                    # reproduce the TAES15 numerical example
pyaraim availability --hal 556 --lat-step 10 --lon-step 10 --jobs 4 --plot --out run1
pyaraim ref-cases cases.json --n 20                   # reference vectors for other implementations
pyaraim ref-cases cases.json --check
pyaraim montecarlo --skies 3                          # verify the PL probability bound
```

## What is in the box

| Module | Content | Paper |
|---|---|---|
| `pyaraim.config` | `ARAIMConfig` (all parameters, 3 presets), `Criteria` (LPV-200, RNP ...) | Table II [TAES15]; Fig. 1 [ITM22] |
| `pyaraim.isd` | Integrity Support Data, priors `P = R·MFD` | Table 1, Eq. (5) [ITM22] |
| `pyaraim.errormodels` | `C_int`, `C_acc`, tropo, ED-259A and Appendix-A user error | Eq. (2), App. A [TAES15] |
| `pyaraim.geometry` | `SatelliteSet`: `G`, covariances, biases, priors | Eq. (4)–(5) |
| `pyaraim.faultmodes` | `N_fault,max`, mode probabilities, consolidation, ordered selection | Eqs. (5)–(18), App. C |
| `pyaraim.solution` | subset WLS, σ, σ_ss, biases, rank-one updates, dual-fault bound | Eqs. (15)–(17), App. I, Eqs. (36)–(41) |
| `pyaraim.protection` | `Qbar`, PL equation solver, `K_fa`, integrity allocation | Eqs. (21)–(30), App. B |
| `pyaraim.araim` | `ARAIM.protection_levels`: VPL/HPL/EMT/σ_acc, worst case over exclusion options | Eqs. (19)–(28) |
| `pyaraim.monitor` | `ARAIMMonitor.process`: tests, chi² check, exclusion, PL after exclusion | Eqs. (18)–(24) [TAES15] |
| `pyaraim.simulation` | almanacs, Kepler propagation, user grids, availability/coverage engine | MAAST `svmrun_araim` |
| `pyaraim.testing` | random skies, fault injection, critical faults, Monte-Carlo, reference vectors | — |
| `pyaraim.plotting`, `pyaraim.cli` | maps with the paper's colour scale; `pyaraim` command | Fig. 2 [ITM22] |

Bundled data: eight almanacs from the MAAST-for-ARAIM repository (GPS 24, 24−1, 24+3; Galileo 24, 24−1, 24+3,
+ an alternative Galileo-24 file).

## Validation

Everything below is executed by `pytest` (118 tests) unless stated otherwise.

**Reproduction of published numbers**

| Quantity | pyaraim | Published | Source |
|---|---|---|---|
| VPL, Appendix J example | 19.06 m | 19.10 m | TAES15 App. J |
| HPL, Appendix J example | 14.2 m | 14.1 m | TAES15 App. J |
| EMT, Appendix J example | 8.30 m | 8.3 m | TAES15 App. J |
| σ_v,acc | 1.470 m | 1.47 m | TAES15 App. J |
| σ_v of the constellation modes | 2.5760 / 2.5577 m | same | TAES15 Eq. (132) |
| σ_ss,v of the constellation modes | 1.5307 / 1.5292 m | same | TAES15 Eq. (132) |
| K_fa,3 with 57 modes | 5.3953 | 5.3953 | TAES15 Eq. (133) |
| `N_fault,max` table | 11/11 entries | Table C.I | TAES15 App. C |

**Internal consistency (property tests)**: subset solutions against the textbook WLS formula; rank-one/two
update formulas of Appendix I against direct solves (to 1e-14); the dual-fault bound is never exceeded;
Eq. (37) identity `σ_ss² = σ_k² − σ_0²`; the PL returned satisfies the PL equation within tolerance and is
monotone in σ, targets and priors; permutation and quarter-turn invariance; the fast (bound) mode is never
optimistic and preserves availability decisions.

**Statistical verification (Monte Carlo)**: the per-mode inequality that underlies the PL equation,
`P(error > L, test passes | fault k) ≤ Qbar((L − T_k − b_k)/σ_k)`, is checked empirically for many modes,
levels and fault magnitudes, including the tightness of the bound; the false-alert union bound and the
χ² threshold are checked at inflated budgets (at the real 1e-7 budgets events are too rare to observe).

**Availability/coverage** (GPS 24 + Galileo 24, ISD and settings of Fig. 1 of [ITM22], 300 s, 99.9 %):

| | pyaraim | Fig. 2 of [ITM22] |
|---|---:|---:|
| Coverage, HAL = 556 m | 96.8 % | 97.04 % |
| Coverage, HAL = 185 m | 83.4 % | 78.6 % |

The tight-HAL result is optimistic by about 5 points (sparse 3-longitude grid, assumed Galileo URE and user-error
model, no polygon mask); nothing was tuned to match. Details and caveats in `docs/validation.md`.

## Design decisions and deviations (please read)

These are the places where the papers leave freedom or where the MATLAB code differs from the text:

1. **Fault priors** – `prior_form="exact"` uses `∏P_i ∏(1−P_i)` (Eq. 10 [ITM22], as in MAAST); the TAES15 preset
   uses the plain product of Eq. (9) [TAES15], which is what reproduces its Appendix J EMT.
2. **Consolidation** – `consolidation="individual"` (default, as MAAST) absorbs each weak same-constellation
   mode into the constellation mode; `"group"` is the literal Eq. (15)–(16) of [ITM22]. Both are conservative.
3. **Modes that cannot be monitored** (e.g. after excluding one constellation the remaining one is faulted as
   a whole) are charged their full exposure probability `p_fault,k(T_EXP)` against the integrity budget
   (Eq. 2 [ITM22]), in every exclusion option.
4. **`fast=True`** reproduces the *screening* use of the dual-fault bound (Eqs. 36–41): the bound is conservative,
   so if it already meets the alert limits the user is available; otherwise exact levels are computed.
5. **HPL** is `√(HPL_E² + HPL_N²)` with the per-axis allocation of Eqs. (21)–(22); the tighter 2-D HPL variants
   of the MATLAB code are not implemented.
6. **Exclusion** follows [ITM22]: any consistent candidate in `J` may be chosen (smallest χ² first); the wrong
   exclusion test of [TAES15] (Eq. 34) is not needed with the pre-allocation `ρ_j`.
7. **Almanac time** – satellite time is measured from the almanac time of applicability; user polygons of MAAST
   (`usrworld.dat`, …) are not redistributed: `UserGrid.region` uses lat/lon boxes.

## Limitations

* Pseudorange *processing* (ionosphere-free combination, smoothing, ephemeris, receiver clock steering) is out
  of scope; the monitor takes residuals. The nominal error model supplied is that of dual-frequency L1/L5.
* Only the baseline estimator and the position optimisation of TAES15 are provided; no SBAS/GBAS, no time-varying
  integrity allocation, no real ISD message parsing.
* Python speed: ≈ 30 ms per user-epoch with exclusion; a 10°×10°, 300 s, 24 h world run is about 1–2 CPU-hours
  (`--jobs N` helps linearly).

## Citing

If you use this software, please cite both papers above and this repository (see `CITATION.cff`).

## License

BSD-3-Clause. The bundled almanac files originate from the MAAST-for-ARAIM repository.
