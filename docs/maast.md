# Correspondence with `maast_for_araim_2` (MATLAB)

| MATLAB file | pyaraim |
|---|---|
| `main_maast_araim.m` (parameters, ISD, scenarios) | `ARAIMConfig`, `ISD`, `Scenario` |
| `init_col_labels_araim.m` | `AvailabilityResult` fields (`vpl, hpl, emt, sigma_acc, n_sat`) |
| `svmrun_araim.m`, `usr_vhpl_araim.m` | `run_availability`, `ARAIM.protection_levels` |
| `usrprocess_araim.m`, `af_cnmp_259a.m` | `SatelliteSet.from_los`, `errormodels.ED259A` |
| `mhss_raim_baseline_v5.m` | `araim.build_terms`, `ARAIM._evaluate` |
| `determine_nmax.m`, `determine_k_subsets.m`, `determine_subsets_v5.m` | `faultmodes.determine_nmax`, `determine_fault_modes` |
| `find_unique_subsets.m`, `filter_out_subsets.m` | `faultmodes.unique_masks`, unobservable-mode handling in `build_terms` |
| `compute_subset_solutions.m` | `solution.wls_coefficients`, `subset_solutions` |
| `max_dual_subset_cov.m` | `solution.dual_fault_bound` |
| `compute_test_thresholds.m` | `protection.false_alert_containment` |
| `compute_protection_level_v5.m` | `protection.solve_protection_level` |
| `compute_adjusted_position_1D.m` | `araim._adjust_position` |
| `coverage.m`, `evaluate_coverage.m` | `AvailabilityResult.availability / coverage` |
| `plot_araim_results.m`, `plot_no_gui_araim.m` | `pyaraim.plotting` |

## Known differences

* **Unmonitorable modes in exclusion options** – charged at their full exposure probability against the integrity
  budget (Eq. 2 of [ITM22]); MAAST handles them through the per-mode allocation cap.
* **Consolidation default** – `"individual"` as MAAST; the literal paper rule is `"group"`.
* **Fault priors** – `"exact"` (MAAST); `"product"` is used by the TAES15 preset only.
* **HPL variants** – `compute_hor_protection_level*.m` (2-D, optimised) are not ported; `hpl = sqrt(hpl_E² + hpl_N²)`.
* **Fast mode** – MAAST uses the bound as a screening step (as here) but also bounds `P_THRES` bookkeeping differently
  in some branches.
* **User polygons / world map backdrop** – lat/lon boxes and a plain grid instead of the `usr*.dat` polygons and the
  world map file.
* **Satellite clock/orbit** – Keplerian almanac propagation (no SP3/broadcast ephemeris).

The reference-vector mechanism (`pyaraim ref-cases`) lets you test the MATLAB code against this implementation, or
the reverse, on identical inputs.
