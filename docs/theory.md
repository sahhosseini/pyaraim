# Theory → code map

Equation numbers refer to **[ITM22]** (Blanch et al., ION ITM 2022) unless prefixed **[T]** (TAES 2015).

## Model

| Item | Equation | Code |
|---|---|---|
| Geometry matrix, one clock column per constellation | [T] (4) | `SatelliteSet.G` |
| `C_int`, `C_acc` diagonals | [T] (2), App. A | `errormodels.nominal_error_terms` |
| Tropospheric residual | [T] (58) | `errormodels.sigma_tropo` |
| Prior `P_sat = R_sat·MFD_sat`, `P_const` | (5) | `ConstellationISD.p_sat/p_const` |
| Prior over the exposure window `P(T) = (1+T/MFD)P` | (6), (9) | `faultmodes._event_tables` |
| Fault-mode prior `p_k = ∏_{i∈k}P_i ∏_{i∉k}(1−P_i)` | (10) | `faultmodes.determine_fault_modes` |

## Fault modes

| Item | Equation | Code |
|---|---|---|
| Probability of ≥ r simultaneous events (exact r ≤ 3; bound r ≥ 4) | [T] App. C, (74) | `faultmodes.p_multiple_bound` |
| `N_fault,max` s.t. `P(≥N+1 events) ≤ P_THRES` | (7), [T] (75)–(78) | `faultmodes.determine_nmax` |
| Merge identical subsets, sum priors | (18) | `faultmodes.unique_masks` |
| Fault consolidation | (15)–(16) | `faultmodes._consolidate` |
| Ordered selection of monitored modes, `P_not_monitored` | (11)–(14) | `faultmodes._select_default` |
| Quasi-fixed list | "Quasi-fixed subset list" | `faultmodes._select_quasi_fixed` |

## Solutions and tests

| Item | Equation | Code |
|---|---|---|
| `S^(k) = (GᵀW^(k)G)⁻¹GᵀW^(k)`, clock column removal | [T] (12)–(13) | `solution.wls_coefficients` |
| `σ_q^(k)`, `b_q^(k)`, `σ_ss,q^(k)` | [T] (15)–(17) | `solution.subset_solutions` |
| Rank-one / rank-two updates | [T] App. I, (41) | `solution.downdate_coefficients` |
| Bound of all dual-satellite-fault subsets | (36)–(41) | `solution.dual_fault_bound` |
| Thresholds `T_k,q = K_fa,q σ_ss,q + b_ss,q` | (26)–(28) | `araim.build_terms` |
| `K_fa` (solution separation or χ²) | (28), (36) | `protection.false_alert_containment` |
| χ² sanity check | [T] (22)–(23) | `monitor.ARAIMMonitor` |

## Protection levels

| Item | Equation | Code |
|---|---|---|
| `Q`, `Q̄` | (29)–(30) | `stats.Q`, `stats.Qbar` |
| PL equation with `ρ_j`, `N_ES` | (21)–(25) | `protection.solve_protection_level`, `integrity_targets` |
| Bracketing bounds of the root | [T] App. B | `solve_protection_level` |
| Exposure cap of each mode (Blanch & Walter 2023) | — | `ARAIMConfig.cap_by_exposure` |
| Exclusion options `J`, `ρ_j = 1/(N_exc+1)` | (19), (24) | `araim.exclusion_candidates` |
| Joint allocation across options | (25) | `ARAIM._joint` |
| EMT, accuracy `σ_acc` | [T] (27)–(30) | `ReferenceTerms.emt/sigma_acc` |
| Integrity-optimised estimator | [T] (Improvements) | `araim._adjust_position` |

## Measurement domain

| Step | Code |
|---|---|
| All-in-view solution and tests `τ_k,q = |Δx|/T` | `ARAIMMonitor._tests` |
| Candidate ordering by χ² (Eq. 31 [T]) | `ARAIMMonitor.process` |
| PL after exclusion with `ρ_j` | `ReferenceTerms.solve` |
