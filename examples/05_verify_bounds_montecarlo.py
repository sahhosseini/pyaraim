"""Check the probability inequality behind the PL equation, mode by mode, by Monte Carlo."""
from pyaraim import ARAIMConfig, ISD
from pyaraim.testing import false_alert_rate, random_sky, verify_mode_bound

sky = random_sky({"GPS": 8, "Galileo": 8}, ISD.paper_2022(), rng=3)
cfg = ARAIMConfig.lpv200(exclusion=False)
print("mode                          fault   empirical        bound    verdict")
for mode in (0, 5, 18):
    for fault in (0.0, 6.0, 12.0):
        c = verify_mode_bound(sky, cfg, mode=mode, fault_bias=fault, n_trials=50000, rng=mode)
        print(f"{c.mode:28s} {fault:5.1f}  {c.empirical:10.3e}  {c.bound:10.3e}    {'ok' if c.holds else 'VIOLATION'}")

# Continuity: at the real budget (1e-7) false alerts are too rare to observe; inflate the budget
cfg2 = cfg.replace(pfa_vert=0.02, pfa_hor=0.02, n_es_cont=1)
emp, upper, analytic = false_alert_rate(sky, cfg2, n_trials=50000, rng=0)
print(f"\nfalse alerts (budget 4 %): empirical {emp:.4f}, 99.9% upper {upper:.4f}, union bound {analytic:.4f}")
