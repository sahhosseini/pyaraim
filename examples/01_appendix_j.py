"""Reproduce the numerical example of Appendix J (Blanch et al., IEEE TAES 2015)."""
from pyaraim import ARAIM, appendix_j as aj

res = ARAIM(aj.config()).protection_levels(aj.satellite_set())
print(f"VPL        {res.vpl:6.2f} m   (paper {aj.PUBLISHED['vpl']})")
print(f"HPL        {res.hpl:6.2f} m   (paper {aj.PUBLISHED['hpl']})")
print(f"EMT        {res.emt:6.2f} m   (paper {aj.PUBLISHED['emt']})")
print(f"sigma_acc  {res.sigma_acc:6.3f} m   (paper {aj.PUBLISHED['sigma_acc']})")
print(f"{res.n_modes} fault modes monitored, P_not_monitored = {res.p_not_monitored:.2e}")
