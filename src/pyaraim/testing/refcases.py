"""Language-neutral reference test vectors.

:func:`generate_reference_cases` produces a JSON file with *inputs* (geometry matrix, nominal
variances, biases, fault priors, algorithm parameters) and the *expected outputs* of this
implementation (fault modes, protection levels, EMT, accuracy). It is meant to validate other
implementations (MATLAB/MAAST, C/C++, Rust ...) without sharing any code, and to detect regressions
of this one. :func:`check_reference_cases` replays a file against the current code.

The geometry convention is that of :mod:`pyaraim.geometry` (``G`` rows = ``[-los_enu, clock columns]``).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from ..araim import ARAIM
from ..config import ARAIMConfig
from ..geometry import SatelliteSet
from ..isd import ISD
from .scenarios import random_sky

__all__ = ["generate_reference_cases", "check_reference_cases", "case_from_dict", "case_to_dict"]

SCHEMA = "pyaraim.refcases/1"


def case_to_dict(name: str, sats: SatelliteSet, cfg: ARAIMConfig) -> dict:
    araim = ARAIM(cfg)
    res = araim.protection_levels(sats, detail=True)
    modes = araim.fault_modes(sats)
    return {
        "name": name,
        "config": asdict(cfg),
        "inputs": {
            "G": sats.G.tolist(),
            "constellation_names": list(sats.const_names),
            "constellation_index": sats.constellation.tolist(),
            "var_int": sats.var_int.tolist(), "var_acc": sats.var_acc.tolist(),
            "b_int": sats.b_int.tolist(), "b_acc": sats.b_acc.tolist(),
            "p_sat": sats.p_sat.tolist(), "mfd_sat": sats.mfd_sat.tolist(),
            "p_const": sats.p_const.tolist(), "mfd_const": sats.mfd_const.tolist(),
        },
        "expected": {
            "n_modes_listed": len(modes),
            "p_not_monitored": res.p_not_monitored,
            "k_fa": [float(x) for x in res.k_fa],
            "vpl": res.vpl, "hpl": res.hpl, "emt": res.emt, "sigma_acc": res.sigma_acc,
        },
    }


def case_from_dict(d: dict):
    i = d["inputs"]
    G = np.array(i["G"])
    sats = SatelliteSet(-G[:, :3], np.array(i["constellation_index"]), tuple(i["constellation_names"]),
                        i["var_int"], i["var_acc"], i["b_int"], i["b_acc"], i["p_sat"], i["mfd_sat"],
                        i["p_const"], i["mfd_const"])
    return sats, ARAIMConfig(**d["config"])


def generate_reference_cases(path: Union[str, Path], n_random: int = 10, seed: int = 2022,
                             isd: Optional[ISD] = None) -> List[dict]:
    """Write ``n_random`` random skies for each of three configurations, plus Appendix J of the TAES paper."""
    isd = isd or ISD.paper_2022()
    rng = np.random.default_rng(seed)
    cases = []
    cfgs = {"lpv200_fde": ARAIMConfig.lpv200(), "rnp_fde": ARAIMConfig.rnp_horizontal(),
            "baseline2015_fd": ARAIMConfig.baseline_2015(subset_selection="default")}
    for k in range(n_random):
        sky = random_sky({"GPS": int(rng.integers(5, 10)), "Galileo": int(rng.integers(5, 10))}, isd, rng)
        for cname, cfg in cfgs.items():
            cases.append(case_to_dict(f"random{k:02d}_{cname}", sky, cfg))
    from .. import appendix_j
    cases.append(case_to_dict("taes2015_appendix_J", appendix_j.satellite_set(), appendix_j.config()))
    Path(path).write_text(json.dumps({"schema": SCHEMA, "cases": cases}, indent=1))
    return cases


def check_reference_cases(path: Union[str, Path], rtol: float = 1e-6) -> Dict[str, bool]:
    """Replay a reference file; returns ``{case: matches}``."""
    data = json.loads(Path(path).read_text())
    out = {}
    for c in data["cases"]:
        sats, cfg = case_from_dict(c)
        res = ARAIM(cfg).protection_levels(sats)
        exp = c["expected"]
        got = np.array([res.vpl, res.hpl, res.emt, res.sigma_acc])
        ref = np.array([exp["vpl"], exp["hpl"], exp["emt"], exp["sigma_acc"]])
        out[c["name"]] = bool(np.allclose(got, ref, rtol=rtol, atol=0, equal_nan=True))
    return out
