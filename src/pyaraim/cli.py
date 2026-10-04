"""Command line interface: ``pyaraim <command>``.

Commands
--------
``list-almanacs``   show the bundled almanacs
``appendix-j``      reproduce the numerical example of the TAES 2015 paper
``availability``    run an availability/coverage simulation and save ``.npz``/``.csv``/``.png``
``ref-cases``       write (or check) language-neutral reference test vectors
``montecarlo``      verify the protection-level bounds on random skies
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

import numpy as np

from . import __version__

PRESETS = {"lpv200": "lpv200", "rnp": "rnp_horizontal", "baseline2015": "baseline_2015"}


def _cfg(name: str, fast: bool):
    from .config import ARAIMConfig
    cfg = getattr(ARAIMConfig, PRESETS[name])()
    return cfg.replace(fast=fast) if name != "baseline2015" else cfg


def _cmd_list(args) -> int:
    from .simulation.almanac import BUNDLED_ALMANACS, bundled_almanac
    for k, (const, f, desc) in BUNDLED_ALMANACS.items():
        print(f"{k:16s} {const:8s} {len(bundled_almanac(k)):3d} sats  {desc}")
    return 0


def _cmd_appendix_j(args) -> int:
    from . import appendix_j as aj
    from .araim import ARAIM
    r = ARAIM(aj.config()).protection_levels(aj.satellite_set())
    print(f"{'':14s}{'pyaraim':>10s}{'paper':>10s}")
    for name, got in (("VPL (m)", r.vpl), ("HPL (m)", r.hpl), ("EMT (m)", r.emt), ("sigma_acc (m)", r.sigma_acc)):
        key = {"VPL (m)": "vpl", "HPL (m)": "hpl", "EMT (m)": "emt", "sigma_acc (m)": "sigma_acc"}[name]
        print(f"{name:14s}{got:10.2f}{aj.PUBLISHED[key]:10.2f}")
    print(f"fault modes used: {r.n_modes}, K_fa,3 = {r.k_fa[2]:.4f} (paper: {aj.PUBLISHED['k_fa3']})")
    return 0


def _cmd_availability(args) -> int:
    from .config import Criteria
    from .isd import ISD
    from .simulation import Scenario, UserGrid, constellation_from_almanacs, run_availability
    cfg = _cfg(args.preset, not args.exact)
    isd = {"paper2022": ISD.paper_2022, "icao2024": ISD.icao_draft_2024}[args.isd]()
    crit = Criteria(val=args.val, hal=args.hal, emt=args.emt, sigma_acc=args.sigma_acc, name=args.name)
    grid = (UserGrid.world(args.lat_step, args.lon_step) if args.region == "world"
            else UserGrid.region(args.region, args.lat_step, args.lon_step))
    times = np.arange(args.t_start, args.t_end, args.t_step)
    scn = Scenario(constellation_from_almanacs(*args.almanacs), isd, cfg, grid, times, args.mask, criteria=crit)
    print(f"{len(grid)} users x {len(times)} epochs, almanacs {args.almanacs}, preset {args.preset}")
    res = run_availability(scn, n_jobs=args.jobs, verbose=args.verbose)
    res.save(args.out + ".npz")
    res.to_csv(args.out + ".csv", crit)
    print(f"coverage({100 * args.level:g}% availability) = {res.coverage(crit, args.level):.2f} %")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .plotting import plot_availability_map
        plot_availability_map(res, crit)
        plt.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
        print("wrote", args.out + ".png")
    return 0


def _cmd_refcases(args) -> int:
    from .testing.refcases import check_reference_cases, generate_reference_cases
    if args.check:
        res = check_reference_cases(args.path)
        bad = [k for k, v in res.items() if not v]
        print(f"{len(res) - len(bad)}/{len(res)} cases match")
        return 1 if bad else 0
    cases = generate_reference_cases(args.path, args.n, args.seed)
    print(f"wrote {len(cases)} cases to {args.path}")
    return 0


def _cmd_mc(args) -> int:
    from .isd import ISD
    from .testing import random_sky, verify_mode_bound
    cfg = _cfg(args.preset, False).replace(exclusion=False)
    isd = ISD.paper_2022()
    ok = True
    for seed in range(args.skies):
        sky = random_sky({"GPS": 7, "Galileo": 7}, isd, seed)
        for mode in (0, 7, 15):
            for fb in (0.0, 8.0, 20.0):
                c = verify_mode_bound(sky, cfg, mode, 2, fault_bias=fb, n_trials=args.trials, rng=seed)
                ok &= c.holds
                print(f"sky{seed} {c.mode:28s} fault {fb:5.1f}  empirical {c.empirical:.2e}  bound {c.bound:.2e}  "
                      f"{'ok' if c.holds else 'VIOLATION'}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("pyaraim", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list-almanacs").set_defaults(fn=_cmd_list)
    sub.add_parser("appendix-j").set_defaults(fn=_cmd_appendix_j)

    a = sub.add_parser("availability")
    a.add_argument("--almanacs", nargs="+", default=["gps_24", "gal_24"])
    a.add_argument("--preset", choices=list(PRESETS), default="rnp")
    a.add_argument("--isd", choices=["paper2022", "icao2024"], default="paper2022")
    a.add_argument("--region", default="world")
    a.add_argument("--lat-step", type=float, default=10.0)
    a.add_argument("--lon-step", type=float, default=10.0)
    a.add_argument("--t-start", type=float, default=0.0)
    a.add_argument("--t-end", type=float, default=86400.0)
    a.add_argument("--t-step", type=float, default=300.0)
    a.add_argument("--mask", type=float, default=5.0)
    a.add_argument("--val", type=float, default=float("inf"))
    a.add_argument("--hal", type=float, default=556.0)
    a.add_argument("--emt", type=float, default=float("inf"))
    a.add_argument("--sigma-acc", type=float, default=float("inf"))
    a.add_argument("--name", default="custom")
    a.add_argument("--level", type=float, default=0.999)
    a.add_argument("--jobs", type=int, default=1)
    a.add_argument("--exact", action="store_true", help="disable the dual-fault bound screening")
    a.add_argument("--plot", action="store_true")
    a.add_argument("--verbose", action="store_true")
    a.add_argument("--out", default="availability")
    a.set_defaults(fn=_cmd_availability)

    r = sub.add_parser("ref-cases")
    r.add_argument("path")
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--seed", type=int, default=2022)
    r.add_argument("--check", action="store_true")
    r.set_defaults(fn=_cmd_refcases)

    m = sub.add_parser("montecarlo")
    m.add_argument("--preset", choices=list(PRESETS), default="lpv200")
    m.add_argument("--skies", type=int, default=3)
    m.add_argument("--trials", type=int, default=20000)
    m.set_defaults(fn=_cmd_mc)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
