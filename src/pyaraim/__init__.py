"""pyaraim - an open reference implementation of the Advanced RAIM baseline user algorithm.

Based on:

* J. Blanch, T. Walter, P. Enge, Y. Lee, B. Pervan, M. Rippl, A. Spletter, V. Kropp,
  "Baseline Advanced RAIM User Algorithm and Possible Improvements", IEEE TAES 51(1), 2015.
* J. Blanch, T. Walter, C. Milner, M. Joerger, B. Pervan, D. Bouvet,
  "Baseline Advanced RAIM User Algorithm: Proposed Updates", ION ITM 2022.
"""
from .araim import ARAIM, PLResult
from .config import ARAIMConfig, Criteria
from .geometry import SatelliteSet
from .isd import ISD, ConstellationISD
from .monitor import ARAIMMonitor, MonitorResult

__version__ = "0.1.0"
__all__ = ["ARAIM", "PLResult", "ARAIMConfig", "Criteria", "SatelliteSet", "ISD", "ConstellationISD",
           "ARAIMMonitor", "MonitorResult", "__version__"]
