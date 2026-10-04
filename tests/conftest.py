import pytest

from pyaraim import ISD
from pyaraim.testing import random_sky


@pytest.fixture(scope="session")
def isd():
    return ISD.paper_2022()


@pytest.fixture()
def sky(isd):
    """16 satellites, deterministic."""
    return random_sky({"GPS": 8, "Galileo": 8}, isd, rng=3)


@pytest.fixture()
def small_sky(isd):
    return random_sky({"GPS": 5, "Galileo": 5}, isd, rng=11)
