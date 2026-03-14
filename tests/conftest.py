"""Shared pytest fixtures for riskstar tests."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from affine import Affine

import geobn

# ---------------------------------------------------------------------------
# Minimal BIF content for a 2-node BN: slope → risk
# ---------------------------------------------------------------------------
_BIF = """network test_bn {
}
variable slope {
    type discrete[2] { low, high };
}
variable risk {
    type discrete[2] { safe, dangerous };
}
probability(slope) {
    table 0.5, 0.5;
}
probability(risk | slope) {
    (low)  0.9, 0.1;
    (high) 0.2, 0.8;
}
"""

# Grid parameters shared by planner integration tests.
# 5×5 pixels, 100 m resolution, UTM 32N origin at (500 000, 7 700 500).
GRID_CRS = "EPSG:32632"
GRID_TRANSFORM = Affine(100.0, 0.0, 500_000.0, 0.0, -100.0, 7_700_500.0)

# Pixel-centre lat/lon values pre-computed for EPSG:32632 grid above.
# Pixel (0, 0) → approx (69.413317°N, 9.001274°E)
# Pixel (4, 4) → approx (69.409730°N, 9.011465°E)
START_LATLON = (69.413317, 9.001274)   # pixel (0, 0)
GOAL_LATLON  = (69.409730, 9.011465)   # pixel (4, 4)


@pytest.fixture
def bif_path(tmp_path):
    """Write the minimal BIF to a temp file and return its path."""
    p = tmp_path / "test.bif"
    p.write_text(_BIF)
    return str(p)


@pytest.fixture
def tiny_bn(bif_path):
    """Fully-configured 5×5 GeoBayesianNetwork ready for inference."""
    bn = geobn.load(bif_path)
    slope_data = np.full((5, 5), 5.0, dtype=np.float32)   # all "low" → safe
    src = geobn.ArraySource(slope_data, crs=GRID_CRS, transform=GRID_TRANSFORM)
    bn.set_input("slope", src)
    bn.set_discretization("slope", [0, 20, 90], labels=["low", "high"])
    return bn
