"""Tests for the risk dilation (spatial maximum filter)."""
from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from risk_aware_a_star._filter import dilate_risk
from risk_aware_a_star import RiskAwareAStarPlanner

from conftest import GRID_CRS, GRID_TRANSFORM, START_LATLON, GOAL_LATLON

# Simple projected transform: 100 m pixels, UTM-like CRS
_TRANSFORM = GRID_TRANSFORM
_CRS = GRID_CRS


def _uniform_grid(value: float, shape=(7, 7)) -> np.ndarray:
    return np.full(shape, value, dtype=float)


# ---------------------------------------------------------------------------
# dilate_risk: identity when radius == 0
# ---------------------------------------------------------------------------

def test_zero_radius_is_identity():
    grid = np.array([[0.1, 0.5], [0.3, 0.9]])
    result = dilate_risk(grid, 0.0, _TRANSFORM, _CRS)
    assert result is grid  # same object returned unchanged


# ---------------------------------------------------------------------------
# dilate_risk: single high-risk pixel raises neighbours
# ---------------------------------------------------------------------------

def test_single_hot_pixel_spreads():
    # 9×9 grid, all 0.1; centre pixel = 0.9
    grid = _uniform_grid(0.1, (9, 9))
    grid[4, 4] = 0.9

    # radius = 150 m → 1.5 pixels → neighbours within 1.5 px should be raised
    result = dilate_risk(grid, 150.0, _TRANSFORM, _CRS)

    # Centre itself should still be 0.9
    assert result[4, 4] == pytest.approx(0.9)
    # Immediate neighbours (within radius) should be raised to 0.9
    assert result[4, 5] == pytest.approx(0.9)
    assert result[5, 4] == pytest.approx(0.9)
    # Corner far from centre should be unchanged
    assert result[0, 0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# dilate_risk: NaN cells never source elevated risk; remain NaN
# ---------------------------------------------------------------------------

def test_nan_cells_not_source_and_preserved():
    grid = np.full((5, 5), 0.2, dtype=float)
    # Make entire left column NaN (land)
    grid[:, 0] = np.nan
    # High risk adjacent to land (column 1)
    grid[2, 1] = 0.8

    result = dilate_risk(grid, 150.0, _TRANSFORM, _CRS)

    # NaN column must remain NaN — land is not overwritten by water risk
    assert np.all(np.isnan(result[:, 0]))
    # The high-risk cell at (2,1) should have spread to (2,2)
    assert result[2, 2] >= 0.8 - 1e-9


def test_nan_cells_do_not_spread_risk():
    # Grid: all zeros except NaN in the middle; after dilation zeros stay zero
    grid = np.zeros((5, 5), dtype=float)
    grid[2, 2] = np.nan

    result = dilate_risk(grid, 150.0, _TRANSFORM, _CRS)

    # Non-NaN cells should not be elevated by the NaN source
    non_nan = ~np.isnan(result)
    assert np.all(result[non_nan] == pytest.approx(0.0))
    # NaN stays NaN
    assert np.isnan(result[2, 2])


# ---------------------------------------------------------------------------
# Integration: risk_dilation_m constructor parameter
# ---------------------------------------------------------------------------

def test_planner_dilation_raises_neighbours(tiny_bn):
    """With dilation, cells near a high-risk pixel should see elevated risk."""
    import tempfile, os
    from pathlib import Path

    # Precompute once and save to a temp file
    tiny_bn.precompute(["risk"])
    with tempfile.TemporaryDirectory() as tmp:
        table_path = Path(tmp) / "table.npz"
        tiny_bn.save_precomputed(table_path)

        # Planner without dilation
        planner_plain = RiskAwareAStarPlanner(
            tiny_bn, "risk", "dangerous", risk_weight=1.0,
        )
        planner_plain.load_precomputed(table_path)
        result_plain = planner_plain.find_path(START_LATLON, GOAL_LATLON)

        # Planner with 150 m dilation (> 1 pixel at 100 m/px)
        planner_dilated = RiskAwareAStarPlanner(
            tiny_bn, "risk", "dangerous", risk_weight=1.0,
            risk_dilation_m=150.0,
        )
        planner_dilated.load_precomputed(table_path)
        result_dilated = planner_dilated.find_path(START_LATLON, GOAL_LATLON)

    # Dilated grid must be >= plain grid everywhere (it's a maximum filter)
    plain_nan = np.isnan(result_plain.risk_grid)
    assert np.all(
        result_dilated.risk_grid[~plain_nan] >= result_plain.risk_grid[~plain_nan] - 1e-9
    )
