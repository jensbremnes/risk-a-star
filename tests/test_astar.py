"""Tests for the pure-Python A* implementation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from risk_aware_a_star._astar import astar


def _uniform(rows: int, cols: int, value: float = 0.0) -> np.ndarray:
    return np.full((rows, cols), value, dtype=float)


# ---------------------------------------------------------------------------
# Basic path-finding
# ---------------------------------------------------------------------------

class TestBasicPathFinding:
    def test_uniform_grid_finds_path(self):
        grid = _uniform(5, 5)
        result = astar(grid, (0, 0), (4, 4))
        assert result is not None
        path, cost = result
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        assert cost > 0

    def test_uniform_grid_straight_line_diagonal(self):
        """On a uniform grid the 8-connected path should travel diagonally."""
        grid = _uniform(5, 5)
        result = astar(grid, (0, 0), (4, 4), connectivity=8)
        assert result is not None
        path, _ = result
        # Diagonal path has 5 steps (length 5)
        assert len(path) == 5

    def test_single_pixel_path(self):
        """Start == goal → path of length 1 with zero cost."""
        grid = _uniform(3, 3)
        result = astar(grid, (1, 1), (1, 1))
        assert result is not None
        path, cost = result
        assert path == [(1, 1)]
        assert cost == 0.0

    def test_adjacent_pixels(self):
        grid = _uniform(3, 3)
        result = astar(grid, (0, 0), (0, 1))
        assert result is not None
        path, cost = result
        assert path == [(0, 0), (0, 1)]
        assert math.isclose(cost, 1.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Risk-weight detour
# ---------------------------------------------------------------------------

class TestRiskWeightDetour:
    def test_high_risk_corridor_detoured(self):
        """A high-risk vertical strip with a safe gap forces detour vs. straight path.

        Layout (7 cols, 5 rows):
          - Column 3 rows 0-3 is high risk (risk=1.0).
          - Row 4 col 3 is safe (risk=0.0) — the only gap.
        A risk-averse planner detours through row 4; a cost-blind planner goes straight.
        """
        grid = _uniform(5, 7)
        grid[0:4, 3] = 1.0          # High-risk segment of column 3

        result_risky = astar(grid, (0, 0), (0, 6), risk_weight=20.0)
        result_naive = astar(grid, (0, 0), (0, 6), risk_weight=0.0)

        assert result_risky is not None
        assert result_naive is not None
        path_risky, cost_risky = result_risky
        path_naive, cost_naive = result_naive

        # The risk-averse path must be longer (detours through row 4)
        assert len(path_risky) > len(path_naive), (
            "Risk-averse path should detour and thus be longer"
        )

    def test_zero_risk_weight_ignores_risk(self):
        """With risk_weight=0 all cells are equally costly, shortest path is taken."""
        grid = _uniform(5, 5)
        grid[:, 2] = 1.0   # High risk but ignored
        result = astar(grid, (0, 0), (0, 4), risk_weight=0.0)
        assert result is not None
        path, cost = result
        # Should go straight through (row=0, cols 0..4)
        assert len(path) == 5


# ---------------------------------------------------------------------------
# NaN barrier
# ---------------------------------------------------------------------------

class TestNaNBarrier:
    def test_nan_row_forces_route_around(self):
        grid = _uniform(5, 5)
        # Row 2 is fully impassable except last col
        grid[2, :4] = float("nan")

        result = astar(grid, (0, 0), (4, 0))
        assert result is not None
        path, _ = result
        # Path must not pass through nan cells in row 2, cols 0-3
        for r, c in path:
            assert not (r == 2 and c < 4), f"Path hit NaN cell ({r},{c})"

    def test_fully_blocked_returns_none(self):
        grid = _uniform(3, 3)
        # Ring of NaN surrounds the centre
        grid[0, :] = float("nan")
        grid[2, :] = float("nan")
        grid[:, 0] = float("nan")
        grid[:, 2] = float("nan")
        # Start = (0,0) is NaN itself → None
        result = astar(grid, (0, 0), (2, 2))
        assert result is None

    def test_impassable_start_returns_none(self):
        grid = _uniform(3, 3)
        grid[0, 0] = float("nan")
        assert astar(grid, (0, 0), (2, 2)) is None

    def test_impassable_goal_returns_none(self):
        grid = _uniform(3, 3)
        grid[2, 2] = float("nan")
        assert astar(grid, (0, 0), (2, 2)) is None

    def test_isolated_start_no_path(self):
        """NaN wall completely isolates start from goal."""
        grid = _uniform(5, 5)
        # Wall after first column
        grid[:, 1] = float("nan")
        result = astar(grid, (2, 0), (2, 4))
        assert result is None


# ---------------------------------------------------------------------------
# Connectivity
# ---------------------------------------------------------------------------

class TestConnectivity:
    def test_4_connectivity_cannot_cut_diagonal_corner(self):
        """4-connected A* must take more steps than 8-connected."""
        grid = _uniform(5, 5)
        result_4 = astar(grid, (0, 0), (4, 4), connectivity=4)
        result_8 = astar(grid, (0, 0), (4, 4), connectivity=8)
        assert result_4 is not None and result_8 is not None
        # 4-connected path is longer
        assert len(result_4[0]) > len(result_8[0])

    def test_4_connectivity_no_diagonal_steps(self):
        grid = _uniform(5, 5)
        result = astar(grid, (0, 0), (4, 4), connectivity=4)
        assert result is not None
        path, _ = result
        for (r1, c1), (r2, c2) in zip(path, path[1:]):
            assert abs(r2 - r1) + abs(c2 - c1) == 1, "Diagonal step found in 4-connected path"

    def test_8_connectivity_allows_diagonals(self):
        grid = _uniform(5, 5)
        result = astar(grid, (0, 0), (4, 4), connectivity=8)
        assert result is not None
        path, _ = result
        has_diagonal = any(
            abs(r2 - r1) == 1 and abs(c2 - c1) == 1
            for (r1, c1), (r2, c2) in zip(path, path[1:])
        )
        assert has_diagonal, "8-connected path should use diagonal steps"


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------

class TestCostCalculation:
    def test_diagonal_step_costs_sqrt2(self):
        """Single diagonal step on a zero-risk grid costs exactly √2."""
        grid = _uniform(3, 3)
        result = astar(grid, (0, 0), (1, 1), risk_weight=0.0)
        assert result is not None
        _, cost = result
        assert math.isclose(cost, math.sqrt(2), rel_tol=1e-6)

    def test_cardinal_step_costs_one(self):
        grid = _uniform(3, 3)
        result = astar(grid, (0, 0), (0, 1), risk_weight=0.0)
        assert result is not None
        _, cost = result
        assert math.isclose(cost, 1.0, rel_tol=1e-6)

    def test_risk_increases_cost(self):
        grid_high = _uniform(1, 2, value=1.0)  # both cells fully risky
        grid_low = _uniform(1, 2, value=0.0)
        r_high = astar(grid_high, (0, 0), (0, 1), risk_weight=1.0)
        r_low = astar(grid_low, (0, 0), (0, 1), risk_weight=1.0)
        assert r_high is not None and r_low is not None
        assert r_high[1] > r_low[1]
