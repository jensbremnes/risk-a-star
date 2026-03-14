"""Integration tests for RiskStarPlanner."""

from __future__ import annotations

import numpy as np
import pytest

from riskstar import PathResult, RiskStarPlanner
from conftest import GOAL_LATLON, START_LATLON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_planner(tiny_bn, risk_weight=1.0):
    return RiskStarPlanner(
        bn=tiny_bn,
        risk_node="risk",
        risk_state="dangerous",
        risk_weight=risk_weight,
    )


# ---------------------------------------------------------------------------
# Guard: precompute required
# ---------------------------------------------------------------------------

class TestPrecomputeGuard:
    def test_find_path_without_precompute_raises(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        with pytest.raises(RuntimeError, match="precompute"):
            planner.find_path(START_LATLON, GOAL_LATLON)

    def test_precompute_then_find_path_ok(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert isinstance(result, PathResult)


# ---------------------------------------------------------------------------
# Full end-to-end
# ---------------------------------------------------------------------------

class TestFindPath:
    def test_returns_path_result(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert isinstance(result, PathResult)

    def test_waypoints_non_empty(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert len(result.waypoints) >= 2

    def test_first_waypoint_near_start(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        lat, lon = result.waypoints[0]
        assert abs(lat - START_LATLON[0]) < 0.01
        assert abs(lon - START_LATLON[1]) < 0.01

    def test_last_waypoint_near_goal(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        lat, lon = result.waypoints[-1]
        assert abs(lat - GOAL_LATLON[0]) < 0.01
        assert abs(lon - GOAL_LATLON[1]) < 0.01

    def test_total_cost_positive(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.total_cost > 0.0

    def test_total_distance_px_positive(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.total_distance_px > 0.0

    def test_risk_grid_shape(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.risk_grid.shape == (5, 5)

    def test_inference_result_attached(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.inference_result is not None


# ---------------------------------------------------------------------------
# return_coords modes
# ---------------------------------------------------------------------------

class TestReturnCoords:
    @pytest.mark.parametrize("mode", ["latlon", "crs", "pixel"])
    def test_valid_return_coords(self, tiny_bn, mode):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON, return_coords=mode)
        assert len(result.waypoints) > 0

    def test_pixel_mode_returns_int_tuples(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        result = planner.find_path(START_LATLON, GOAL_LATLON, return_coords="pixel")
        for wp in result.waypoints:
            assert len(wp) == 2


# ---------------------------------------------------------------------------
# Out-of-bounds
# ---------------------------------------------------------------------------

class TestOutOfBounds:
    def test_start_outside_grid_raises(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        far_away = (0.0, 0.0)   # Equator / prime meridian — outside UTM 32N grid
        with pytest.raises(ValueError, match="start"):
            planner.find_path(far_away, GOAL_LATLON)

    def test_goal_outside_grid_raises(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        planner.precompute()
        far_away = (0.0, 0.0)
        with pytest.raises(ValueError, match="goal"):
            planner.find_path(START_LATLON, far_away)
