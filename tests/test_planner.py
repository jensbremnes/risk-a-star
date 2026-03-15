"""Integration tests for RiskAwareAStarPlanner."""

from __future__ import annotations

import numpy as np
import pytest

import geobn
from risk_aware_a_star import PathResult, RiskAwareAStarPlanner
from conftest import GOAL_LATLON, GRID_CRS, GRID_TRANSFORM, START_LATLON


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_planner(tiny_bn, risk_weight=1.0):
    return RiskAwareAStarPlanner(
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
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert isinstance(result, PathResult)


# ---------------------------------------------------------------------------
# Full end-to-end
# ---------------------------------------------------------------------------

class TestFindPath:
    def test_returns_path_result(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert isinstance(result, PathResult)

    def test_waypoints_non_empty(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert len(result.waypoints) >= 2

    def test_first_waypoint_near_start(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        lat, lon = result.waypoints[0]
        assert abs(lat - START_LATLON[0]) < 0.01
        assert abs(lon - START_LATLON[1]) < 0.01

    def test_last_waypoint_near_goal(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        lat, lon = result.waypoints[-1]
        assert abs(lat - GOAL_LATLON[0]) < 0.01
        assert abs(lon - GOAL_LATLON[1]) < 0.01

    def test_total_cost_positive(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.total_cost > 0.0

    def test_total_distance_px_positive(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.total_distance_px > 0.0

    def test_risk_grid_shape(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.risk_grid.shape == (5, 5)

    def test_inference_result_attached(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON)
        assert result.inference_result is not None


# ---------------------------------------------------------------------------
# return_coords modes
# ---------------------------------------------------------------------------

class TestReturnCoords:
    @pytest.mark.parametrize("mode", ["latlon", "crs", "pixel"])
    def test_valid_return_coords(self, tiny_bn, mode):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON, return_coords=mode)
        assert len(result.waypoints) > 0

    def test_pixel_mode_returns_int_tuples(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        result = planner.find_path(START_LATLON, GOAL_LATLON, return_coords="pixel")
        for wp in result.waypoints:
            assert len(wp) == 2


# ---------------------------------------------------------------------------
# Out-of-bounds
# ---------------------------------------------------------------------------

class TestOutOfBounds:
    def test_start_outside_grid_raises(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        far_away = (0.0, 0.0)   # Equator / prime meridian — outside UTM 32N grid
        with pytest.raises(ValueError, match="start"):
            planner.find_path(far_away, GOAL_LATLON)

    def test_goal_outside_grid_raises(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        far_away = (0.0, 0.0)
        with pytest.raises(ValueError, match="goal"):
            planner.find_path(START_LATLON, far_away)


# ---------------------------------------------------------------------------
# Precomputed persistence
# ---------------------------------------------------------------------------

def _make_fresh_bn(bif_path):
    """Create a fully-configured but not-yet-precomputed BN."""
    bn = geobn.load(bif_path)
    slope_data = np.full((5, 5), 5.0, dtype=np.float32)
    bn.set_input("slope", geobn.ArraySource(slope_data, crs=GRID_CRS, transform=GRID_TRANSFORM))
    bn.set_discretization("slope", [0, 20, 90], labels=["low", "high"])
    return bn


# ---------------------------------------------------------------------------
# Risk-grid caching
# ---------------------------------------------------------------------------

class TestRiskGridCaching:
    def test_risk_stale_true_on_construction(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        assert planner._risk_stale is True

    def test_cache_none_before_first_call(self, tiny_bn):
        planner = _make_planner(tiny_bn)
        assert planner._risk_grid is None
        assert planner._infer_result is None

    def test_first_call_populates_cache_and_clears_stale_flag(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        planner.find_path(START_LATLON, GOAL_LATLON)
        assert planner._risk_stale is False
        assert planner._risk_grid is not None
        assert planner._infer_result is not None

    def test_second_call_reuses_same_risk_grid_object(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        planner.find_path(START_LATLON, GOAL_LATLON)
        grid_id = id(planner._risk_grid)
        planner.find_path(START_LATLON, GOAL_LATLON)
        assert id(planner._risk_grid) == grid_id

    def test_update_input_sets_stale_flag(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        planner.find_path(START_LATLON, GOAL_LATLON)
        assert planner._risk_stale is False
        new_src = geobn.ArraySource(
            np.full((5, 5), 10.0, dtype=np.float32), crs=GRID_CRS, transform=GRID_TRANSFORM
        )
        planner.update_input("slope", new_src)
        assert planner._risk_stale is True

    def test_update_input_triggers_reinference(self, tiny_bn):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        planner.find_path(START_LATLON, GOAL_LATLON)
        old_id = id(planner._risk_grid)
        new_src = geobn.ArraySource(
            np.full((5, 5), 10.0, dtype=np.float32), crs=GRID_CRS, transform=GRID_TRANSFORM
        )
        planner.update_input("slope", new_src)
        planner.find_path(START_LATLON, GOAL_LATLON)
        assert id(planner._risk_grid) != old_id

    def test_freeze_static_nodes_delegates_to_bn(self, tiny_bn, monkeypatch):
        tiny_bn.precompute(["risk"])
        planner = _make_planner(tiny_bn)
        calls = []
        monkeypatch.setattr(tiny_bn, "freeze", lambda *nodes: calls.append(nodes))
        planner.freeze_static_nodes("slope")
        assert calls == [("slope",)]


class TestPrecomputedPersistence:
    def test_save_creates_file(self, tiny_bn, tmp_path):
        tiny_bn.precompute(["risk"])
        out = tmp_path / "table.npz"
        tiny_bn.save_precomputed(out)
        assert out.exists()

    def test_load_marks_ready(self, tiny_bn, bif_path, tmp_path):
        tiny_bn.precompute(["risk"])
        path = tmp_path / "table.npz"
        tiny_bn.save_precomputed(path)
        fresh = _make_planner(_make_fresh_bn(bif_path))
        assert not fresh._precomputed
        fresh.load_precomputed(path)
        assert fresh._precomputed

    def test_load_then_find_path(self, tiny_bn, bif_path, tmp_path):
        tiny_bn.precompute(["risk"])
        path = tmp_path / "table.npz"
        tiny_bn.save_precomputed(path)
        fresh = _make_planner(_make_fresh_bn(bif_path))
        fresh.load_precomputed(path)
        result = fresh.find_path(START_LATLON, GOAL_LATLON)
        assert len(result.waypoints) >= 2
