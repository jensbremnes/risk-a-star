"""Tests for risk grid extraction."""

from __future__ import annotations

import numpy as np
import pytest

from riskstar._risk import extract_risk_grid


# ---------------------------------------------------------------------------
# Minimal stub for geobn.InferenceResult
# ---------------------------------------------------------------------------

class _FakeResult:
    """Minimal stand-in for geobn.InferenceResult."""

    def __init__(self, probs: np.ndarray, state_names: list[str], node: str = "risk"):
        self.node = node
        self.probabilities = {node: probs}
        self.state_names = {node: state_names}


# ---------------------------------------------------------------------------
# String-state extraction
# ---------------------------------------------------------------------------

class TestStringState:
    def test_extracts_correct_state_index(self):
        probs = np.array([[[0.7, 0.3], [0.4, 0.6]]], dtype=float)  # (1,2,2)
        result = _FakeResult(probs, ["safe", "dangerous"])
        grid = extract_risk_grid(result, "risk", "dangerous")
        expected = np.array([[0.3, 0.6]])
        np.testing.assert_allclose(grid, expected)

    def test_first_state(self):
        probs = np.array([[[0.8, 0.2]]], dtype=float)  # (1,1,2)
        result = _FakeResult(probs, ["safe", "risky"])
        grid = extract_risk_grid(result, "risk", "safe")
        np.testing.assert_allclose(grid, np.array([[0.8]]))

    def test_invalid_state_raises(self):
        probs = np.ones((2, 2, 2), dtype=float) * 0.5
        result = _FakeResult(probs, ["safe", "dangerous"])
        with pytest.raises(ValueError):
            extract_risk_grid(result, "risk", "nonexistent")


# ---------------------------------------------------------------------------
# Dict-state weighted combination
# ---------------------------------------------------------------------------

class TestDictState:
    def test_weighted_sum(self):
        probs = np.array([[[0.5, 0.3, 0.2]]], dtype=float)  # (1,1,3)
        result = _FakeResult(probs, ["low", "medium", "high"])
        grid = extract_risk_grid(result, "risk", {"medium": 0.5, "high": 1.0})
        expected = 0.5 * 0.3 + 1.0 * 0.2
        np.testing.assert_allclose(grid[0, 0], expected)

    def test_full_weight_single_state(self):
        probs = np.array([[[0.6, 0.4]]], dtype=float)
        result = _FakeResult(probs, ["safe", "dangerous"])
        grid = extract_risk_grid(result, "risk", {"dangerous": 1.0})
        np.testing.assert_allclose(grid, np.array([[0.4]]))

    def test_dict_clipped_to_one(self):
        """Weighted sum > 1 should be clipped to 1."""
        probs = np.array([[[0.6, 0.4]]], dtype=float)
        result = _FakeResult(probs, ["a", "b"])
        # Weight 3.0 on "b" (0.4) → 1.2 → should clip to 1.0
        grid = extract_risk_grid(result, "risk", {"a": 3.0, "b": 3.0})
        assert float(grid[0, 0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NaN preservation
# ---------------------------------------------------------------------------

class TestNaNPreservation:
    def test_nan_preserved_string(self):
        probs = np.array([[[np.nan, np.nan], [0.4, 0.6]]], dtype=float)
        result = _FakeResult(probs, ["safe", "dangerous"])
        grid = extract_risk_grid(result, "risk", "dangerous")
        assert np.isnan(grid[0, 0])
        assert not np.isnan(grid[0, 1])

    def test_nan_preserved_dict(self):
        probs = np.array([[[np.nan, np.nan], [0.3, 0.7]]], dtype=float)
        result = _FakeResult(probs, ["safe", "dangerous"])
        grid = extract_risk_grid(result, "risk", {"dangerous": 1.0})
        assert np.isnan(grid[0, 0])
        assert not np.isnan(grid[0, 1])


# ---------------------------------------------------------------------------
# Clipping
# ---------------------------------------------------------------------------

class TestClipping:
    def test_values_clipped_to_zero_one(self):
        probs = np.array([[[1.5, -0.5]]], dtype=float)  # pathological input
        result = _FakeResult(probs, ["a", "b"])
        grid = extract_risk_grid(result, "risk", "a")
        assert grid[0, 0] == pytest.approx(1.0)

    def test_negative_clipped_to_zero(self):
        probs = np.array([[[-0.1, 1.1]]], dtype=float)
        result = _FakeResult(probs, ["a", "b"])
        grid = extract_risk_grid(result, "risk", "b")
        assert grid[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bad input type
# ---------------------------------------------------------------------------

class TestBadInput:
    def test_invalid_risk_state_type_raises(self):
        probs = np.ones((2, 2, 2), dtype=float) * 0.5
        result = _FakeResult(probs, ["safe", "dangerous"])
        with pytest.raises(TypeError):
            extract_risk_grid(result, "risk", 42)  # type: ignore[arg-type]
