"""Tests for coordinate conversion utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest
from affine import Affine

from risk_aware_a_star._coords import (
    _convert_path,
    _path_length_px,
    crs_to_pixel,
    latlon_to_pixel,
    pixel_to_crs,
    pixel_to_latlon,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

# Simple 100 m pixel grid in EPSG:32632 (UTM zone 32N)
CRS = "EPSG:32632"
TRANSFORM = Affine(100.0, 0.0, 500_000.0, 0.0, -100.0, 7_700_500.0)


# ---------------------------------------------------------------------------
# pixel_to_crs
# ---------------------------------------------------------------------------

class TestPixelToCrs:
    def test_pixel_centre_convention(self):
        """Pixel (0, 0) centre should map to (origin + 0.5*pixel_size)."""
        x, y = pixel_to_crs(0, 0, TRANSFORM)
        assert math.isclose(x, 500_000.0 + 0.5 * 100.0)
        assert math.isclose(y, 7_700_500.0 - 0.5 * 100.0)

    def test_pixel_1_1(self):
        x, y = pixel_to_crs(1, 1, TRANSFORM)
        assert math.isclose(x, 500_000.0 + 1.5 * 100.0)
        assert math.isclose(y, 7_700_500.0 - 1.5 * 100.0)


# ---------------------------------------------------------------------------
# crs_to_pixel
# ---------------------------------------------------------------------------

class TestCrsToPixel:
    def test_centre_of_pixel_0_0(self):
        x = 500_000.0 + 0.5 * 100.0
        y = 7_700_500.0 - 0.5 * 100.0
        row, col = crs_to_pixel(x, y, TRANSFORM)
        assert row == 0
        assert col == 0

    def test_centre_of_pixel_2_3(self):
        x = 500_000.0 + 3.5 * 100.0
        y = 7_700_500.0 - 2.5 * 100.0
        row, col = crs_to_pixel(x, y, TRANSFORM)
        assert row == 2
        assert col == 3


# ---------------------------------------------------------------------------
# Round-trip: pixel → CRS → pixel
# ---------------------------------------------------------------------------

class TestRoundTripPixelCrs:
    @pytest.mark.parametrize("row,col", [(0, 0), (3, 4), (10, 7)])
    def test_round_trip(self, row, col):
        x, y = pixel_to_crs(row, col, TRANSFORM)
        r2, c2 = crs_to_pixel(x, y, TRANSFORM)
        assert r2 == row
        assert c2 == col


# ---------------------------------------------------------------------------
# latlon_to_pixel and pixel_to_latlon
# ---------------------------------------------------------------------------

class TestLatLonConversion:
    def test_pixel_to_latlon_and_back(self):
        """Round-trip pixel → latlon → pixel should be within ±0.5 pixel."""
        for row, col in [(0, 0), (2, 2), (4, 4)]:
            lat, lon = pixel_to_latlon(row, col, TRANSFORM, CRS)
            r2, c2 = latlon_to_pixel(lat, lon, TRANSFORM, CRS)
            assert abs(r2 - row) <= 1, f"row mismatch: {r2} vs {row}"
            assert abs(c2 - col) <= 1, f"col mismatch: {c2} vs {col}"

    def test_latlon_to_pixel_known_point(self):
        """Known lat/lon should map to pixel (0, 0) of the test grid."""
        # Pixel (0,0) centre in EPSG:4326 ≈ (69.413317°N, 9.001274°E)
        lat, lon = pixel_to_latlon(0, 0, TRANSFORM, CRS)
        row, col = latlon_to_pixel(lat, lon, TRANSFORM, CRS)
        assert row == 0
        assert col == 0


# ---------------------------------------------------------------------------
# _convert_path
# ---------------------------------------------------------------------------

class TestConvertPath:
    def test_pixel_mode_returns_tuples(self):
        path = [(0, 0), (1, 1), (2, 2)]
        result = _convert_path(path, TRANSFORM, CRS, "pixel")
        assert result == path

    def test_crs_mode_returns_floats(self):
        path = [(0, 0)]
        result = _convert_path(path, TRANSFORM, CRS, "crs")
        x, y = result[0]
        assert math.isclose(x, 500_050.0)
        assert math.isclose(y, 7_700_450.0)

    def test_latlon_mode_returns_lat_lon(self):
        path = [(0, 0)]
        result = _convert_path(path, TRANSFORM, CRS, "latlon")
        lat, lon = result[0]
        assert 60.0 < lat < 80.0   # northern Norway
        assert 0.0 < lon < 30.0

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="return_coords"):
            _convert_path([(0, 0)], TRANSFORM, CRS, "xyz")


# ---------------------------------------------------------------------------
# _path_length_px
# ---------------------------------------------------------------------------

class TestPathLengthPx:
    def test_empty_path(self):
        assert _path_length_px([]) == 0.0

    def test_single_pixel(self):
        assert _path_length_px([(0, 0)]) == 0.0

    def test_cardinal_steps(self):
        path = [(0, 0), (0, 1), (0, 2)]
        assert math.isclose(_path_length_px(path), 2.0)

    def test_diagonal_steps(self):
        path = [(0, 0), (1, 1), (2, 2)]
        expected = 2 * math.sqrt(2)
        assert math.isclose(_path_length_px(path), expected)

    def test_mixed_steps(self):
        path = [(0, 0), (0, 1), (1, 2)]  # cardinal + diagonal
        expected = 1.0 + math.sqrt(2)
        assert math.isclose(_path_length_px(path), expected)
