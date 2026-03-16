"""Coordinate conversion utilities for riskstar."""

from __future__ import annotations

import math

from affine import Affine
from pyproj import Transformer


def pixel_to_crs(row: int | float, col: int | float, transform: Affine) -> tuple[float, float]:
    """Convert pixel (row, col) to CRS (x, y) using pixel-centre convention."""
    x, y = transform * (col + 0.5, row + 0.5)
    return x, y


def crs_to_pixel(x: float, y: float, transform: Affine) -> tuple[int, int]:
    """Convert CRS (x, y) to pixel (row, col) using pixel-centre convention."""
    inv = ~transform
    col_f, row_f = inv * (x, y)
    # Pixel-centre: subtract 0.5 before floor to get the pixel index
    row = int(math.floor(row_f - 0.5 + 0.5))
    col = int(math.floor(col_f - 0.5 + 0.5))
    return row, col


def latlon_to_pixel(
    lat: float, lon: float, transform: Affine, crs: str
) -> tuple[int, int]:
    """Convert WGS84 (lat, lon) to pixel (row, col)."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return crs_to_pixel(x, y, transform)


def pixel_to_latlon(
    row: int | float, col: int | float, transform: Affine, crs: str
) -> tuple[float, float]:
    """Convert pixel (row, col) to WGS84 (lat, lon)."""
    x, y = pixel_to_crs(row, col, transform)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(x, y)
    return lat, lon


def _convert_path(
    pixel_path: list[tuple[int, int]],
    transform: Affine,
    crs: str,
    return_coords: str,
) -> list[tuple[float, float]]:
    """Convert a pixel path to the requested coordinate system.

    Parameters
    ----------
    return_coords:
        ``"latlon"`` → WGS84 (lat, lon),
        ``"crs"``    → native CRS (x, y),
        ``"pixel"``  → (row, col) as floats.
    """
    if return_coords == "pixel":
        return list(pixel_path)
    if return_coords == "crs":
        return [pixel_to_crs(r, c, transform) for r, c in pixel_path]
    if return_coords == "latlon":
        return [pixel_to_latlon(r, c, transform, crs) for r, c in pixel_path]
    raise ValueError(f"Unknown return_coords: {return_coords!r}. Use 'latlon', 'crs', or 'pixel'.")


def _path_length_px(pixel_path: list[tuple[int, int]]) -> float:
    """Sum Euclidean step distances along a pixel path (1.0 cardinal, √2 diagonal)."""
    if len(pixel_path) < 2:
        return 0.0
    total = 0.0
    for (r1, c1), (r2, c2) in zip(pixel_path, pixel_path[1:]):
        total += math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
    return total
