"""Spatial maximum filter (risk inflation) for the risk grid."""
from __future__ import annotations
import math
import numpy as np
from pyproj import CRS as ProjCRS


def _pixel_size_metres(transform, crs_str: str) -> tuple[float, float]:
    """Return (dx_m, dy_m) — pixel size in metres.

    For geographic CRS the x-size is latitude-corrected using the grid's
    y-origin (top edge) as a reference latitude — close enough for buffer
    radius purposes.
    """
    dx = abs(transform.a)   # CRS units / pixel (x)
    dy = abs(transform.e)   # CRS units / pixel (y)
    proj_crs = ProjCRS.from_user_input(crs_str)
    if proj_crs.is_geographic:
        ref_lat = transform.f   # y_origin ≈ top-left latitude
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(ref_lat))
        return dx * m_per_deg_lon, dy * m_per_deg_lat
    return dx, dy


def inflate_risk(
    risk_grid: np.ndarray,
    radius_m: float,
    transform,
    crs_str: str,
) -> np.ndarray:
    """Apply a circular maximum filter to *risk_grid*.

    Each output cell receives the highest risk value found within *radius_m*
    metres of that cell.  NaN cells (land / impassable) are never the source
    of a max value and always remain NaN in the output.

    Parameters
    ----------
    risk_grid:
        2-D float array in [0, 1] with NaN for impassable cells.
    radius_m:
        Buffer radius in metres.  0 → identity (no change).
    transform:
        Affine transform of the grid (from ``InferenceResult.transform``).
    crs_str:
        CRS string (from ``InferenceResult.crs``).
    """
    if radius_m <= 0.0:
        return risk_grid

    dx_m, dy_m = _pixel_size_metres(transform, crs_str)
    rx = radius_m / dx_m   # radius in x-pixels
    ry = radius_m / dy_m   # radius in y-pixels
    r = int(math.ceil(max(rx, ry)))   # integer pixel radius (bounding box)

    # Build disk mask in pixel space
    cols_off, rows_off = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1))
    disk = (cols_off / rx) ** 2 + (rows_off / ry) ** 2 <= 1.0

    nan_mask = np.isnan(risk_grid)
    out = risk_grid.copy()

    for dr, dc in zip(rows_off[disk], cols_off[disk]):
        if dr == 0 and dc == 0:
            continue
        shifted = np.roll(np.roll(risk_grid, int(dr), axis=0), int(dc), axis=1)
        # Zero out wrapped edges so they don't bleed across borders
        if dr > 0:
            shifted[:dr, :] = np.nan
        elif dr < 0:
            shifted[dr:, :] = np.nan
        if dc > 0:
            shifted[:, :dc] = np.nan
        elif dc < 0:
            shifted[:, dc:] = np.nan
        out = np.fmax(out, shifted)   # fmax: NaN < any valid value

    out[nan_mask] = np.nan   # restore land pixels
    return out
