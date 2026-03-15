"""Benchmark: precompute() and find_path() on a 400×400 grid.

Run with:
    uv run pytest tests/test_benchmark.py -s -v

Skip in CI:
    uv run pytest tests/ -m "not benchmark"
"""

from __future__ import annotations

import statistics
import tempfile
import time

import numpy as np
import pytest
from affine import Affine
from pyproj import Transformer

import geobn

from risk_aware_a_star import RiskAwareAStarPlanner

# ---------------------------------------------------------------------------
# Minimal 2-node BIF: slope → risk  (same topology as conftest)
# ---------------------------------------------------------------------------
_BIF = """network bench_bn {
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

ROWS, COLS = 400, 400
RES = 10.0
CRS = "EPSG:32632"
TRANSFORM = Affine(RES, 0.0, 500_000.0, 0.0, -RES, 7_700_000.0)

REPS = 20


@pytest.mark.benchmark
def test_benchmark_400x400(tmp_path):
    """Print precompute() and find_path() wall-clock times on a 400×400 grid."""

    # --- Build grid: mostly low-slope, partial high-risk band in the middle --
    # Leave a gap (cols 180-220) so A* can route around the band without
    # exhausting max_nodes on a fully-blocked row.
    slope_data = np.full((ROWS, COLS), 5.0, dtype=np.float32)
    slope_data[150:250, :180] = 25.0    # left portion of band (slope ≥ 20 → "high")
    slope_data[150:250, 220:] = 25.0    # right portion of band

    # --- Write BIF and configure BN ------------------------------------------
    bif_file = tmp_path / "bench.bif"
    bif_file.write_text(_BIF)

    bn = geobn.load(str(bif_file))
    bn.set_input("slope", geobn.ArraySource(slope_data, crs=CRS, transform=TRANSFORM))
    bn.set_discretization("slope", [0, 20, 90], labels=["low", "high"])

    planner = RiskAwareAStarPlanner(
        bn=bn,
        risk_node="risk",
        risk_state="dangerous",
        risk_weight=5.0,
        connectivity=8,
    )

    # --- Benchmark precompute() ----------------------------------------------
    t0 = time.perf_counter()
    planner.precompute()
    precompute_ms = (time.perf_counter() - t0) * 1_000

    # --- Compute lat/lon for top-left and bottom-right pixel centres ----------
    # Pixel-centre formula: x = origin_x + res*(col+0.5), y = origin_y - res*(row+0.5)
    to_wgs84 = Transformer.from_crs(CRS, "EPSG:4326", always_xy=False)

    def pixel_to_latlon(row: int, col: int) -> tuple[float, float]:
        x = TRANSFORM.c + TRANSFORM.a * (col + 0.5)
        y = TRANSFORM.f + TRANSFORM.e * (row + 0.5)
        lat, lon = to_wgs84.transform(x, y)
        return (lat, lon)

    start_ll = pixel_to_latlon(0, 0)
    goal_ll = pixel_to_latlon(ROWS - 1, COLS - 1)

    # --- Benchmark find_path() -----------------------------------------------
    times_ms: list[float] = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        planner.find_path(start_ll, goal_ll, return_coords="pixel")
        times_ms.append((time.perf_counter() - t0) * 1_000)

    median_ms = statistics.median(times_ms)
    min_ms = min(times_ms)

    # --- Report --------------------------------------------------------------
    print()
    print("=" * 58)
    print(f"  Grid:          {ROWS} × {COLS} px")
    print(f"  BN topology:   2-node (slope → risk)")
    print(f"  precompute():  {precompute_ms:.1f} ms")
    print(f"  find_path():   {median_ms:.2f} ms  (median of {REPS} reps)")
    print(f"  find_path():   {min_ms:.2f} ms  (min)")
    print("=" * 58)
