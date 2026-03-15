"""Benchmark: three-phase timing (precompute / risk map / A*) on a 400×400 grid.

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

import geobn

from risk_aware_a_star._astar import astar
from risk_aware_a_star._risk import extract_risk_grid

# ---------------------------------------------------------------------------
# Minimal 2-node BIF: slope → risk
# ---------------------------------------------------------------------------
_BIF_2NODE = """network bench_bn {
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

_BIF_8NODE = """network auv_risk {
}

variable Slope {
  type discrete [ 3 ] { Low, Medium, High };
}

variable Ruggedness {
  type discrete [ 3 ] { Low, Medium, High };
}

variable BPI {
  type discrete [ 3 ] { Low, Medium, High };
}

variable Current {
  type discrete [ 3 ] { Low, Medium, High };
}

variable Altitude_setpoint {
  type discrete [ 3 ] { Close, Moderate, Far };
}

variable Terrain_complexity {
  type discrete [ 3 ] { Low, Medium, High };
}

variable Navigation_difficulty {
  type discrete [ 3 ] { Low, Medium, High };
}

variable auv_risk {
  type discrete [ 3 ] { Low, Medium, High };
}

probability ( Slope ) {
  table 0.33300000, 0.33400000, 0.33300000;
}

probability ( Ruggedness ) {
  table 0.33300000, 0.33400000, 0.33300000;
}

probability ( BPI ) {
  table 0.33300000, 0.33400000, 0.33300000;
}

probability ( Current ) {
  table 0.33300000, 0.33400000, 0.33300000;
}

probability ( Altitude_setpoint ) {
  table 0.33300000, 0.33400000, 0.33300000;
}

probability ( Terrain_complexity | Slope, Ruggedness, BPI ) {
  (Low, Low, Low) 0.94000000, 0.05000000, 0.01000000;
  (Low, Low, Medium) 0.88000000, 0.10000000, 0.02000000;
  (Low, Low, High) 0.77000000, 0.18000000, 0.05000000;
  (Low, Medium, Low) 0.54000000, 0.38000000, 0.08000000;
  (Low, Medium, Medium) 0.38000000, 0.48000000, 0.14000000;
  (Low, Medium, High) 0.28000000, 0.50000000, 0.22000000;
  (Low, High, Low) 0.17000000, 0.48000000, 0.35000000;
  (Low, High, Medium) 0.13000000, 0.42000000, 0.45000000;
  (Low, High, High) 0.10000000, 0.35000000, 0.55000000;
  (Medium, Low, Low) 0.55000000, 0.35000000, 0.10000000;
  (Medium, Low, Medium) 0.40000000, 0.45000000, 0.15000000;
  (Medium, Low, High) 0.30000000, 0.50000000, 0.20000000;
  (Medium, Medium, Low) 0.22000000, 0.50000000, 0.28000000;
  (Medium, Medium, Medium) 0.15000000, 0.45000000, 0.40000000;
  (Medium, Medium, High) 0.10000000, 0.36000000, 0.54000000;
  (Medium, High, Low) 0.10000000, 0.28000000, 0.62000000;
  (Medium, High, Medium) 0.06000000, 0.20000000, 0.74000000;
  (Medium, High, High) 0.04000000, 0.14000000, 0.82000000;
  (High, Low, Low) 0.18000000, 0.52000000, 0.30000000;
  (High, Low, Medium) 0.15000000, 0.45000000, 0.40000000;
  (High, Low, High) 0.10000000, 0.38000000, 0.52000000;
  (High, Medium, Low) 0.06000000, 0.24000000, 0.70000000;
  (High, Medium, Medium) 0.04000000, 0.18000000, 0.78000000;
  (High, Medium, High) 0.03000000, 0.12000000, 0.85000000;
  (High, High, Low) 0.03000000, 0.12000000, 0.85000000;
  (High, High, Medium) 0.02000000, 0.08000000, 0.90000000;
  (High, High, High) 0.01000000, 0.04000000, 0.95000000;
}

probability ( Navigation_difficulty | Terrain_complexity, Current, Altitude_setpoint ) {
  (Low, Low, Close) 0.33000000, 0.50000000, 0.17000000;
  (Low, Low, Moderate) 0.65000000, 0.30000000, 0.05000000;
  (Low, Low, Far) 0.86000000, 0.12000000, 0.02000000;
  (Low, Medium, Close) 0.17000000, 0.48000000, 0.35000000;
  (Low, Medium, Moderate) 0.33000000, 0.50000000, 0.17000000;
  (Low, Medium, Far) 0.55000000, 0.35000000, 0.10000000;
  (Low, High, Close) 0.10000000, 0.33000000, 0.57000000;
  (Low, High, Moderate) 0.17000000, 0.45000000, 0.38000000;
  (Low, High, Far) 0.33000000, 0.50000000, 0.17000000;
  (Medium, Low, Close) 0.15000000, 0.45000000, 0.40000000;
  (Medium, Low, Moderate) 0.30000000, 0.50000000, 0.20000000;
  (Medium, Low, Far) 0.60000000, 0.32000000, 0.08000000;
  (Medium, Medium, Close) 0.10000000, 0.30000000, 0.60000000;
  (Medium, Medium, Moderate) 0.15000000, 0.45000000, 0.40000000;
  (Medium, Medium, Far) 0.30000000, 0.50000000, 0.20000000;
  (Medium, High, Close) 0.05000000, 0.20000000, 0.75000000;
  (Medium, High, Moderate) 0.10000000, 0.30000000, 0.60000000;
  (Medium, High, Far) 0.17000000, 0.45000000, 0.38000000;
  (High, Low, Close) 0.07000000, 0.28000000, 0.65000000;
  (High, Low, Moderate) 0.13000000, 0.42000000, 0.45000000;
  (High, Low, Far) 0.30000000, 0.48000000, 0.22000000;
  (High, Medium, Close) 0.03000000, 0.17000000, 0.80000000;
  (High, Medium, Moderate) 0.07000000, 0.28000000, 0.65000000;
  (High, Medium, Far) 0.13000000, 0.42000000, 0.45000000;
  (High, High, Close) 0.01000000, 0.08000000, 0.91000000;
  (High, High, Moderate) 0.03000000, 0.17000000, 0.80000000;
  (High, High, Far) 0.07000000, 0.28000000, 0.65000000;
}

probability ( auv_risk | Terrain_complexity, Navigation_difficulty ) {
  (Low, Low) 0.92000000, 0.07000000, 0.01000000;
  (Low, Medium) 0.70000000, 0.25000000, 0.05000000;
  (Low, High) 0.40000000, 0.45000000, 0.15000000;
  (Medium, Low) 0.55000000, 0.38000000, 0.07000000;
  (Medium, Medium) 0.25000000, 0.55000000, 0.20000000;
  (Medium, High) 0.10000000, 0.45000000, 0.45000000;
  (High, Low) 0.20000000, 0.55000000, 0.25000000;
  (High, Medium) 0.07000000, 0.38000000, 0.55000000;
  (High, High) 0.02000000, 0.18000000, 0.80000000;
}
"""

REPS = 20


@pytest.mark.benchmark
def test_benchmark_400x400(tmp_path):
    """2-node BN: precompute / risk map / A* on a 400×400 grid."""

    # --- Build grid ----------------------------------------------------------
    slope_data = np.full((ROWS, COLS), 5.0, dtype=np.float32)
    slope_data[150:250, :180] = 25.0
    slope_data[150:250, 220:] = 25.0

    # --- Configure BN --------------------------------------------------------
    bif_file = tmp_path / "bench.bif"
    bif_file.write_text(_BIF_2NODE)

    bn = geobn.load(str(bif_file))
    bn.set_input("slope", geobn.ArraySource(slope_data, crs=CRS, transform=TRANSFORM))
    bn.set_discretization("slope", [0, 20, 90], labels=["low", "high"])

    # --- Benchmark precompute() ----------------------------------------------
    t0 = time.perf_counter()
    bn.precompute(query=["risk"])
    precompute_ms = (time.perf_counter() - t0) * 1_000

    # --- Benchmark risk map (infer + extract) --------------------------------
    risk_map_times: list[float] = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        result = bn.infer(query=["risk"])
        extract_risk_grid(result, "risk", "dangerous")
        risk_map_times.append((time.perf_counter() - t0) * 1_000)

    risk_map_ms = statistics.median(risk_map_times)

    # --- Build risk grid once for A* benchmark -------------------------------
    result = bn.infer(query=["risk"])
    risk_grid = extract_risk_grid(result, "risk", "dangerous")

    # --- Benchmark A* --------------------------------------------------------
    astar_times: list[float] = []
    start_px = (0, 0)
    goal_px = (ROWS - 1, COLS - 1)
    for _ in range(REPS):
        t0 = time.perf_counter()
        astar(risk_grid, start_px, goal_px, 5.0, 8)
        astar_times.append((time.perf_counter() - t0) * 1_000)

    astar_ms = statistics.median(astar_times)

    # --- Report --------------------------------------------------------------
    print()
    print("=" * 62)
    print("BENCHMARK: 2-node BN  (slope → risk)")
    print(f"  Grid:           {ROWS} × {COLS} px")
    print(f"  State combos:   2   (1 input node × 2 states)")
    print(f"  precompute():   {precompute_ms:.1f} ms")
    print(f"  risk map:       {risk_map_ms:.2f} ms   (median of {REPS} reps)")
    print(f"  A* path:        {astar_ms:.2f} ms   (median of {REPS} reps)")
    print("=" * 62)


@pytest.mark.benchmark
def test_benchmark_8node_400x400(tmp_path):
    """8-node AUV BN: precompute / risk map / A* on a 400×400 grid."""

    bif_file = tmp_path / "auv_risk.bif"
    bif_file.write_text(_BIF_8NODE)

    # --- Build synthetic rasters ---------------------------------------------
    slope_data = np.full((ROWS, COLS), 5.0, dtype=np.float32)
    slope_data[150:250, :180] = 20.0
    slope_data[150:250, 220:] = 20.0

    ruggedness_data = np.full((ROWS, COLS), 0.003, dtype=np.float32)
    bpi_data = np.full((ROWS, COLS), 0.2, dtype=np.float32)

    current_data = np.full((ROWS, COLS), 0.05, dtype=np.float32)
    current_data[150:250, :180] = 0.18
    current_data[150:250, 220:] = 0.18

    # --- Configure BN --------------------------------------------------------
    bn = geobn.load(str(bif_file))
    bn.set_input("Slope",             geobn.ArraySource(slope_data,     crs=CRS, transform=TRANSFORM))
    bn.set_input("Ruggedness",        geobn.ArraySource(ruggedness_data, crs=CRS, transform=TRANSFORM))
    bn.set_input("BPI",               geobn.ArraySource(bpi_data,        crs=CRS, transform=TRANSFORM))
    bn.set_input("Current",           geobn.ArraySource(current_data,    crs=CRS, transform=TRANSFORM))
    bn.set_input("Altitude_setpoint", geobn.ConstantSource(5.0))

    bn.set_discretization("Slope",             [0, 10, 15, 90],   labels=["Low", "Medium", "High"])
    bn.set_discretization("Ruggedness",        [0, 0.005, 0.01, 1.0], labels=["Low", "Medium", "High"])
    bn.set_discretization("BPI",               [0, 0.3, 0.7, 2.0],   labels=["Low", "Medium", "High"])
    bn.set_discretization("Current",           [0, 0.08, 0.15, 2.0], labels=["Low", "Medium", "High"])
    bn.set_discretization("Altitude_setpoint", [0, 5, 20, 200],      labels=["Close", "Moderate", "Far"])

    # --- Benchmark precompute() ----------------------------------------------
    t0 = time.perf_counter()
    bn.precompute(query=["auv_risk"])
    precompute_ms = (time.perf_counter() - t0) * 1_000

    # --- Benchmark risk map (infer + extract) --------------------------------
    risk_state = {"Medium": 0.5, "High": 1.0}
    risk_map_times: list[float] = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        result = bn.infer(query=["auv_risk"])
        extract_risk_grid(result, "auv_risk", risk_state)
        risk_map_times.append((time.perf_counter() - t0) * 1_000)

    risk_map_ms = statistics.median(risk_map_times)

    # --- Build risk grid once for A* benchmark -------------------------------
    result = bn.infer(query=["auv_risk"])
    risk_grid = extract_risk_grid(result, "auv_risk", risk_state)

    # --- Benchmark A* --------------------------------------------------------
    astar_times: list[float] = []
    start_px = (0, 0)
    goal_px = (ROWS - 1, COLS - 1)
    for _ in range(REPS):
        t0 = time.perf_counter()
        astar(risk_grid, start_px, goal_px, 5.0, 8)
        astar_times.append((time.perf_counter() - t0) * 1_000)

    astar_ms = statistics.median(astar_times)

    # --- Report --------------------------------------------------------------
    print()
    print("=" * 62)
    print("BENCHMARK: 8-node AUV BN  (5 inputs, 2 latent, 1 output)")
    print(f"  Grid:           {ROWS} × {COLS} px")
    print(f"  State combos:   243   (5 input nodes × 3 states = 3^5)")
    print(f"  precompute():   {precompute_ms:.1f} ms")
    print(f"  risk map:       {risk_map_ms:.2f} ms   (median of {REPS} reps)")
    print(f"  A* path:        {astar_ms:.2f} ms   (median of {REPS} reps)")
    print("=" * 62)
