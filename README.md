# risk-aware-a-star

**Risk-informed A\* path planning using Bayesian network risk models**

![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue)
![uv](https://img.shields.io/badge/managed%20with-uv-purple)

---

## Real-time Bayesian-network risk planning

`risk-aware-a-star` runs full Bayesian-network inference across a spatial grid and plans a
risk-optimal route — **all in milliseconds**. Timing breaks into three phases: `precompute()`
executes all evidence-state combinations once and caches the inference table; the **risk map**
phase applies the cached table to the full grid (zero pgmpy calls); and **A\*** searches the
resulting risk grid for the optimal path.

| BN | Nodes | Grid | State combos | Precompute | Risk map (median) | A* path (median) |
|----|-------|------|-------------|-----------|------------------|------------------|
| slope → risk | 2 | 400 × 400 | 2 | 1 ms | 1.6 ms | 397 ms |
| AUV (8-node) | 8 | 400 × 400 | 243 | 27 ms | 6 ms | 454 ms |

> Benchmarked with `tests/test_benchmark.py` on a MacBook Air 2025 (M4, 16 GB).

---

## Overview

`risk-aware-a-star` solves the problem of path planning through terrain or environments where risk
is spatially variable modeled through Bayesian networks. Traditional shortest-path
algorithms minimise distance; `risk-aware-a-star` minimises a combined objective that trades
distance against risk, producing routes that are longer when necessary to avoid
high-probability hazard zones.

The library pairs a [`geobn`](https://github.com/your-org/geobn) Bayesian network
with an A\* planner. `geobn` ingests raster inputs (GeoTIFFs, NumPy arrays, or
scalar constants), runs discrete Bayesian inference over a 2-D spatial grid, and
produces a per-pixel marginal probability for any node in the network, for example, collision probability. `risk-aware-a-star`
extracts that marginal as a risk grid and feeds it into A\* with a risk-weighted step
cost, returning an ordered list of waypoints together with summary statistics.

Typical use cases include AUV mission planning over seafloor hazard maps, mountaineering
and search-and-rescue route optimisation over avalanche or terrain-difficulty models,
and any domain where a spatial risk raster can be derived from a probabilistic model.
The public API is deliberately minimal: construct a `RiskAwareAStarPlanner`, call
`bn.precompute([risk_node])` offline (or `load_precomputed()` at runtime), then call `find_path()` for each query.

---

## How it works

```
OFFLINE (workstation)
───────────────────────────────────────────────────────────────────
geobn BayesianNetwork
        │
bn.precompute([risk_node])  →  bn.save_precomputed("table.npz")

RUNTIME (robot)
───────────────────────────────────────────────────────────────────
geobn BayesianNetwork  +  load_precomputed("table.npz")
                        │
                   infer()  →  extract_risk_grid()  →  2-D risk array  [0, 1]
                                                               │
                                               A* (risk-weighted cost)
                                                               │
                                                         PathResult
                                               (waypoints · cost · distance)
```

**Step cost formula**

```
step_cost = dist_px × (1 + risk_weight × risk[r, c])
```

`dist_px` is 1.0 for cardinal steps and √2 for diagonal steps (8-connectivity).
`risk[r, c]` is the marginal probability (or weighted combination of state
probabilities) at that cell, in [0, 1].

Grid cells where inference returns `NaN` (outside the data extent, masked water,
etc.) are treated as impassable barriers — A\* will never route through them.

---

## Installation

```bash
# Using uv (recommended)
uv add risk-aware-a-star

# Or with pip
pip install risk-aware-a-star
```

**Requirements:** Python ≥ 3.11, `geobn>=0.1`, `numpy`, `pyproj`, `affine`.

---

## Quick start — AUV mission planning

**OFFLINE** — run once on a workstation to build the inference table:

```python
import geobn

bn = geobn.load("auv_mission.bif")
bn.precompute(["auv_risk"])
bn.save_precomputed("table.npz")
```

**RUNTIME** — load the cached table on the robot (no pgmpy calls):

```python
import geobn
from risk_aware_a_star import RiskAwareAStarPlanner

# BN must be fully configured with the same inputs and discretizations as offline.
# The .npz caches the combinatorial inference table, not the spatial rasters.
bn = geobn.load("auv_mission.bif")
bn.set_input("Slope",             geobn.RasterSource("slope.tif"))
bn.set_input("Ruggedness",        geobn.RasterSource("ruggedness.tif"))
bn.set_input("Current",           geobn.RasterSource("current.tif"))
bn.set_input("Altitude_setpoint", geobn.ConstantSource(5.0))

bn.set_discretization("Slope",             [0,     10,    15,    90],  ["Low", "Medium", "High"])
bn.set_discretization("Ruggedness",        [0,     0.005, 0.01,  1.0], ["Low", "Medium", "High"])
bn.set_discretization("Current",           [0,     0.08,  0.15,  2.0], ["Low", "Medium", "High"])
bn.set_discretization("Altitude_setpoint", [0,     5,     20,  200],   ["Close", "Moderate", "Far"])

planner = RiskAwareAStarPlanner(
    bn=bn,
    risk_node="auv_risk",
    risk_state={"Medium": 0.5, "High": 1.0},
    risk_weight=5.0,
    connectivity=8,
)
planner.load_precomputed("table.npz")

# ── Plan a mission route ──────────────────────────────────────────────────────
START = (61.300, 5.050)   # (lat, lon) WGS84 — replace with your start point
GOAL  = (61.318, 5.085)   # (lat, lon) WGS84 — replace with your goal point

result = planner.find_path(START, GOAL, return_coords="latlon")
print(f"{len(result.waypoints)} waypoints  "
      f"distance={result.total_distance_px:.0f} px  "
      f"cost={result.total_cost:.2f}")
```

The planner reprojects and resamples automatically; all inputs must overlap
spatially but need not share the same resolution or CRS.

---

## Replanning and reuse

### Multiple routes on a fixed risk map

```python
planner.load_precomputed("table.npz")

# Risk grid is computed once and cached automatically.
route_a = planner.find_path((61.300, 5.050), (61.318, 5.085))
route_b = planner.find_path((61.305, 5.060), (61.322, 5.090))
route_c = planner.find_path((61.310, 5.070), (61.320, 5.095))
# All three share the same risk grid — only A* runs for routes b and c.
```

### Dynamic replanning when sensor data changes

```python
import numpy as np

planner.load_precomputed("table.npz")

# Plan with initial current measurements.
route_1 = planner.find_path(START, GOAL)

# New current raster arrives from onboard sensor.
planner.update_input("Current", geobn.RasterSource("current_updated.tif"))

# Risk grid is recomputed automatically on the next find_path() call.
route_2 = planner.find_path(START, GOAL)
```

---

## API Reference

### `RiskAwareAStarPlanner`

```python
RiskAwareAStarPlanner(bn, risk_node, risk_state, risk_weight=1.0, connectivity=8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bn` | `geobn.BayesianNetwork` | required | Configured BN with inputs and discretizations already set |
| `risk_node` | `str` | required | Name of the BN node whose marginal represents risk |
| `risk_state` | `str \| dict[str, float]` | required | State name or weighted dict, e.g. `{"Medium": 0.5, "High": 1.0}` |
| `risk_weight` | `float` | `1.0` | Trade-off factor; higher = more risk-averse, longer but safer routes |
| `connectivity` | `int` | `8` | `4` (cardinal only) or `8` (cardinal + diagonal) |

#### Methods

**`load_precomputed(path) → None`**

Restore the inference table from a `.npz` file (runtime step, replaces `precompute()`).
No pgmpy calls are made. The `bn` object must still be fully configured with the same
inputs and discretizations used offline — the `.npz` caches the combinatorial table,
not the spatial rasters.

---

**`find_path(start, goal, return_coords="latlon") → PathResult`**

Plans a risk-aware path from `start` to `goal`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `start` | `(float, float)` | `(lat, lon)` in WGS84 |
| `goal` | `(float, float)` | `(lat, lon)` in WGS84 |
| `return_coords` | `str` | Output coordinate system for `PathResult.waypoints` |

`return_coords` options:

| Value | Output type | Description |
|-------|-------------|-------------|
| `"latlon"` | `(lat, lon)` float tuples | WGS84 geographic (default) |
| `"crs"` | `(x, y)` float tuples | Native CRS of the raster |
| `"pixel"` | `(row, col)` int tuples | Grid pixel indices |

Raises `RuntimeError` if `load_precomputed()` (or `bn.precompute()` before construction) has not been called, `ValueError` if `start` or `goal` is outside the grid bounds, and `RuntimeError` if no path exists between the two points.

---

### `PathResult`

Returned by `RiskAwareAStarPlanner.find_path()`.

| Attribute | Type | Description |
|-----------|------|-------------|
| `waypoints` | `list[tuple[float, float]]` | Ordered path in the requested coordinate system |
| `total_cost` | `float` | Cumulative A\* cost (distance × risk weighted) |
| `total_distance_px` | `float` | Path length in pixels (sum of Euclidean step distances) |
| `risk_grid` | `np.ndarray` | 2-D float array in [0, 1] used for planning |
| `inference_result` | `geobn.InferenceResult` | Full BN inference result (maps, GeoTIFFs, etc.) |

---

## Cost model

Each grid step from cell `(r, c)` to a neighbour costs:

```
step_cost = dist_px × (1 + risk_weight × risk[r, c])
```

where `dist_px` is 1.0 for cardinal steps and √2 for diagonal steps.

With `risk_weight=0` the planner reduces to the shortest Euclidean path.
Increasing `risk_weight` pushes routes progressively further around risky terrain
at the cost of additional travel distance. A value of 5–10 is a reasonable starting
point for most applications.

---

## Impassable cells

Grid cells where the BN inference returns `NaN` are treated as impassable. This
covers pixels that fall outside the extent of any input raster, masked ocean cells,
no-data regions, etc. A\* will never include them in a route. If `start` or `goal`
maps to a NaN cell, `find_path()` raises a `ValueError` before the search begins.

---

## Examples

### Lyngen Alps — avalanche-risk ski touring

```bash
uv run examples/lyngen_alps_route/run_example.py
```

Synthetic 100×100 grid at 100 m resolution (EPSG:32633) over the Lyngen Alps,
Norway. Two-node BN: `slope` + `weather_index` → `avalanche_risk`. Demonstrates
multi-state weighted risk and a 5-km NW→SE route through variable terrain.
Outputs `lyngen_risk_map.html` in the current directory.

### Tautra — AUV seafloor route

Real GeoTIFF rasters at ~10 m resolution (374×358 grid) over the Tautra reef,
Trondheim Fjord. Eight-node BN with two intermediate nodes. Plans a NW→SE
risk-optimal AUV route across the reef, avoiding shallow and high-current zones.
Demonstrates the offline / runtime two-phase pattern.

```bash
# Step 1 — offline: build and save the inference table
uv run examples/tautra_route/precompute_offline.py   # creates tautra_table.npz

# Step 2 — runtime: load table and plan the route
uv run examples/tautra_route/run_auv_route.py        # opens auv_route_map.html
```

---

## Project layout

```
risk-star/
├── src/risk_aware_a_star/
│   ├── __init__.py        # public exports: RiskAwareAStarPlanner, PathResult
│   ├── planner.py         # RiskAwareAStarPlanner + PathResult dataclass
│   ├── _astar.py          # pure-Python heapq A* implementation
│   ├── _risk.py           # extract_risk_grid() helper
│   └── _coords.py         # coordinate conversion utilities
├── examples/
│   ├── lyngen_alps_route/ # avalanche-risk mountain routing
│   ├── tautra/            # AUV risk assessment (no path planning)
│   └── tautra_route/      # AUV risk-optimal route planning
├── tests/                 # 62 tests, ~99 % coverage
└── pyproject.toml
```

---

## Running tests

```bash
uv run pytest
uv run pytest --cov=risk_aware_a_star --cov-report=term-missing
```
