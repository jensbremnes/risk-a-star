# riskstar

**Risk-aware A\* path planning over Bayesian-network risk fields**

![Python ≥3.11](https://img.shields.io/badge/python-%E2%89%A53.11-blue)
![uv](https://img.shields.io/badge/managed%20with-uv-purple)

---

## Overview

`riskstar` solves the problem of routing through terrain or environments where risk
is spatially variable and probabilistically modelled. Traditional shortest-path
algorithms minimise distance; `riskstar` minimises a combined objective that trades
distance against risk, producing routes that are longer when necessary to avoid
high-probability hazard zones.

The library pairs a [`geobn`](https://github.com/your-org/geobn) Bayesian network
with an A\* planner. `geobn` ingests raster inputs (GeoTIFFs, NumPy arrays, or
scalar constants), runs discrete Bayesian inference over a 2-D spatial grid, and
produces a per-pixel marginal probability for any node in the network — for example,
the probability that a grid cell is in the "high" avalanche-risk state. `riskstar`
extracts that marginal as a risk grid and feeds it into A\* with a risk-weighted step
cost, returning an ordered list of waypoints together with summary statistics.

Typical use cases include AUV mission planning over seafloor hazard maps, mountaineering
and search-and-rescue route optimisation over avalanche or terrain-difficulty models,
and any domain where a spatial risk raster can be derived from a probabilistic model.
The public API is deliberately minimal: construct a `RiskStarPlanner`, call
`precompute()` once, then call `find_path()` for each query.

---

## How it works

```
Raster inputs (GeoTIFF / NumPy array / constant)
        │
        ▼
geobn BayesianNetwork  ─── precompute() ──→  cached inference table
        │
        │  infer()
        ▼
InferenceResult  →  extract_risk_grid()  →  2-D risk array  [0, 1]
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
uv add riskstar

# Or with pip
pip install riskstar
```

**Requirements:** Python ≥ 3.11, `geobn>=0.1`, `numpy`, `pyproj`, `affine`.

---

## Quick start

The snippet below uses `ArraySource` so it requires no external data files.

```python
from pathlib import Path
import numpy as np
from affine import Affine
import geobn
from riskstar import RiskStarPlanner

# 1. Build a tiny synthetic DEM and attach it to a BN
rng = np.random.default_rng(42)
slope_arr = rng.uniform(0, 60, size=(100, 100)).astype(np.float32)
transform = Affine(100.0, 0, 693_000, 0, -100.0, 7_746_000)

bn = geobn.load("avalanche_risk.bif")
bn.set_input("slope", geobn.ArraySource(slope_arr, crs="EPSG:32633", transform=transform))
bn.set_input("weather_index", geobn.ConstantSource(1.5))
bn.set_discretization("slope", [0, 15, 35, 90], labels=["flat", "moderate", "steep"])
bn.set_discretization("weather_index", [0.0, 1.0, 3.0], labels=["low", "high"])

# 2. Create the planner and precompute
planner = RiskStarPlanner(
    bn=bn,
    risk_node="avalanche_risk",
    risk_state={"moderate": 0.5, "high": 1.0},  # weighted multi-state risk
    risk_weight=5.0,
    connectivity=8,
)
planner.precompute()

# 3. Find a risk-optimal path (WGS84 lat/lon in, WGS84 lat/lon out)
result = planner.find_path(
    start=(69.680, 20.060),
    goal=(69.725, 20.120),
    return_coords="latlon",
)
print(f"{len(result.waypoints)} waypoints, cost={result.total_cost:.2f}")

# 4. Export an interactive HTML risk map
result.inference_result.show_map(output_dir=".", filename="risk_map.html")
```

---

## API Reference

### `RiskStarPlanner`

```python
RiskStarPlanner(bn, risk_node, risk_state, risk_weight=1.0, connectivity=8)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bn` | `geobn.BayesianNetwork` | required | Configured BN with inputs and discretizations already set |
| `risk_node` | `str` | required | Name of the BN node whose marginal represents risk |
| `risk_state` | `str \| dict[str, float]` | required | State name or weighted dict, e.g. `{"Medium": 0.5, "High": 1.0}` |
| `risk_weight` | `float` | `1.0` | Trade-off factor; higher = more risk-averse, longer but safer routes |
| `connectivity` | `int` | `8` | `4` (cardinal only) or `8` (cardinal + diagonal) |

#### Methods

**`precompute() → None`**

Pre-runs all evidence-state combinations and caches the inference table.
Must be called once before `find_path()`. Subsequent `find_path()` calls use
the cached table for fast O(H×W) inference.

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

Raises `RuntimeError` if `precompute()` has not been called, `ValueError` if
`start` or `goal` is outside the grid bounds, and `RuntimeError` if no path
exists between the two points.

---

### `PathResult`

Returned by `RiskStarPlanner.find_path()`.

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

```bash
uv run examples/tautra_route/run_auv_route.py
```

Real GeoTIFF rasters at ~10 m resolution (374×358 grid) over the Tautra reef,
Trondheim Fjord. Eight-node BN with two intermediate nodes. Plans a NW→SE
risk-optimal AUV route across the reef, avoiding shallow and high-current zones.
Outputs `examples/tautra_route/output/auv_route_map.html`.

---

## Project layout

```
risk-star/
├── src/riskstar/
│   ├── __init__.py        # public exports: RiskStarPlanner, PathResult
│   ├── planner.py         # RiskStarPlanner + PathResult dataclass
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
uv run pytest --cov=riskstar --cov-report=term-missing
```
