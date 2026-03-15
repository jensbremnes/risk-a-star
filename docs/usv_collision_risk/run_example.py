"""USV collision risk — coastal passage planning example.

Synthetic 80×80 grid (EPSG:32632, 100 m resolution).
Two-node BN: vessel_traffic + sea_state → collision_risk.

Run with:
    uv run python docs/usv_collision_risk/run_example.py
"""

from pathlib import Path

import numpy as np
from affine import Affine
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import geobn

from risk_aware_a_star._astar import astar
from risk_aware_a_star._risk import extract_risk_grid

HERE = Path(__file__).parent
DOCS = HERE.parent
GIF_OUT = DOCS / "assets" / "demo.gif"

ROWS = COLS = 80
CRS = "EPSG:32632"
RESOLUTION = 100.0          # metres per pixel
XMIN, YMIN = 400_000.0, 5_000_000.0
TRANSFORM = Affine(RESOLUTION, 0.0, XMIN, 0.0, -RESOLUTION, YMIN + ROWS * RESOLUTION)

START = (10, 5)    # (row, col) — NW corner area
GOAL  = (68, 72)   # (row, col) — SE corner area


# ---------------------------------------------------------------------------
# Synthetic rasters
# ---------------------------------------------------------------------------

def _vessel_traffic(rows: int, cols: int) -> np.ndarray:
    """Two Gaussian blobs: a diagonal shipping lane + a port approach zone."""
    rr, cc = np.mgrid[0:rows, 0:cols]
    # Shipping lane: diagonal band from NE corner toward SW
    lane_dist = np.abs((rr / rows) - (1.0 - cc / cols)) * np.sqrt(2) / 2
    lane = np.exp(-0.5 * (lane_dist / 0.12) ** 2)
    # Port approach: near the goal corner (SE)
    port_r, port_c = rows * 0.85, cols * 0.85
    port_dist = np.sqrt(((rr - port_r) / rows) ** 2 + ((cc - port_c) / cols) ** 2)
    port = np.exp(-0.5 * (port_dist / 0.15) ** 2)
    traffic = np.clip(lane + port, 0.0, 1.0).astype(np.float32)
    return traffic


def _sea_state(rows: int, cols: int) -> np.ndarray:
    """Smooth SW→NE gradient: sheltered in SW, rougher in NE."""
    rr, cc = np.mgrid[0:rows, 0:cols]
    # NE corner (low row index, high col index) is more exposed
    val = (cc / (cols - 1) + (rows - 1 - rr) / (rows - 1)) / 2.0
    return val.astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

traffic_arr = _vessel_traffic(ROWS, COLS)
sea_arr = _sea_state(ROWS, COLS)

traffic_src = geobn.ArraySource(traffic_arr, crs=CRS, transform=TRANSFORM)
sea_src     = geobn.ArraySource(sea_arr,     crs=CRS, transform=TRANSFORM)

bn = geobn.load(str(HERE / "usv_collision.bif"))
bn.set_input("vessel_traffic", traffic_src)
bn.set_input("sea_state",      sea_src)
bn.set_discretization("vessel_traffic", [0.0, 0.35, 0.65, 1.0], labels=["Low", "Medium", "High"])
bn.set_discretization("sea_state",      [0.0, 0.5,  1.0],       labels=["Calm", "Rough"])

bn.precompute(["collision_risk"])
result = bn.infer(["collision_risk"])

risk_grid = extract_risk_grid(result, "collision_risk", {"Medium": 0.5, "High": 1.0})

# Plan two routes
naive_result  = astar(risk_grid, START, GOAL, risk_weight=0.0, connectivity=8)
aware_result  = astar(risk_grid, START, GOAL, risk_weight=6.0, connectivity=8)

naive_path = naive_result[0]  if naive_result  else []
aware_path = aware_result[0]  if aware_result  else []


def _path_xy(path):
    """Convert (row, col) list to (col_vals, row_vals) for imshow overlay."""
    if not path:
        return [], []
    rows_p, cols_p = zip(*path)
    return list(cols_p), list(rows_p)


# ---------------------------------------------------------------------------
# Build 4-frame animated GIF
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(7, 6), dpi=120)
fig.patch.set_facecolor("#1a1a2e")
for ax in axes.flat:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

ax_tl, ax_tr, ax_bl, ax_br = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# Pre-render static images
im_traffic = ax_tl.imshow(traffic_arr, origin="upper", cmap="YlOrRd", vmin=0, vmax=1, interpolation="bilinear")
ax_tl.set_title("Vessel traffic", color="white", fontsize=9, pad=4)

im_risk_tr = ax_tr.imshow(risk_grid, origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
ax_tr.set_title("Collision risk — Bayesian network", color="white", fontsize=9, pad=4)

im_risk_bl = ax_bl.imshow(risk_grid, origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
ax_bl.set_title("Shortest path", color="white", fontsize=9, pad=4)
naive_x, naive_y = _path_xy(naive_path)
line_naive, = ax_bl.plot(naive_x, naive_y, color="white", linewidth=1.5, alpha=0.9)
ax_bl.plot(*_path_xy([START]),  "o", color="limegreen", markersize=5)
ax_bl.plot(*_path_xy([GOAL]),   "o", color="tomato",    markersize=5)

im_risk_br = ax_br.imshow(risk_grid, origin="upper", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="bilinear")
ax_br.set_title("Risk-aware route", color="white", fontsize=9, pad=4)
aware_x, aware_y = _path_xy(aware_path)
line_aware, = ax_br.plot(aware_x, aware_y, color="limegreen", linewidth=1.5, alpha=0.9)
ax_br.plot(*_path_xy([START]),  "o", color="limegreen", markersize=5)
ax_br.plot(*_path_xy([GOAL]),   "o", color="tomato",    markersize=5)

fig.tight_layout(pad=1.2)

# Frames: each frame shows one highlighted panel (border) while all four are visible
HIGHLIGHT_COLOR = "#00d4ff"
NORMAL_COLOR = "#1a1a2e"

frame_highlights = [ax_tl, ax_tr, ax_bl, ax_br]

writer = PillowWriter(fps=0.56)   # ≈1.8 s per frame
GIF_OUT.parent.mkdir(parents=True, exist_ok=True)

with writer.saving(fig, str(GIF_OUT), dpi=120):
    for focus_ax in frame_highlights:
        # Reset all borders
        for ax in axes.flat:
            for spine in ax.spines.values():
                spine.set_visible(False)
        # Highlight the focus panel
        for spine in focus_ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(HIGHLIGHT_COLOR)
            spine.set_linewidth(2.0)
        writer.grab_frame()

plt.close(fig)

naive_dist  = naive_result[1]  if naive_result  else float("nan")
aware_dist  = aware_result[1]  if aware_result  else float("nan")
print(f"Naive path:      {len(naive_path)} waypoints, cost={naive_dist:.1f}")
print(f"Risk-aware path: {len(aware_path)} waypoints, cost={aware_dist:.1f}")
print(f"GIF saved to:    {GIF_OUT}")
