"""Karmsundet USV route planner — risk-aware A* demo.

Plans a route through the Karmsundet strait near Haugesund, Norway as weather
deteriorates from calm to storm across 6 frames. Outputs an animated GIF and
an interactive Leaflet risk map.

Run
---
    uv run --extra examples python examples/karmsundet_usv/run_example.py

Outputs
-------
    examples/karmsundet_usv/output/karmsundet_risk.gif  — 6-frame animated GIF
    examples/karmsundet_usv/output/usv_risk_map.html    — interactive Leaflet map
    docs/assets/demo.gif                                — README front-page GIF
"""
from __future__ import annotations

import shutil
import sys
from io import BytesIO
from pathlib import Path

import numpy as np

import geobn
from risk_aware_a_star import RiskAwareAStarPlanner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEST, SOUTH, EAST, NORTH = 5.00, 59.25, 5.55, 59.55
CRS, RESOLUTION = "EPSG:4326", 0.002          # 150 rows × 275 cols
START, GOAL = (59.44, 5.19), (59.29, 5.17)    # Haugesund harbour → south exit

HERE        = Path(__file__).parent
BIF_PATH    = HERE / "usv_risk.bif"
CACHE_DIR   = HERE / "cache"
OUT_DIR     = HERE / "output"
TABLE_PATH  = OUT_DIR / "table.npz"
DOCS_ASSETS = Path(__file__).parents[2] / "docs" / "assets"

FRAMES = [
    {"wave": 0.3, "wind":  2.0, "fog": 0.02, "label": "Calm • Clear"},
    {"wave": 0.8, "wind":  5.0, "fog": 0.08, "label": "Light chop"},
    {"wave": 1.4, "wind":  9.0, "fog": 0.22, "label": "Moderate sea"},
    {"wave": 2.0, "wind": 14.0, "fog": 0.42, "label": "Rough sea"},
    {"wave": 3.5, "wind": 23.0, "fog": 0.88, "label": "Storm + dense fog"},
]

# ---------------------------------------------------------------------------
# Local helpers (not part of the library)
# ---------------------------------------------------------------------------


def _synthetic_depth(rows: int, cols: int) -> np.ndarray:
    """Fallback bathymetry: channel cols 15–110 covers START col≈95, GOAL col≈85."""
    depth = np.full((rows, cols), np.nan, dtype=np.float32)
    for r in range(rows):
        depth[r, 15:110] = 10.0 + 70.0 * (r / max(rows - 1, 1))
    return depth


def _inject_path(
    html_path: Path,
    waypoints: list[tuple[float, float]],
    start_label: str = "Haugesund harbour",
    end_label: str = "South exit",
) -> None:
    """Inject a Leaflet PolyLine + start/end markers into a folium HTML file."""
    coords_js = str([[round(lat, 6), round(lon, 6)] for lat, lon in waypoints])
    script = f"""
<script>
(function () {{
    var mapObj = null;
    for (var key in window) {{
        if (key.startsWith('map_') && window[key] && window[key].addLayer) {{
            mapObj = window[key];
            break;
        }}
    }}
    if (!mapObj) return;

    var coords = {coords_js};

    L.polyline(coords, {{
        color: '#0055cc', weight: 5, opacity: 0.85, lineJoin: 'round'
    }}).addTo(mapObj).bindTooltip('Planned route', {{sticky: true}});

    L.circleMarker(coords[0], {{
        radius: 9, color: '#fff', weight: 2,
        fillColor: '#22aa44', fillOpacity: 1
    }}).addTo(mapObj).bindPopup('<b>{start_label}</b>');

    L.circleMarker(coords[coords.length - 1], {{
        radius: 9, color: '#fff', weight: 2,
        fillColor: '#cc2222', fillOpacity: 1
    }}).addTo(mapObj).bindPopup('<b>{end_label}</b>');
}})();
</script>
</html>"""

    html = html_path.read_text(encoding="utf-8")
    html = html.replace("</html>", script, 1)
    html_path.write_text(html, encoding="utf-8")


def _render_frame(
    sea_mask: np.ndarray,
    risk_grid: np.ndarray,
    waypoints: list[tuple[float, float]],
    frame: dict,
    frame_idx: int,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Return a 14×6-inch matplotlib figure for one weather frame."""
    import matplotlib.pyplot as plt

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
    extent = [WEST, EAST, SOUTH, NORTH]

    for ax in (ax_left, ax_right):
        ax.set_xlim(WEST, EAST)
        ax.set_ylim(SOUTH, NORTH)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # ------------------------------------------------------------------
    # Try contextily basemap; fall back to solid ocean colour
    # ------------------------------------------------------------------
    try:
        import contextily
        for ax in (ax_left, ax_right):
            contextily.add_basemap(
                ax,
                crs="EPSG:4326",
                source=contextily.providers.OpenTopoMap,
                zoom=11,
                attribution=False,
            )
    except Exception:
        for ax in (ax_left, ax_right):
            ax.set_facecolor("#b8d4e8")

    # ------------------------------------------------------------------
    # Left panel: static geography + start/goal markers
    # ------------------------------------------------------------------
    ax_left.set_title("Study area", fontsize=11)
    ax_left.plot(START[1], START[0], "g*", markersize=14,
                 label="Start (Haugesund)", zorder=5)
    ax_left.plot(GOAL[1], GOAL[0], "rD", markersize=10,
                 label="Goal (south exit)", zorder=5)
    ax_left.legend(fontsize=8, loc="lower left")

    # ------------------------------------------------------------------
    # Right panel: risk heatmap + planned path
    # ------------------------------------------------------------------
    risk_display = np.where(sea_mask, risk_grid, np.nan)
    ax_right.imshow(
        risk_display,
        cmap="RdYlGn_r",
        vmin=0.0, vmax=0.8,
        alpha=0.65,
        extent=extent,
        origin="upper",
        aspect="equal",
    )
    ax_right.set_title(f"Frame {frame_idx + 1}/{len(FRAMES)}: {frame['label']}",
                       fontsize=11)

    if waypoints:
        lons = [w[1] for w in waypoints]
        lats = [w[0] for w in waypoints]
        ax_right.plot(lons, lats, "-", color="#0055cc", linewidth=2.5,
                      alpha=0.9, zorder=6)
        ax_right.plot(START[1], START[0], "g*", markersize=14, zorder=7)
        ax_right.plot(GOAL[1],  GOAL[0],  "rD", markersize=10, zorder=7)

    title = (
        f"Frame {frame_idx + 1}/{len(FRAMES)}  •  {frame['label']}  •  "
        f"Wave {frame['wave']} m  |  Wind {frame['wind']} m/s  |  "
        f"Fog {frame['fog'] * 100:.0f} %"
    )
    fig.suptitle(title, y=0.02, fontsize=9, va="bottom")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def _fig_to_pil(fig: "matplotlib.figure.Figure") -> "PIL.Image.Image":  # type: ignore[name-defined]
    """Convert a matplotlib figure to a PIL Image (RGB)."""
    from PIL import Image

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    import matplotlib.pyplot as plt
    plt.close(fig)
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ASSETS.mkdir(parents=True, exist_ok=True)

    H = round((NORTH - SOUTH) / RESOLUTION)   # 150
    W = round((EAST  - WEST)  / RESOLUTION)   # 200

    print(f"Karmsundet USV route planner  ({H}×{W} grid)")
    print(f"Start: {START}  →  Goal: {GOAL}")

    # ------------------------------------------------------------------
    # 1. Load BN and configure grid
    # ------------------------------------------------------------------
    bn = geobn.load(BIF_PATH)
    bn.set_grid(CRS, RESOLUTION, (WEST, SOUTH, EAST, NORTH))

    # ------------------------------------------------------------------
    # 2. EMODnet bathymetry (cached WCS fetch; synthetic fallback)
    # ------------------------------------------------------------------
    print("\nFetching EMODnet bathymetry ...")
    try:
        raw_depth = bn.fetch_raw(geobn.WCSSource(
            url="https://ows.emodnet-bathymetry.eu/wcs",
            layer="emodnet:mean",
            version="2.0.1",
            valid_range=(-1000.0, 100.0),
            cache_dir=CACHE_DIR,
        ))
        depth = -raw_depth                  # flip: positive = depth below surface
        depth[depth < 0] = np.nan           # land pixels → NaN
        print(f"  WCS ok  (depth range {np.nanmin(depth):.0f}–{np.nanmax(depth):.0f} m)")
    except Exception as exc:
        print(f"  WCS unavailable ({exc}); using synthetic channel depth.")
        depth = _synthetic_depth(H, W)

    sea_mask = np.isfinite(depth)
    print(f"  Sea pixels: {int(sea_mask.sum()):,} / {H * W:,}")

    # ------------------------------------------------------------------
    # 3. Register ALL inputs in fixed order
    #    (must match between precompute and load_precomputed branches)
    # ------------------------------------------------------------------
    bn.set_input("water_depth",    geobn.ArraySource(depth))
    bn.set_input("vessel_traffic", geobn.ConstantSource(2.0))          # medium
    bn.set_input("current_speed",  geobn.ConstantSource(0.4))          # moderate tidal
    bn.set_input("wave_height",    geobn.ConstantSource(FRAMES[0]["wave"]))
    bn.set_input("wind_speed",     geobn.ConstantSource(FRAMES[0]["wind"]))
    bn.set_input("fog_fraction",   geobn.ConstantSource(FRAMES[0]["fog"]))

    bn.set_discretization("water_depth",    [0, 5, 20, 50, 200, 2000])
    bn.set_discretization("vessel_traffic", [0.0, 1.0, 3.0, 1000.0])
    bn.set_discretization("current_speed",  [0.0, 0.3, 1.0, 5.0])
    bn.set_discretization("wave_height",    [0.0, 0.5, 1.5, 15.0])
    bn.set_discretization("wind_speed",     [0.0, 5.0, 12.0, 50.0])
    bn.set_discretization("fog_fraction",   [0.0, 0.2, 0.6, 1.01])

    # ------------------------------------------------------------------
    # 4. Precomputed table: load if cached, otherwise compute and save
    # ------------------------------------------------------------------
    if TABLE_PATH.exists():
        print("\nLoading precomputed inference table ...")
        planner = RiskAwareAStarPlanner(
            bn, "usv_risk",
            {"medium": 0.5, "high": 1.0},
            risk_weight=500.0,
            connectivity=8,
            risk_dilation_m=200,
            risk_exponent=3,
            risk_threshold=0.4,
        )
        planner.load_precomputed(TABLE_PATH)
    else:
        print("\nPrecomputing inference table (1,215 combos) ...")
        bn.precompute(["usv_risk"])
        bn.save_precomputed(TABLE_PATH)
        print(f"  Saved → {TABLE_PATH}")
        planner = RiskAwareAStarPlanner(
            bn, "usv_risk",
            {"medium": 0.5, "high": 1.0},
            risk_weight=500.0,
            connectivity=8,
            risk_dilation_m=200,
            risk_exponent=3,
            risk_threshold=0.4,
        )

    # ------------------------------------------------------------------
    # 5. Freeze static nodes (safe after load_precomputed)
    # ------------------------------------------------------------------
    planner.freeze_static_nodes("water_depth", "vessel_traffic", "current_speed")

    # ------------------------------------------------------------------
    # 6. GIF frame loop — weather deterioration
    # ------------------------------------------------------------------
    try:
        from PIL import Image as _PilImage
    except ImportError:
        sys.exit("Pillow is required for GIF output.  Run with --extra examples.")

    gif_frames: list = []
    last_result = None

    print()
    for i, frame in enumerate(FRAMES):
        planner.update_input("wave_height",  geobn.ConstantSource(frame["wave"]))
        planner.update_input("wind_speed",   geobn.ConstantSource(frame["wind"]))
        planner.update_input("fog_fraction", geobn.ConstantSource(frame["fog"]))

        path_result = planner.find_path(START, GOAL, return_coords="latlon")
        last_result = path_result

        dist_km = path_result.total_distance_px * RESOLUTION * 111.0  # rough km
        print(
            f"  Frame {i + 1}/6  {frame['label']:<25s}"
            f"  waypoints={len(path_result.waypoints):3d}"
            f"  dist≈{dist_km:.1f} km"
            f"  cost={path_result.total_cost:.1f}"
        )

        fig = _render_frame(sea_mask, path_result.risk_grid,
                            path_result.waypoints, frame, i)
        gif_frames.append(_fig_to_pil(fig))

    # ------------------------------------------------------------------
    # 7. Assemble and save GIF
    # ------------------------------------------------------------------
    gif_path = OUT_DIR / "karmsundet_risk.gif"
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=1500,
        loop=0,
        optimize=False,
    )
    print(f"\nGIF saved → {gif_path}")

    shutil.copy2(gif_path, DOCS_ASSETS / "demo.gif")
    print(f"GIF copied → {DOCS_ASSETS / 'demo.gif'}")

    # ------------------------------------------------------------------
    # 8. Interactive HTML map (last frame)
    # ------------------------------------------------------------------
    html_path = last_result.inference_result.show_map(
        OUT_DIR,
        filename="usv_risk_map.html",
        open_browser=False,
        extra_layers={"Water depth (m)": depth},
    )
    _inject_path(html_path, last_result.waypoints)
    print(f"HTML map   → {html_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
