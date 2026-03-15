"""RiskAwareAStarPlanner — risk-aware A* path planner backed by a geobn Bayesian network."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ._astar import astar
from ._coords import (
    _convert_path,
    _path_length_px,
    latlon_to_pixel,
)
from ._risk import extract_risk_grid


@dataclass
class PathResult:
    """Result of a :meth:`RiskAwareAStarPlanner.find_path` call.

    Attributes
    ----------
    waypoints:
        Path waypoints in the coordinate system requested via *return_coords*.
    total_cost:
        Cumulative A* cost (distance-weighted by risk).
    total_distance_px:
        Path length in pixels (sum of Euclidean step distances).
    risk_grid:
        2-D risk probability array used for planning (shape = grid shape).
    inference_result:
        The raw ``geobn.InferenceResult`` from the Bayesian network.
    """

    waypoints: list[tuple[float, float]]
    total_cost: float
    total_distance_px: float
    risk_grid: np.ndarray
    inference_result: object


class RiskAwareAStarPlanner:
    """Risk-aware A* path planner.

    Parameters
    ----------
    bn:
        A ``geobn.BayesianNetwork`` instance (or compatible object).
    risk_node:
        Name of the network node whose marginal represents risk.
    risk_state:
        Either a state name (``str``) or a ``dict[str, float]`` of weighted states.
    risk_weight:
        Trade-off between path distance and risk (default 1.0).
        Higher values produce more risk-averse routes.
    connectivity:
        ``4`` (cardinal) or ``8`` (cardinal + diagonal, default).

    After construction, call ``bn.precompute([risk_node])`` offline, then
    :meth:`load_precomputed` at runtime before :meth:`find_path`.
    """

    def __init__(
        self,
        bn,
        risk_node: str,
        risk_state: str | dict[str, float],
        risk_weight: float = 1.0,
        connectivity: int = 8,
    ) -> None:
        self._bn = bn
        self._risk_node = risk_node
        self._risk_state = risk_state
        self._risk_weight = risk_weight
        self._connectivity = connectivity
        self._precomputed = bool(getattr(bn, '_inference_table', {}))

    def load_precomputed(self, path: str | Path) -> None:
        """Restore a precomputed inference table from *path* (.npz).

        Use instead of :meth:`precompute` at runtime.  After a successful load,
        :meth:`find_path` can be called immediately — no pgmpy calls are made.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file was saved with a different BN configuration.
        """
        self._bn.load_precomputed(path)
        self._precomputed = True

    # ------------------------------------------------------------------
    def find_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        return_coords: str = "latlon",
    ) -> PathResult:
        """Plan a risk-aware path from *start* to *goal*.

        Parameters
        ----------
        start, goal:
            ``(lat, lon)`` coordinates in WGS84.
        return_coords:
            Output coordinate system for :attr:`PathResult.waypoints`:
            ``"latlon"`` (default), ``"crs"``, or ``"pixel"``.

        Returns
        -------
        PathResult

        Raises
        ------
        RuntimeError
            If :meth:`load_precomputed` (or ``bn.precompute()`` before construction) has not been called.
        ValueError
            If *start* or *goal* is outside the grid bounds.
        RuntimeError
            If no path exists between *start* and *goal*.
        """
        if not self._precomputed:
            raise RuntimeError(
                "call bn.precompute([risk_node]) + bn.save_precomputed(), "
                "then load_precomputed() before find_path()."
            )

        # --- Bayesian network inference ---------------------------------
        infer_result = self._bn.infer(query=[self._risk_node])
        risk_grid = extract_risk_grid(infer_result, self._risk_node, self._risk_state)

        # --- Coordinate conversion --------------------------------------
        transform = infer_result.transform
        crs = infer_result.crs

        start_px = latlon_to_pixel(start[0], start[1], transform, crs)
        goal_px = latlon_to_pixel(goal[0], goal[1], transform, crs)

        rows, cols = risk_grid.shape
        for name, px in [("start", start_px), ("goal", goal_px)]:
            r, c = px
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(
                    f"{name} pixel {px} is outside grid bounds ({rows}×{cols})."
                )

        # --- A* search --------------------------------------------------
        astar_result = astar(
            risk_grid,
            start_px,
            goal_px,
            self._risk_weight,
            self._connectivity,
        )
        if astar_result is None:
            raise RuntimeError(
                f"No path found between {start} and {goal}."
            )
        pixel_path, total_cost = astar_result

        # --- Build result -----------------------------------------------
        waypoints = _convert_path(pixel_path, transform, crs, return_coords)
        distance_px = _path_length_px(pixel_path)

        return PathResult(
            waypoints=waypoints,
            total_cost=total_cost,
            total_distance_px=distance_px,
            risk_grid=risk_grid,
            inference_result=infer_result,
        )
