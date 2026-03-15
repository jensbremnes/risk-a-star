"""Numpy-based A* path planner over a 2-D risk grid."""

from __future__ import annotations

import heapq
import math

import numpy as np

_CARDINAL = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
_SQRT2 = math.sqrt(2)


def astar(
    risk_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    risk_weight: float = 1.0,
    connectivity: int = 8,
    max_nodes: int = 500 * 500,
) -> tuple[list[tuple[int, int]], float] | None:
    """Find the minimum-cost path through *risk_grid* from *start* to *goal*.

    Parameters
    ----------
    risk_grid:
        2-D float array of risk values in ``[0, 1]``. ``NaN`` cells are
        treated as impassable.
    start, goal:
        ``(row, col)`` pixel coordinates.
    risk_weight:
        Controls the trade-off between distance and risk.
        Cost of a step = ``dist_px * (1 + risk_weight * risk_grid[r, c])``.
    connectivity:
        ``4`` (cardinal only) or ``8`` (cardinal + diagonal).
    max_nodes:
        Safety limit on the number of nodes expanded.

    Returns
    -------
    ``(pixel_path, total_cost)`` or ``None`` if no path exists.
    """
    rows, cols = risk_grid.shape
    neighbors = _CARDINAL if connectivity == 4 else _CARDINAL + _DIAGONAL

    gr, gc = goal
    sr, sc = start

    def _passable(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and not math.isnan(risk_grid[r, c])

    if not _passable(sr, sc) or not _passable(gr, gc):
        return None

    # Early return when start == goal
    if start == goal:
        return [start], 0.0

    # Pre-allocate numpy arrays — flat-index encoded
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    came_from = np.full((rows, cols), -1, dtype=np.int32)
    closed = np.zeros((rows, cols), dtype=bool)

    g_score[sr, sc] = 0.0
    # Sentinel: start has no predecessor
    came_from[sr, sc] = -2

    goal_flat = gr * cols + gc
    h_start = math.sqrt((sr - gr) ** 2 + (sc - gc) ** 2)
    heap: list[tuple[float, int]] = []
    heapq.heappush(heap, (h_start, sr * cols + sc))
    expanded = 0

    while heap:
        if expanded >= max_nodes:
            return None
        _, flat = heapq.heappop(heap)
        r0, c0 = flat // cols, flat % cols

        if closed[r0, c0]:
            continue
        closed[r0, c0] = True

        if flat == goal_flat:
            # Reconstruct path by walking came_from back to sentinel -2
            path: list[tuple[int, int]] = []
            cur = flat
            while cur != -2:
                r, c = cur // cols, cur % cols
                path.append((r, c))
                cur = int(came_from[r, c])
            path.reverse()
            return path, float(g_score[gr, gc])

        expanded += 1

        for dr, dc in neighbors:
            nr, nc = r0 + dr, c0 + dc
            if not _passable(nr, nc) or closed[nr, nc]:
                continue
            dist_px = _SQRT2 if (dr != 0 and dc != 0) else 1.0
            step_cost = dist_px * (1.0 + risk_weight * float(risk_grid[nr, nc]))
            tentative_g = g_score[r0, c0] + step_cost

            if tentative_g < g_score[nr, nc]:
                g_score[nr, nc] = tentative_g
                came_from[nr, nc] = flat
                f = tentative_g + math.sqrt((nr - gr) ** 2 + (nc - gc) ** 2)
                heapq.heappush(heap, (f, nr * cols + nc))

    return None
