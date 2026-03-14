"""Pure-Python A* path planner over a 2-D risk grid."""

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

    def _heuristic(r: int, c: int) -> float:
        return math.sqrt((r - goal[0]) ** 2 + (c - goal[1]) ** 2)

    def _passable(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and not math.isnan(risk_grid[r, c])

    if not _passable(*start) or not _passable(*goal):
        return None

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(heap, (_heuristic(*start), start))
    expanded = 0

    while heap:
        if expanded >= max_nodes:
            return None
        _, current = heapq.heappop(heap)

        if current == goal:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = current
            while node is not None:
                path.append(node)
                node = came_from[node]
            path.reverse()
            return path, g_score[goal]

        expanded += 1
        r0, c0 = current

        for dr, dc in neighbors:
            nr, nc = r0 + dr, c0 + dc
            if not _passable(nr, nc):
                continue
            dist_px = _SQRT2 if (dr != 0 and dc != 0) else 1.0
            step_cost = dist_px * (1.0 + risk_weight * float(risk_grid[nr, nc]))
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get((nr, nc), math.inf):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = current
                f = tentative_g + _heuristic(nr, nc)
                heapq.heappush(heap, (f, (nr, nc)))

    return None
