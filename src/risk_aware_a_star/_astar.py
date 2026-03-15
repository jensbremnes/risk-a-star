"""Numba-accelerated A* path planner over a 2-D risk grid."""

from __future__ import annotations

import math

import numba
import numpy as np

_SQRT2 = math.sqrt(2)

_OFFSETS_4 = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=np.int32)
_OFFSETS_8 = np.array(
    [[0, 1], [0, -1], [1, 0], [-1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]],
    dtype=np.int32,
)


@numba.njit(cache=True)  # pragma: no cover
def _heap_push(hf, hn, size, f, node):
    """Push (f, node) onto the min-heap and return the new size."""
    hf[size] = f
    hn[size] = node
    size += 1
    i = size - 1
    while i > 0:
        parent = (i - 1) // 2
        if hf[parent] > hf[i]:
            tmp_f = hf[parent]
            hf[parent] = hf[i]
            hf[i] = tmp_f
            tmp_n = hn[parent]
            hn[parent] = hn[i]
            hn[i] = tmp_n
            i = parent
        else:
            break
    return size


@numba.njit(cache=True)  # pragma: no cover
def _heap_pop(hf, hn, size):
    """Pop minimum (f, node) and return (f, node, new_size)."""
    f = hf[0]
    node = hn[0]
    size -= 1
    hf[0] = hf[size]
    hn[0] = hn[size]
    i = 0
    while True:
        left = 2 * i + 1
        right = 2 * i + 2
        smallest = i
        if left < size and hf[left] < hf[smallest]:
            smallest = left
        if right < size and hf[right] < hf[smallest]:
            smallest = right
        if smallest != i:
            tmp_f = hf[smallest]
            hf[smallest] = hf[i]
            hf[i] = tmp_f
            tmp_n = hn[smallest]
            hn[smallest] = hn[i]
            hn[i] = tmp_n
            i = smallest
        else:
            break
    return f, node, size


@numba.njit(cache=True)  # pragma: no cover
def _astar_core(risk_grid, passable, sr, sc, gr, gc,
                risk_weight, offsets, max_nodes):
    """Core A* loop compiled to native code.

    Returns (path_buf, path_len, cost).  path_len == 0 means no path found.
    path_buf[0..path_len-1] holds flat indices from goal back to start.
    """
    rows = risk_grid.shape[0]
    cols = risk_grid.shape[1]
    n = rows * cols

    g_score = np.full(n, np.inf)
    came_from = np.full(n, -1, dtype=np.int32)
    closed = np.zeros(n, dtype=np.bool_)

    # Heap capacity: each node can be pushed at most 8 times (one per incoming
    # edge in 8-connectivity), so n * 8 is a tight upper bound.
    heap_cap = n * 8
    hf = np.empty(heap_cap, dtype=np.float64)
    hn = np.empty(heap_cap, dtype=np.int32)
    heap_size = 0

    path_buf = np.empty(n, dtype=np.int32)

    start_flat = sr * cols + sc
    goal_flat = gr * cols + gc

    g_score[start_flat] = 0.0
    came_from[start_flat] = -2  # sentinel: start has no predecessor

    h_start = math.sqrt(float((sr - gr) ** 2 + (sc - gc) ** 2))
    heap_size = _heap_push(hf, hn, heap_size, h_start, start_flat)

    expanded = 0
    n_offsets = offsets.shape[0]

    while heap_size > 0:
        if expanded >= max_nodes:
            return path_buf, 0, 0.0

        f_cur, flat, heap_size = _heap_pop(hf, hn, heap_size)
        r0 = flat // cols
        c0 = flat % cols

        if closed[flat]:
            continue
        closed[flat] = True

        if flat == goal_flat:
            path_len = 0
            cur = flat
            while cur != -2:
                path_buf[path_len] = cur
                path_len += 1
                cur = came_from[cur]
            return path_buf, path_len, g_score[goal_flat]

        expanded += 1

        for k in range(n_offsets):
            dr = offsets[k, 0]
            dc = offsets[k, 1]
            nr = r0 + dr
            nc = c0 + dc

            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            nflat = nr * cols + nc
            if not passable[nr, nc] or closed[nflat]:
                continue

            dist_px = 1.4142135623730951 if (dr != 0 and dc != 0) else 1.0
            step_cost = dist_px * (1.0 + risk_weight * risk_grid[nr, nc])
            tentative_g = g_score[flat] + step_cost

            if tentative_g < g_score[nflat]:
                g_score[nflat] = tentative_g
                came_from[nflat] = flat
                h = math.sqrt(float((nr - gr) ** 2 + (nc - gc) ** 2))
                heap_size = _heap_push(hf, hn, heap_size, tentative_g + h, nflat)

    return path_buf, 0, 0.0


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
    sr, sc = start
    gr, gc = goal

    passable = ~np.isnan(risk_grid)

    if not passable[sr, sc] or not passable[gr, gc]:
        return None

    if start == goal:
        return [start], 0.0

    offsets = _OFFSETS_4 if connectivity == 4 else _OFFSETS_8
    path_buf, path_len, cost = _astar_core(
        risk_grid, passable, sr, sc, gr, gc,
        float(risk_weight), offsets, int(max_nodes),
    )

    if path_len == 0:
        return None

    path = [
        (int(path_buf[i] // cols), int(path_buf[i] % cols))
        for i in range(path_len - 1, -1, -1)
    ]
    return path, float(cost)
