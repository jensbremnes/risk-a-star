"""Risk grid extraction from geobn inference results."""

from __future__ import annotations

import numpy as np


def extract_risk_grid(result, risk_node: str, risk_state: str | dict[str, float]) -> np.ndarray:
    """Extract a 2-D risk probability grid from a geobn InferenceResult.

    Parameters
    ----------
    result:
        A ``geobn.InferenceResult`` (or compatible object) with attributes
        ``state_names`` and ``probabilities``.
    risk_node:
        Name of the node in the Bayesian network whose marginal to use.
    risk_state:
        Either:
        - ``str``  — name of the single state whose probability is the risk,
        - ``dict[str, float]`` — weighted combination of named states.

    Returns
    -------
    np.ndarray
        2-D float array with values in ``[0, 1]``; NaN where inference
        produced NaN (e.g. masked / outside domain).
    """
    probs = result.probabilities[risk_node]          # shape (..., n_states)
    state_names: list[str] = result.state_names[risk_node]

    if isinstance(risk_state, str):
        idx = state_names.index(risk_state)
        grid = probs[..., idx].astype(float)
    elif isinstance(risk_state, dict):
        grid = np.zeros(probs.shape[:-1], dtype=float)
        for name, weight in risk_state.items():
            idx = state_names.index(name)
            grid += weight * probs[..., idx]
    else:
        raise TypeError(f"risk_state must be str or dict, got {type(risk_state)}")

    # Preserve NaN; clip valid values to [0, 1]
    nan_mask = np.isnan(grid)
    np.clip(grid, 0.0, 1.0, out=grid, where=~nan_mask)
    return grid
