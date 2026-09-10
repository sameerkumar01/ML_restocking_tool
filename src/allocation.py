import numpy as np


def allocate_units(weights, total_units):
    values = np.asarray(weights, dtype=float)
    if total_units < 0:
        raise ValueError("total_units must be non-negative")
    if len(values) == 0:
        return np.array([], dtype=int)
    values = np.clip(values, 0, None)
    values = values / values.sum() if values.sum() else np.full(len(values), 1 / len(values))
    exact = values * int(total_units)
    result = np.floor(exact).astype(int)
    remaining = int(total_units) - int(result.sum())
    if remaining:
        order = np.argsort(-(exact - result))
        result[order[:remaining]] += 1
    return result
