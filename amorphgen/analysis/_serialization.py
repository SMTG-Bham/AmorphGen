"""NumPy conversion for analysis results and strict JSON exports."""

import numpy as np


def numpy_json_default(value):
    """Convert NumPy containers/scalars without hiding nonfinite JSON values.

    Exporters use ``allow_nan=False`` to reject invalid results. This adapter
    deliberately does not replace NaN or infinity with missing observations.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def finite_native(values):
    """Return native floats/lists, replacing undefined statistics with null."""
    values = np.asarray(values)
    if values.ndim == 0:
        value = float(values)
        return value if np.isfinite(value) else None
    return [finite_native(value) for value in values]
