"""Fractional coordinate wrapping into a half-open, one-cell interval."""

import numpy as np


def validate_wrap_interval(bounds):
    """Return finite bounds spanning exactly one fractional cell."""
    interval = np.asarray(bounds, dtype=float)
    if (interval.shape != (2,) or not np.isfinite(interval).all()
            or not np.isclose(interval[1] - interval[0], 1.0, rtol=0.0, atol=1e-12)):
        raise ValueError("Wrapping bounds must be finite and span exactly one cell.")
    return tuple(interval)


def wrap_fractional(coordinates, bounds=(0.0, 1.0)):
    """Wrap any number of cell translations; snap roundoff at the seam."""
    lower, _ = validate_wrap_interval(bounds)
    values = np.asarray(coordinates, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fractional coordinates must be finite.")
    remainder = np.remainder(values - lower, 1.0)
    seam = (remainder < 1e-12) | (1.0 - remainder < 1e-12)
    return np.where(seam, 0.0, remainder) + lower
