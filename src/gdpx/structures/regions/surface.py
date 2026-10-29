"""A slab spanning the periodic xy plane with Cartesian height bounds."""

import math

import numpy as np

from .base import BaseRegion


def _wrap_interval(wrap):
    """Return a one-cell fractional interval, or None for unwrapped positions."""
    if wrap is False:
        return None
    if wrap is True:
        return (-0.1, 0.9)
    if (not isinstance(wrap, (list, tuple)) or len(wrap) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in wrap)):
        raise ValueError("Surface region wrap must be a boolean or [lower, upper] fractional bounds.")
    lower, upper = wrap
    if (not all(math.isfinite(x) for x in wrap)
            or not math.isclose(upper - lower, 1.0, rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("Surface region wrap bounds must be finite and span exactly one cell.")
    return (float(lower), float(upper))


class SurfaceRegion(BaseRegion):
    """Cover all xy images and the inclusive, nonperiodic interval z_range.

    The cell is needed only for sampling and volume. If omitted, preprocess
    obtains it from the current structure. Membership wraps the structure into
    a configurable fractional interval before checking Cartesian height.
    """

    def __init__(self, z_range, cell=None, wrap=True):
        super().__init__(origin=[0.0, 0.0, 0.0])
        self._wrap = _wrap_interval(wrap)
        self.z_range = np.asarray(z_range, dtype=float)
        if (self.z_range.shape != (2,) or not np.isfinite(self.z_range).all()
                or self.z_range[0] >= self.z_range[1]):
            raise ValueError("Surface z_range must contain two finite increasing heights.")
        self._configured_cell = None if cell is None else self._validate_cell(cell)
        self._cell = self._configured_cell

    @staticmethod
    def _validate_cell(cell):
        cell = np.asarray(cell, dtype=float)
        if cell.shape != (3, 3) or not np.isfinite(cell).all():
            raise ValueError("Surface cell must be a finite 3 by 3 matrix.")
        if not np.allclose(cell[:2, 2], 0.0, atol=1e-12):
            raise ValueError("Surface cell vectors a and b must lie in the xy plane.")
        if abs(np.linalg.det(cell[:2, :2])) < 1e-12:
            raise ValueError("Surface cell must have nonzero xy area.")
        return cell.copy()

    def preprocess(self, atoms):
        if self._configured_cell is None:
            self._cell = np.asarray(atoms.cell).copy()

    def _require_cell(self):
        if self._cell is None:
            raise RuntimeError("Surface sampling and volume require a cell or preprocess(atoms).")
        return self._validate_cell(self._cell)

    @staticmethod
    def from_str(command):
        fields = command.split()
        if len(fields) not in (3, 5):
            raise ValueError("Surface region syntax: surface zmin zmax [wrap_lower wrap_upper].")
        wrap = True if len(fields) == 3 else [float(x) for x in fields[3:]]
        return SurfaceRegion([float(x) for x in fields[1:3]], wrap=wrap)

    def _get_positions_for_tags(self, atoms):
        """Use the same coordinate convention for atoms and tagged particles."""
        if self._wrap is None:
            return atoms.get_positions()
        # Avoid sending the exact lower boundary to the upper image through
        # roundoff in ASE's center-to-shift conversion (e.g. -0.1).
        return atoms.get_positions(wrap=True, center=sum(self._wrap) / 2.0, eps=1e-12)

    def get_contained_indices(self, atoms):
        positions = self._get_positions_for_tags(atoms)
        return [i for i, position in enumerate(positions) if self._is_within_region(position)]

    def _get_a_random_position(self, rng):
        cell = self._require_cell()
        uv = rng.uniform(size=2)
        position = uv @ cell[:2]
        position[2] = rng.uniform(*self.z_range)
        return position

    def _is_within_region(self, position):
        return bool(np.isfinite(position).all() and self.z_range[0] <= position[2] <= self.z_range[1])

    def get_volume(self):
        cell = self._require_cell()
        return abs(np.linalg.det(cell[:2, :2])) * (self.z_range[1] - self.z_range[0])

    def as_dict(self):
        result = dict(method="surface", z_range=self.z_range.tolist(),
                      wrap=False if self._wrap is None else list(self._wrap))
        if self._configured_cell is not None:
            result["cell"] = self._configured_cell.tolist()
        return result

    def __repr__(self):
        return f"SurfaceRegion(z_range={self.z_range.tolist()})"
