"""LatticeRegion implementation."""

import numpy as np

from gdpx.structures.geometry.wrapping import wrap_fractional

from .base import BaseRegion


class LatticeRegion(BaseRegion):
    def __init__(self, origin: list[float], cell: list[float], *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._cell = np.reshape(cell, (3, 3))

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        cell = data[3:]

        return LatticeRegion(origin, cell)

    def _get_a_random_position(self, rng):
        """"""
        ran_frac_coord = rng.uniform(0, 1, 3)
        ran_pos = np.dot(ran_frac_coord, self._cell)
        ran_pos += self._origin

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """Include every finite periodic image in the region's lattice cell.

        The region's origin and cell define fractional coordinates; structure
        PBC flags do not restrict this explicitly periodic region.
        """
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            return False
        fractional = np.linalg.solve(self._cell.T, position - self._origin)
        wrapped = wrap_fractional(fractional)
        return bool(np.all((0.0 <= wrapped) & (wrapped < 1.0)))

    def get_volume(self) -> float:
        """"""
        a, b, c = self._cell

        return np.dot(np.cross(a, b), c)

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__}\n"
        content += f"origin\n"
        content += ("  " + "{:<12.8f}  " * 3 + "\n").format(*self._origin)
        content += f"cell\n"
        content += (("  " + "{:<12.8f}  " * 3 + "\n") * 3).format(*self._cell.flatten())

        return content

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "lattice"
        region_params["origin"] = self._origin.tolist()
        region_params["cell"] = self._cell.tolist()

        return region_params
