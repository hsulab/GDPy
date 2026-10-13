"""LatticeRegion implementation."""

import numpy as np

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
        """Check if a position is in the region.

        Some atoms may have negative coordinates, thus,
        users need set origin with a small negative value (e.g. 1e-7)
        to make atoms at boundary included in the region

        Args:
            position: An array that has a shape of (3,).

        Returns:
            A boolen.

        """
        is_in = False
        pos_ = position - self._origin
        frac_pos_ = np.dot(np.linalg.inv(self._cell.T), pos_)
        if (
            0.0 <= np.modf(frac_pos_[0])[0] < 1.0
            and 0.0 <= np.modf(frac_pos_[1])[0] < 1.0
            and 0.0 <= np.modf(frac_pos_[2])[0] < 1.0
        ):
            is_in = True

        return is_in

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
