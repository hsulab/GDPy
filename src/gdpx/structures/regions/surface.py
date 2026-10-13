"""SurfaceRegion implementation."""

import numpy as np

from .base import BaseRegion


class SurfaceRegion(BaseRegion):
    def __init__(
        self,
        origin: list[float],
        normal: list[float],
        thickness: float,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._normal = np.array(normal, dtype=float)
        self._thickness = thickness

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        normal = data[3:6]
        thickness = data[6:9]

        return SurfaceRegion(origin, normal, thickness)

    def _get_a_random_position(self, rng):
        """"""
        ran_frac_coord = rng.uniform(0, 1, 3)
        ran_pos = np.dot(ran_frac_coord, self._cell)
        ran_pos += self._origin

        raise NotImplementedError()

    def _is_within_region(self, position) -> bool:
        """"""
        is_in = False
        pos_ = position - self._origin
        frac_pos_ = np.dot(np.linalg.inv(self._cell.T), pos_)
        if (
            0.0 <= np.modf(frac_pos_[0])[0] < 1.0
            and 0.0 <= np.modf(frac_pos_[1])[0] < 1.0
            and 0.0 <= np.modf(frac_pos_[2])[0] < 1.0
        ):
            is_in = True

        raise NotImplementedError()

    def get_volume(self) -> float:
        """"""
        a, b, c = self._cell

        # return np.dot(np.cross(a,b), c)
        raise NotImplementedError()

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__}\n"
        content += f"origin\n"
        content += ("  " + "{:<12.8f}  " * 3 + "\n").format(*self._origin)
        # content += f"cell\n"
        # content += (("  "+"{:<12.8f}  "*3+"\n")*3).format(*self._cell.flatten())

        return content
