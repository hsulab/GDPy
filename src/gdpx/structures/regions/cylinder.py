"""CylinderRegion implementation."""

import numpy as np

from .base import BaseRegion


class CylinderRegion(BaseRegion):
    """Region by a vertical cylinder."""

    def __init__(
        self,
        origin: list[float],
        radius: float,
        height: float,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(origin=origin, *args, **kwargs)

        self._radius = radius
        self._height = height

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        radius = data[3]
        height = data[4]

        return CylinderRegion(origin, radius, height)

    def _get_a_random_position(self, rng):
        """"""
        r, theta, h = rng.uniform(0, 1, 3)  # r, theta, h

        ran_pos = np.array([r * np.cos(theta), r * np.sin(theta), h])
        ran_pos += self._origin

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """"""
        ox, oy, oz = self._origin

        is_in = False
        if oz <= position[2] <= oz + self._height:
            distance = np.linalg.norm(position[:2] - self._origin[:2])
            if distance <= self._radius:
                is_in = True

        return is_in

    def get_volume(self) -> float:
        """"""
        return np.pi * self._radius**2 * self._height

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"radius {self._radius} "
        content += f"height   {self._height} "

        return content

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "cylinder"
        region_params["origin"] = self._origin.tolist()
        region_params["radius"] = self._radius
        region_params["height"] = self._height

        return region_params
