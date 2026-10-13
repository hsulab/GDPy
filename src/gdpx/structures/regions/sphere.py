"""SphereRegion implementation."""

import numpy as np

from .base import BaseRegion


class SphereRegion(BaseRegion):
    def __init__(self, origin: list[float], radius: float, *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._radius = radius

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        radius = data[3]

        return SphereRegion(origin, radius)

    def _get_a_random_position(self, rng):
        """"""
        ran_coord = rng.uniform(0, 1, 3)
        polar = np.array([self._radius, np.pi, 2 * np.pi]) * ran_coord
        r, theta, phi = polar

        ran_pos = np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        )
        ran_pos += self._origin

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """"""
        is_in = False

        position = np.array(position)
        distance = np.linalg.norm(position - self._origin)
        if distance <= self._radius:
            is_in = True

        return is_in

    def get_volume(self):
        """"""

        return 4.0 / 3.0 * np.pi * self._radius**3

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"radius {self._radius} "
        content += f"volume {self.get_volume()} "

        return content

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "sphere"
        region_params["origin"] = self._origin.tolist()
        region_params["radius"] = self._radius

        return region_params
