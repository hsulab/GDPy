"""CubeRegion implementation."""

import copy

import numpy as np

from .base import BaseRegion


class CubeRegion(BaseRegion):
    def __init__(self, origin: list[float], boundary: list[float], *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        boundaries_ = np.array(boundary, dtype=np.float64)
        assert len(boundaries_) == 6, "Cubic region needs 6 numbers to define."
        self.boundaries = boundaries_

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        boundary = data[3:]

        return CubeRegion(origin, boundary)

    def _get_a_random_position(self, rng=np.random):
        """"""
        boundaries_ = copy.deepcopy(self.boundaries)
        boundaries_ = np.reshape(boundaries_, (2, 3))

        ran_frac_pos = rng.uniform(0, 1, 3)
        ran_pos = self._origin + boundaries_[0, :] + (boundaries_[0, :] - boundaries_[1, :]) * ran_frac_pos

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """"""
        ox, oy, oz = self._origin
        (xl, yl, zl, xh, yh, zh) = self.boundaries

        position = np.array(position)

        is_in = False
        if (
            (ox + xl <= position[0] <= ox + xh)
            and (oy + yl <= position[1] <= oy + yh)
            and (oz + zl <= position[2] <= oz + zh)
        ):
            is_in = True

        return is_in

    def get_volume(self) -> float:
        """"""
        (xl, yl, zl, xh, yh, zh) = self.boundaries

        return (xh - xl) * (yh - yl) * (zh - zl)

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"origin {self.boundaries[:3]} "
        content += f"lmin {self.boundaries[3:6]} "
        content += f"lmax {self.boundaries[6:9]} "

        return content

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "cube"
        region_params["origin"] = self._origin.tolist()
        region_params["boundary"] = self.boundaries.tolist()

        return region_params
