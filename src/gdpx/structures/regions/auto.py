"""AutoRegion implementation."""

from typing import Optional

import numpy as np
from ase import Atoms

from .base import BaseRegion


class AutoRegion(BaseRegion):
    """AutoRegion is a region that is automatically determined by the atoms."""

    def __init__(
        self,
        origin: list[float] = [0.0, 0.0, 0.0],
        atoms: Optional[Atoms] = None,
    ) -> None:
        """"""
        super().__init__(origin)

        self._atoms = atoms

        return

    @staticmethod
    def from_str(command: str):
        """"""

        raise NotImplementedError("AutoRegion does support init_from_str.")

    def _get_a_random_position(self, rng=np.random):
        """"""
        if self._atoms is None:
            raise RuntimeError(f"No atoms is attached to {self.__class__.__name__}.")

        frac_pos = rng.uniform(0, 1, 3)
        ran_pos = np.dot(frac_pos, self._atoms.get_cell())

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """"""
        if self._atoms is None:
            raise RuntimeError(f"No atoms is attached to {self.__class__.__name__}")

        is_in = False
        pos_ = position - self._origin
        # Make sure the fractional coordinates are positive by adding a large number,
        # otherwise, some may be negative and make the modf() function fail.
        frac_pos_ = np.dot(np.linalg.inv(self._atoms.get_cell().T), pos_) + 1e8
        if (
            0.0 <= np.modf(frac_pos_[0])[0] < 1.0
            and 0.0 <= np.modf(frac_pos_[1])[0] < 1.0
            and 0.0 <= np.modf(frac_pos_[2])[0] < 1.0
        ):
            is_in = True

        return is_in

    def get_volume(self) -> float:
        """"""
        if self._atoms is None:
            raise RuntimeError(f"No atoms is attached to {self.__class__.__name__}")

        return self._atoms.get_volume()

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "

        return content

    def as_dict(self) -> dict:
        """"""
        region_params = {}
        region_params["method"] = "auto"

        return region_params
