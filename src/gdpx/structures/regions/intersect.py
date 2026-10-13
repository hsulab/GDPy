"""IntersectRegion implementation."""

import copy

import numpy as np

from .base import BaseRegion


class IntersectRegion(BaseRegion):
    #: Maximum number of attempts to get a random position.
    MAX_ATTEMPTS: int = 1000

    def __init__(self, regions, origin=np.zeros(0), *args, **kwargs):
        """Initialise an IntersectRegion.

        Args:
            regions: A list of regions.
            origin: Region origin should be always zero.

        """
        super().__init__(origin, *args, **kwargs)

        # NOTE: Can sub-regions be intersect ones?
        self.regions = regions
        assert len(self.regions), "IntersectRegion supports only twp sub-regions."

        self._regions = []
        for r in copy.deepcopy(regions):
            from gdpx.structures.regions.registry import REGION_REGISTRY

            method = r.pop("method", "auto")
            class_name = "".join(part.capitalize() for part in method.split("_")) + "Region"
            curr_region = REGION_REGISTRY[class_name](**r)
            self._regions.append(curr_region)

        return

    @staticmethod
    def from_str(command: str):
        raise NotImplementedError()

    def _get_a_random_position(self, rng):
        """"""
        for i in range(self.MAX_ATTEMPTS):
            r1, r2 = self._regions
            ran_pos = r1._get_a_random_position(rng)
            if not r2._is_within_region(ran_pos):
                break
        else:
            raise RuntimeError("Fail to get a random position in the IntersectRegion.")

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """"""

        raise NotImplementedError()

    def get_volume(self) -> float:
        """"""

        raise NotImplementedError()

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "intersect"
        region_params["origin"] = self._origin.tolist()
        region_params["regions"] = self.regions

        return region_params
