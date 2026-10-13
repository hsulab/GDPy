"""SurfaceLatticeRegion implementation."""

from .lattice import LatticeRegion


class SurfaceLatticeRegion(LatticeRegion):
    def __init__(self, origin: list[float], cell: list[float], *args, **kwargs):
        """"""
        super().__init__(origin, cell, *args, **kwargs)

        assert self._origin[0] == 0.0 and self._origin[1] == 0.0, "The x and y of origin should be ZERO."
        assert self._cell[2][0] == 0.0 and self._cell[2][1] == 0.0, "The x and y of cell 3rd vec should be ZERO."

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        cell = data[3:]

        return SurfaceLatticeRegion(origin, cell)

    def _is_within_region(self, position) -> bool:
        """"""
        is_in = False
        # vec1, vec2 = self._cell[0], self._cell[1]
        # normal = np.cross(vec1, vec2)
        # normal = normal/np.linalg.norm(normal)
        if self._origin[2] <= position[2] < self._origin[2] + self._cell[2][2]:
            is_in = super()._is_within_region(position)

        return is_in

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "surface_lattice"
        region_params["origin"] = self._origin.tolist()
        region_params["cell"] = self._cell.tolist()

        return region_params
