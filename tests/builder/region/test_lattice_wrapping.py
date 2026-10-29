import numpy as np
import pytest
from ase import Atoms

from gdpx.structures.regions import LatticeRegion
from gdpx.structures.regions.surface_lattice import SurfaceLatticeRegion


def test_lattice_includes_negative_images_and_boundaries():
    cell = np.array([[4, 0, 0], [1, 3, 0], [0.5, 0.2, 5]])
    origin = np.array([1, -2, 3])
    region = LatticeRegion(origin, cell)
    fractional = np.array([[0, 0, 0], [1, 1, 1], [-1, -1, -1],
                           [-0.2, -0.3, -0.4], [4.8, -7.3, 20.6], [-1e-15, 0, 0]])
    atoms = Atoms('H6', positions=origin + fractional @ cell, pbc=False)
    original = atoms.positions.copy()
    assert region.get_contained_indices(atoms) == list(range(6))
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize('position', [[np.nan, 0, 0], [0, np.inf, 0], [0, 0, -np.inf]])
def test_nonfinite_positions_are_outside(position):
    assert not LatticeRegion([0, 0, 0], np.eye(3))._is_within_region(position)


def test_legacy_surface_lattice_keeps_finite_z_bounds():
    region = SurfaceLatticeRegion([0, 0, 2], np.diag([10, 10, 3]))
    assert region._is_within_region([-12, -23, 2])
    assert region._is_within_region([-12, -23, 4.9])
    assert not region._is_within_region([-12, -23, 5])
    assert not region._is_within_region([-12, -23, 1.9])
