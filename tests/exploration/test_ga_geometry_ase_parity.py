"""Parity checks for GDPy's replacements of ASE-GA geometry helpers."""

import numpy as np
import pytest
from ase import Atoms

from gdpx.structures.geometry.ga import (
    CellBounds,
    atoms_too_close,
    atoms_too_close_two_sets,
    closest_distances_generator,
    get_cell_angles_lengths,
    get_nnmat,
)

ase_utilities = pytest.importorskip("ase.ga.utilities")


def test_distance_helpers_match_ase_ga():
    numbers = [28, 29]
    gdpx_distances = closest_distances_generator(numbers, 0.7)
    ase_distances = ase_utilities.closest_distances_generator(numbers, 0.7)
    assert gdpx_distances == ase_distances

    atoms = Atoms(
        "Cu2Ni2",
        positions=[[0, 0, 0], [1.0, 0, 0], [4, 0, 0], [6.5, 0, 0]],
        cell=[10, 10, 10],
        pbc=True,
        tags=[1, 1, 2, 3],
    )
    for use_tags in (False, True):
        assert atoms_too_close(atoms, gdpx_distances, use_tags) == ase_utilities.atoms_too_close(
            atoms, ase_distances, use_tags
        )
    assert atoms_too_close_two_sets(atoms[:2], atoms[2:], gdpx_distances) == (
        ase_utilities.atoms_too_close_two_sets(atoms[:2], atoms[2:], ase_distances)
    )


def test_cell_and_nnmat_helpers_match_ase_ga():
    cell = np.array([[5.0, 0.0, 0.0], [0.5, 6.0, 0.0], [0.2, 0.3, 7.0]])
    gdpx_parameters = get_cell_angles_lengths(cell)
    ase_parameters = ase_utilities.get_cell_angles_lengths(cell)
    for name in gdpx_parameters:
        np.testing.assert_allclose(gdpx_parameters[name], ase_parameters[name], atol=1e-15, rtol=1e-15)
    bounds = {"a": [4, 8], "phi": [30, 150]}
    assert CellBounds(bounds).is_within_bounds(cell) == ase_utilities.CellBounds(bounds).is_within_bounds(cell)

    atoms = Atoms(
        "Cu3Ni",
        positions=[[2, 2, 2], [4.4, 2, 2], [2, 4.4, 2], [4.4, 4.4, 2]],
        cell=[20, 20, 20],
        pbc=False,
    )
    np.testing.assert_allclose(get_nnmat(atoms), ase_utilities.get_nnmat(atoms), atol=0.0, rtol=0.0)
