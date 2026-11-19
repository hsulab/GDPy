"""Batched fingerprints preserve the scalar ASE distance convention."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.population.comparators import interatomic_distance as module


def scalar_fingerprint(atoms, mic=False):
    result = {}
    for number in set(atoms.numbers):
        indices = np.flatnonzero(atoms.numbers == number)
        result[number] = np.sort([
            atoms.get_distance(i, j, mic=mic)
            for offset, i in enumerate(indices) for j in indices[offset + 1:]
        ])
    return result


@pytest.mark.parametrize("cell,pbc", [
    (np.diag([8., 9., 10.]), [True, True, True]),
    ([[8., 0., 0.], [3., 7., 0.], [2., 1., 9.]], [True, True, True]),
    ([[8., 0., 0.], [3., 7., 0.], [0., 0., 0.]], [True, True, False]),
    (np.zeros((3, 3)), [False, False, False]),
])
@pytest.mark.parametrize("mic", [False, True])
def test_fingerprints_match_scalar_with_chunking(cell, pbc, mic, monkeypatch):
    monkeypatch.setattr(module, "_DISTANCE_BATCH_SIZE", 3)
    rng = np.random.default_rng(281)
    atoms = Atoms("O7H3Ti", positions=rng.uniform(-15, 25, (11, 3)),
                  cell=cell, pbc=pbc)
    original = atoms.positions.copy()
    expected = scalar_fingerprint(atoms, mic)
    actual = module.get_sorted_dist_list(atoms, mic)
    assert actual.keys() == expected.keys()
    for number in expected:
        np.testing.assert_allclose(actual[number], expected[number], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(atoms.positions, original)


@pytest.mark.parametrize("symbols", ["", "H", "HOTi"])
def test_no_same_element_pairs(symbols):
    atoms = Atoms(symbols, cell=[10]*3, pbc=True)
    actual = module.get_sorted_dist_list(atoms, mic=True)
    assert set(actual) == set(atoms.numbers)
    assert all(value.shape == (0,) for value in actual.values())


def test_periodic_boundary_and_coincident_atoms():
    atoms = Atoms("H4", positions=[[.1, 0, 0], [9.9, 0, 0], [5.1, 0, 0], [.1, 0, 0]],
                  cell=[10]*3, pbc=True)
    np.testing.assert_allclose(module.get_sorted_dist_list(atoms, True)[1],
                               scalar_fingerprint(atoms, True)[1], atol=1e-12)


@pytest.mark.parametrize("mic", [False, True])
@pytest.mark.parametrize("displacement,energy", [(0.001, 0.), (1., 0.), (0.001, 1.)])
def test_comparison_decisions_match_scalar(mic, displacement, energy, monkeypatch):
    first = Atoms("O4H2", positions=np.random.default_rng(7).uniform(0, 8, (6, 3)),
                  cell=[[9, 0, 0], [2, 8, 0], [1, 2, 10]], pbc=True)
    first.calc = SinglePointCalculator(first, energy=0.)
    second = first.copy()
    second.positions[0] += displacement
    second.calc = SinglePointCalculator(second, energy=energy)
    comparator = module.InteratomicDistanceComparator(mic=mic)
    actual = comparator.looks_like(first, second)
    first.info.clear()
    second.info.clear()
    monkeypatch.setattr(module, "get_sorted_dist_list", scalar_fingerprint)
    assert comparator.looks_like(first, second) == actual



@pytest.mark.parametrize("pbc", [
    [True, True, True], [True, True, False],
    [True, False, False], [False, False, False],
])
@pytest.mark.parametrize("batch_size", [1, 3, 16384])
def test_distances_match_explicit_periodic_images(pbc, batch_size, monkeypatch):
    from itertools import product

    monkeypatch.setattr(module, "_DISTANCE_BATCH_SIZE", batch_size)
    # Moderately skewed cell and positions in its primary parallelepiped.
    # Enumerating these neighbouring cells contains the nearest image.
    cell = np.array([[8., 0., 0.], [2., 7., 0.], [1., 2., 9.]])
    fractional = np.random.default_rng(419).uniform(0, 1, (9, 3))
    fractional[:2] = [[.01, .02, .03], [.99, .98, .97]]
    atoms = Atoms("O5H3Ti", positions=fractional @ cell, cell=cell, pbc=pbc)
    shifts = np.array(list(product(*[
        range(-2, 3) if periodic else [0] for periodic in pbc
    ]))) @ cell
    expected = {}
    for number in set(atoms.numbers):
        indices = np.flatnonzero(atoms.numbers == number)
        distances = []
        for offset, i in enumerate(indices):
            for j in indices[offset + 1:]:
                images = atoms.positions[j] - atoms.positions[i] + shifts
                distances.append(np.sqrt(np.sum(images * images, axis=1)).min())
        expected[number] = np.sort(distances)
    actual = module.get_sorted_dist_list(atoms, mic=True)
    assert actual.keys() == expected.keys()
    for number in expected:
        np.testing.assert_allclose(actual[number], expected[number], rtol=1e-12, atol=1e-12)


def test_known_boundary_distances_respect_partial_periodicity():
    atoms = Atoms("H2", positions=[[.1, .1, 0], [9.9, 9.9, 0]],
                  cell=[10]*3, pbc=[True, False, False])
    expected = np.sqrt(.2**2 + 9.8**2)
    np.testing.assert_allclose(module.get_sorted_dist_list(atoms, True)[1], [expected])
    atoms.pbc = True
    np.testing.assert_allclose(module.get_sorted_dist_list(atoms, True)[1], [np.sqrt(2)*.2])
    np.testing.assert_allclose(module.get_sorted_dist_list(atoms, False)[1], [np.sqrt(2)*9.8])


@pytest.mark.parametrize("pbc", [[True, True, True], [True, True, False]])
def test_fingerprint_invariant_to_lattice_shifts_translation_and_atom_order(pbc):
    rng = np.random.default_rng(920)
    atoms = Atoms("O5H3Ti", scaled_positions=rng.uniform(0, 1, (9, 3)),
                  cell=[[8, 0, 0], [2, 7, 0], [1, 2, 9]], pbc=pbc)
    expected = module.get_sorted_dist_list(atoms, True)
    shifted = atoms.copy()
    lattice_shifts = rng.integers(-5, 6, (len(atoms), 3)) * np.array(pbc)
    shifted.positions += lattice_shifts @ atoms.cell.array
    shifted.positions += [17.3, -11.7, 6.2]
    shifted = shifted[rng.permutation(len(shifted))]
    actual = module.get_sorted_dist_list(shifted, True)
    for number in expected:
        np.testing.assert_allclose(actual[number], expected[number], rtol=1e-12, atol=1e-12)
