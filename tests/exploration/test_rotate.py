"""Rigid rotation, periodic geometry, collision filtering and proposal ownership."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gdpx.exploration.sampling import parse_operators
from gdpx.exploration.sampling.geometry import prepare_operators


def waters():
    atoms = Atoms("TiOH2OH2", positions=[(5, 5, 2), (5, 5, 4), (6, 5, 4), (5, 6, 4),
                                            (8, 5, 4), (9, 5, 4), (8, 6, 4)],
                  tags=[0, 1, 1, 1, 2, 2, 2], cell=[15]*3, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.new_array("custom", np.arange(len(atoms)))
    atoms.calc = SinglePointCalculator(atoms, energy=1.0)
    return atoms


def operator(**kwargs):
    settings = dict(method="rotate", particles=["H2O"], skip_distance_check=True)
    settings.update(kwargs)
    op = parse_operators([settings])[0][0]
    op._print = op._debug = lambda *args: None
    return op


class AngleRng:
    """Select the first molecule and prescribe an angle for geometric checks."""

    def __init__(self, angle):
        self.angle = angle

    def choice(self, count):
        return 0

    def uniform(self, low, high):
        assert low <= self.angle <= high
        return self.angle


@pytest.mark.parametrize("center", ["com", "cop", "O"])
def test_rotation_is_rigid_keeps_pivot_and_rolls_back_exactly(center, monkeypatch):
    atoms = waters()
    original = atoms.copy()
    arrays = {name: values.copy() for name, values in atoms.arrays.items()}
    calc = atoms.calc
    masses = atoms.get_masses()[1:4] if center == "com" else np.ones(3)
    pivot = original.positions[1] if center == "O" else np.average(original.positions[1:4], axis=0, weights=masses)
    monkeypatch.setattr(Atoms, "copy", lambda *args: pytest.fail("rotation must not copy the full structure"))
    op = operator(center=center, max_angle=90, axis=[0, 0, 1])
    proposal = op.propose(atoms, AngleRng(90))
    assert proposal.valid and proposal.atoms is atoms
    np.testing.assert_allclose(atoms.get_all_distances(mic=True)[1:4, 1:4],
                               original.get_all_distances(mic=True)[1:4, 1:4], atol=1e-12)
    new_pivot = atoms.positions[1] if center == "O" else np.average(atoms.positions[1:4], axis=0, weights=masses)
    np.testing.assert_allclose(new_pivot, pivot, atol=1e-12)
    np.testing.assert_array_equal(atoms.positions[[0, 4, 5, 6]], original.positions[[0, 4, 5, 6]])
    assert not np.array_equal(atoms.positions[1:4], original.positions[1:4])
    assert op.acceptance.probability(proposal.metadata, 1., 0.) == 1.
    proposal.rollback()
    for name, values in arrays.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)
    assert atoms.calc is calc
    np.testing.assert_array_equal(atoms.constraints[0].index, [0])


def test_fixed_axis_rotation_has_correct_direction_and_inverse():
    atoms = waters()
    original = atoms.positions.copy()
    op = operator(center="O", axis=[0, 0, 1], max_angle=90)
    op.propose(atoms, AngleRng(90)).commit()
    np.testing.assert_allclose(atoms.positions[2], [5, 6, 4], atol=1e-12)
    np.testing.assert_allclose(atoms.positions[3], [4, 5, 4], atol=1e-12)
    op.propose(atoms, AngleRng(-90)).commit()
    np.testing.assert_allclose(atoms.positions, original, atol=1e-12)


def test_periodic_molecule_is_rotated_using_short_bonds():
    atoms = waters()
    atoms.positions[1:4] = [[14.8, 5, 4], [0.8, 5, 4], [14.8, 6, 4]]
    distances = atoms.get_all_distances(mic=True)[1:4, 1:4]
    op = operator(center="O", axis=[0, 0, 1], max_angle=90)
    proposal = op.propose(atoms, AngleRng(90))
    assert proposal.valid
    np.testing.assert_array_equal(atoms.positions[1], [14.8, 5, 4])
    np.testing.assert_allclose(atoms.get_all_distances(mic=True)[1:4, 1:4], distances, atol=1e-12)
    np.testing.assert_allclose(atoms.get_distance(1, 2, mic=True, vector=True), [0, 1, 0], atol=1e-12)
    proposal.rollback()


def test_collision_exhaustion_restores_coordinates_and_calculator():
    atoms = waters()
    atoms.numbers[0] = 1
    atoms.positions[0] = [4, 5, 4]  # Clear initially; +90 degrees puts a water H here.
    original = atoms.positions.copy()
    calc = atoms.calc
    op = operator(center="O", axis=[0, 0, 1], max_angle=90, skip_distance_check=False,
                  allow_isolated=True, max_random_attempts=3)
    prepare_operators([op], sorted(set(atoms.numbers)))
    initial = op.propose(atoms, AngleRng(0))
    assert initial.valid
    initial.rollback()
    proposal = op.propose(atoms, AngleRng(90))
    assert not proposal.valid and proposal.closed and proposal.diagnostic == "Rotate_Failed"
    np.testing.assert_array_equal(atoms.positions, original)
    assert atoms.calc is calc


@pytest.mark.parametrize("particles,center", [(["Cu"], "com"), (["Ti"], "com"), (["H2O"], "H")])
def test_no_eligible_molecule_or_ambiguous_anchor_is_invalid(particles, center):
    atoms = waters()
    original = atoms.positions.copy()
    proposal = operator(particles=particles, center=center).propose(atoms, np.random.default_rng(8))
    assert not proposal.valid and proposal.closed
    np.testing.assert_array_equal(atoms.positions, original)


def test_rng_replay_and_operator_config_round_trip():
    op = operator(max_angle=70, center="O")
    replay = parse_operators([op.as_dict()])[0][0]
    assert replay.as_dict() == op.as_dict()
    first, second = waters(), waters()
    op.propose(first, np.random.default_rng(31)).commit()
    replay.propose(second, np.random.default_rng(31)).commit()
    np.testing.assert_array_equal(first.positions, second.positions)


@pytest.mark.parametrize("settings", [dict(max_angle=0), dict(max_angle=181), dict(max_angle=float("nan")),
                                     dict(max_angle=True), dict(max_angle="90"), dict(center="bad"), dict(axis=[0, 0, 0]),
                                     dict(axis=[1, 2]), dict(axis=[1, 0, float("inf")]), dict(particles=[])])
def test_invalid_rotation_settings(settings):
    with pytest.raises(ValueError):
        operator(**settings)
