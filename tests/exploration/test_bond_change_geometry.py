"""Independent candidate windows, bond classification and MC wrapper parity."""

import copy

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii

from gdpx.sampling import parse_operators
from gdpx.sampling.geometry import prepare_operators
from gdpx.structures.geometry.bond_change import (
    bond_break_settings, break_bond, form_bond,
    find_bond_break_candidates, find_bond_form_candidates,
)
from gdpx.structures.geometry.fragments import FragmentPolicy


def water():
    atoms = Atoms("OOHH", positions=[[5, 5, 3], [5, 5, 5], [6, 5, 5], [5, 6, 5]],
                  tags=[0, 1, 1, 1], cell=[20]*3, pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0, 1]))
    atoms.calc = SinglePointCalculator(atoms, energy=0., forces=np.zeros((4, 3)))
    atoms.info["bond_change"] = {"previous": True}
    return atoms


def diatomic(distance, symbols="OH", ratio=1.2):
    atoms = Atoms(symbols, positions=[[0, 0, 0], [distance, 0, 0]], tags=[1, 1], cell=[20]*3)
    atoms.set_constraint(FixAtoms(indices=[0]))
    FragmentPolicy(bond_ratio=ratio).reconcile(atoms)
    return atoms


@pytest.mark.parametrize("bond", ["O-H", "H-O", ["O", "H"], ["H", "O"]])
def test_bond_and_particle_shorthands_are_normalized(bond):
    first = bond_break_settings(particles=["O", "H", "H"], bond=bond)
    second = bond_break_settings(particles=["H", "O"], bond="O-H")
    assert first == second
    assert first["cutoffs"] == [0.0, 3.5]
    assert first["covalent_ratio"] == [0.8, 1.2]


@pytest.mark.parametrize("distance,expected", [(1.0, "break"), (2.0, "form"),
                                               (0.99, None), (2.01, None)])
def test_absolute_cutoff_window_is_inclusive(distance, expected):
    atoms = diatomic(distance)
    found = {"break": find_bond_break_candidates(atoms, particles=["H"], bond="O-H", cutoffs=[1., 2.]),
             "form": find_bond_form_candidates(atoms, particles=["H"], bond="O-H", cutoffs=[1., 2.])}
    assert {key for key, choices in found.items() if choices} == ({expected} if expected else set())


@pytest.mark.parametrize("boundary,offset,expected", [
    (0.8, -1, None), (0.8, 0, "break"), (0.8, 1, "break"),
    (1.2, -1, "break"), (1.2, 0, "break"), (1.2, 1, "form"),
])
def test_covalent_ratio_classification_and_connectivity(boundary, offset, expected):
    radius = covalent_radii[8] + covalent_radii[1]
    distance = boundary*radius
    if offset:
        distance = np.nextafter(distance, np.inf if offset > 0 else -np.inf)
    atoms = diatomic(distance)
    policy = FragmentPolicy()
    graph, _ = policy.graph(atoms)
    assert (1 in graph[0]) == (distance <= 1.2*radius)
    found = {"break": find_bond_break_candidates(atoms, particles=["H"], bond="O-H"),
             "form": find_bond_form_candidates(atoms, particles=["H"], bond="O-H")}
    assert {key for key, choices in found.items() if choices} == ({expected} if expected else set())


@pytest.mark.parametrize("symbols", ["OH", "CC"])
@pytest.mark.parametrize("upper", [1.1, 1.4])
def test_targets_follow_element_radii_and_upper_ratio(symbols, upper):
    radius = sum(covalent_radii[atomic_numbers[s]] for s in symbols)
    atoms = diatomic(radius, symbols, ratio=upper)
    event, _ = break_bond(atoms, np.random.default_rng(1), particles=[atoms[1].symbol],
                           bond=list(symbols), covalent_ratio=[0.8, upper], cutoffs=[0., upper*radius])
    assert event is not None and event["operation"] == "break"
    assert atoms.get_distance(0, 1) == pytest.approx(1.25*upper*radius)
    # Pick just the detached moving atom and exclude the bonded range.
    event, _ = form_bond(atoms, np.random.default_rng(3), particles=[atoms[1].symbol],
                           bond=list(symbols), covalent_ratio=[0.8, upper], cutoffs=[upper*radius, 4.])
    assert event is not None and event["operation"] == "form"
    assert atoms.get_distance(0, 1) == pytest.approx(radius)


def test_one_selected_endpoint_allows_an_unselected_substrate_partner():
    atoms = Atoms("OH", positions=[[0, 0, 0], [2.5, 0, 0]], tags=[0, 4], cell=[20]*3)
    event, _ = form_bond(atoms, np.random.default_rng(3), particles=["H"], bond="O-H",
                           eligible_tags=[4])
    assert event is not None and event["operation"] == "form"
    np.testing.assert_array_equal(atoms.positions[0], [0, 0, 0])
    np.testing.assert_array_equal(atoms.get_tags(), [0, 4])


def test_geometry_and_mc_wrapper_replay_the_same_break_and_form():
    direct, wrapped = water(), water()
    calc, info = direct.calc, copy.deepcopy(direct.info)
    for particles, cutoffs, operation, seed in [
        (["H"], [0., 1.2], "break", 1), (["H"], [1.2, 3.5], "form", 3),
    ]:
        recipe = dict(particles=particles, bond="O-H", cutoffs=cutoffs)
        function = break_bond if operation == "break" else form_bond
        event, diagnostic = function(direct, np.random.default_rng(seed), **recipe)
        assert event is not None and event["operation"] == operation
        op = parse_operators([dict(method=f"bond_{operation}", **recipe)])[0][0]
        op._print = op._debug = lambda *args: None
        prepare_operators([op], sorted(set(wrapped.numbers)))
        proposal = op.propose(wrapped, np.random.default_rng(seed))
        assert proposal.valid and proposal.metadata["bond_change"] == event
        assert proposal.diagnostic == diagnostic
        proposal.commit()
        np.testing.assert_array_equal(wrapped.positions, direct.positions)
        np.testing.assert_array_equal(wrapped.get_tags(), direct.get_tags())
        assert direct.calc is calc and direct.info == info
        replay = parse_operators([op.as_dict()])[0][0]
        assert replay.as_dict() == op.as_dict()
        exported = op.as_dict()
        exported["bond"][0] = "C"
        assert op.as_dict() == replay.as_dict()


@pytest.mark.parametrize("raises", [False, True])
def test_standalone_failure_restores_stale_tags_and_coordinates(monkeypatch, raises):
    import gdpx.structures.geometry.bond_change as geometry

    atoms = water()
    atoms.set_tags([0, 1, 2, 3])
    arrays = {name: values.copy() for name, values in atoms.arrays.items()}
    calc, info = atoms.calc, copy.deepcopy(atoms.info)

    def reject(*args, **kwargs):
        if raises:
            raise RuntimeError("collision checker failed")
        return False

    monkeypatch.setattr(geometry, "check_atomic_distances", reject)
    recipe = dict(particles=["H"], bond="O-H", max_random_attempts=2)
    if raises:
        with pytest.raises(RuntimeError, match="collision checker failed"):
            break_bond(atoms, np.random.default_rng(1), **recipe)
    else:
        event, diagnostic = break_bond(atoms, np.random.default_rng(1), **recipe)
        assert event is None and diagnostic == "BondChange_Failed"
    for name, values in arrays.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)
    assert atoms.calc is calc and atoms.info == info


@pytest.mark.parametrize("selection", [dict(eligible_tags=[]), dict(particles=["C"])])
def test_no_selected_particles_skips_and_restores_reconciliation(selection):
    atoms = water()
    atoms.set_tags([0, 1, 2, 3])
    tags = atoms.get_tags()
    recipe = dict(particles=["H"], bond="O-H")
    recipe.update(selection)
    event, diagnostic = break_bond(atoms, np.random.default_rng(1), **recipe)
    assert event is None and diagnostic == "BondChange_Skipped"
    np.testing.assert_array_equal(atoms.get_tags(), tags)


def test_sampling_and_geometry_use_the_same_fragment_policy():
    from gdpx.sampling.fragments import FragmentPolicy as SamplingPolicy
    assert SamplingPolicy is FragmentPolicy


@pytest.mark.parametrize("obsolete", ["preset", "pairs", "mode", "min_cutoff", "max_cutoff",
    "form_distance", "break_distance", "max_pair_distance", "break_angle", "bond_ratio",
    "bond_cutoffs", "break_fragments", "form_fragments"])
def test_obsolete_arguments_have_a_migration_error(obsolete):
    recipe = dict(particles=["H"], bond="O-H", **{obsolete: None})
    with pytest.raises(TypeError, match="Use bond_break or bond_form"):
        parse_operators([dict(method="bond_break", **recipe)])
    with pytest.raises(TypeError, match="Obsolete bond_change"):
        break_bond(water(), np.random.default_rng(1), **recipe)


@pytest.mark.parametrize("value", [None, [], [1.], [0., 1., 2.], "0,3", [False, 3.],
                                   [0., float("inf")], [float("nan"), 3.]])
def test_invalid_cutoff_shapes_and_values(value):
    with pytest.raises(ValueError, match="cutoffs"):
        bond_break_settings(particles=["H"], bond="O-H", cutoffs=value)
