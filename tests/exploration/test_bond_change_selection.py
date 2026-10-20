"""Explicit fragment motion and stationary acceptor selection."""

import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from gdpx.sampling import parse_operators
from gdpx.sampling.geometry import prepare_operators
from gdpx.structures.geometry.bond_change import break_bond, form_bond


def water():
    return Atoms("OHH", positions=[[5, 5, 5], [6, 5, 5], [5, 6, 5]],
                 tags=[1, 1, 1], cell=[20]*3)


def test_break_moves_exactly_one_selected_hydrogen_and_leaves_oh_stationary():
    atoms = water()
    before = atoms.positions.copy()
    event, _ = break_bond(atoms, np.random.default_rng(1), particles=["H"], bond="O-H")
    assert event is not None and event["operation"] == "break"
    indices = event["moved_indices"]
    assert len(indices) == 1 and atoms[indices[0]].symbol == "H"
    stationary = sorted(set(range(len(atoms))) - set(indices))
    np.testing.assert_array_equal(atoms.positions[stationary], before[stationary])
    assert np.linalg.norm(atoms.positions[indices] - before[indices]) > 0


def test_fixed_selected_atoms_are_not_replaced_by_an_unselected_atom():
    atoms = water()
    atoms.set_constraint(FixAtoms(indices=[1, 2]))
    original = {name: values.copy() for name, values in atoms.arrays.items()}
    event, diagnostic = break_bond(atoms, np.random.default_rng(1), particles=["H"], bond="O-H")
    assert event is None and diagnostic == "BondChange_Skipped"
    for name, values in original.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)


def test_single_atom_break_wrapper_roundtrip_and_rollback():
    atoms = water()
    original = {name: values.copy() for name, values in atoms.arrays.items()}
    op = parse_operators([dict(method="bond_break", particles=["H"], bond="O-H")])[0][0]
    op._print = op._debug = lambda *args: None
    prepare_operators([op], sorted(set(atoms.numbers)))
    assert "moving_particles" not in op.as_dict()
    assert parse_operators([op.as_dict()])[0][0].as_dict() == op.as_dict()
    proposal = op.propose(atoms, np.random.default_rng(1))
    assert proposal.valid
    assert len(proposal.metadata["bond_change"]["moved_indices"]) == 1
    proposal.rollback()
    for name, values in original.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)


@pytest.mark.parametrize("particles", [[], "H", ["Q"], [None], ["H2O"], ["HO"], ["H2"]])
@pytest.mark.parametrize("function", [break_bond, form_bond])
def test_only_element_symbols_are_accepted(particles, function):
    with pytest.raises(ValueError):
        function(water(), np.random.default_rng(1), particles=particles, bond="O-H")


@pytest.mark.parametrize("name", ["moving_particles", "departing_fragments"])
def test_removed_fragment_selector_has_a_migration_error(name):
    with pytest.raises(TypeError, match="Obsolete"):
        break_bond(water(), np.random.default_rng(1), particles=["H"], bond="O-H", **{name: ["H"]})
    with pytest.raises(TypeError, match="Obsolete"):
        parse_operators([dict(method="bond_break", particles=["H"], bond="O-H", **{name: ["H"]})])


def test_nonterminal_atom_cannot_drag_the_rest_of_its_molecule():
    atoms = water()
    before = atoms.positions.copy()
    event, _ = break_bond(atoms, np.random.default_rng(1), particles=["O"], bond="O-H")
    assert event is None
    np.testing.assert_array_equal(atoms.positions, before)


def test_formation_cannot_drag_a_bonded_pair_of_selected_atoms():
    atoms = Atoms("OHH", positions=[[5, 5, 5], [7.5, 5, 5], [8.12, 5, 5]],
                  tags=[1, 2, 2], cell=[20]*3)
    before = atoms.positions.copy()
    event, _ = form_bond(atoms, np.random.default_rng(1), particles=["H"], bond="O-H")
    assert event is None
    np.testing.assert_array_equal(atoms.positions, before)


def acceptor_fixture():
    return Atoms("OOH", positions=[[5, 5, 5], [9, 5, 5], [7, 5, 5]],
                 tags=[1, 2, 3], cell=[20]*3)


@pytest.mark.parametrize("group", ["`index 1`", "`id 2`", "`tag 2`", "`symbol O` and `index 1`"])
def test_form_moves_selected_fragment_onto_only_the_selected_acceptor(group):
    atoms = acceptor_fixture()
    before = atoms.positions.copy()
    event, _ = form_bond(atoms, np.random.default_rng(3), particles=["H"], bond="O-H",
                         acceptor_group=group, eligible_tags=[3])
    assert event is not None and event["operation"] == "form"
    assert event["pair"] == [1, 2] and event["moved_indices"] == [2]
    np.testing.assert_array_equal(atoms.positions[:2], before[:2])
    assert atoms.get_distance(1, 2) == pytest.approx(0.97)


@pytest.mark.parametrize("group", ["`symbol C`", "`index 2`"])
def test_empty_or_self_acceptor_group_never_uses_another_site(group):
    atoms = acceptor_fixture()
    before = {name: values.copy() for name, values in atoms.arrays.items()}
    event, diagnostic = form_bond(atoms, np.random.default_rng(1), particles=["H"], bond="O-H",
                                  acceptor_group=group)
    assert event is None and diagnostic == "BondChange_Skipped"
    for name, values in before.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)


@pytest.mark.parametrize("group", ["`index -1`", "`index 3`", "`missing O`", "`symbol O` and"])
def test_invalid_acceptor_expression_restores_tags_and_coordinates(group):
    atoms = acceptor_fixture()
    atoms.set_tags([1, 1, 3])
    before = {name: values.copy() for name, values in atoms.arrays.items()}
    with pytest.raises(ValueError, match="acceptor_group"):
        form_bond(atoms, np.random.default_rng(1), particles=["H"], bond="O-H", acceptor_group=group)
    for name, values in before.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)


def test_form_does_not_move_an_unselected_partner_when_selected_atom_is_fixed():
    atoms = acceptor_fixture()
    atoms.set_constraint(FixAtoms(indices=[2]))
    before = atoms.positions.copy()
    event, _ = form_bond(atoms, np.random.default_rng(1), particles=["H"], bond="O-H")
    assert event is None
    np.testing.assert_array_equal(atoms.positions, before)


def test_form_wrapper_roundtrip_rollback_and_stationary_acceptor():
    atoms = acceptor_fixture()
    before = {name: values.copy() for name, values in atoms.arrays.items()}
    op = parse_operators([dict(method="bond_form", particles=["H"], bond="O-H",
                               acceptor_group="`index 1`")])[0][0]
    op._print = op._debug = lambda *args: None
    prepare_operators([op], sorted(set(atoms.numbers)))
    assert parse_operators([op.as_dict()])[0][0].as_dict() == op.as_dict()
    proposal = op.propose(atoms, np.random.default_rng(3))
    assert proposal.valid and proposal.metadata["bond_change"]["pair"] == [1, 2]
    np.testing.assert_array_equal(atoms.positions[:2], before["positions"][:2])
    proposal.rollback()
    for name, values in before.items():
        np.testing.assert_array_equal(atoms.arrays[name], values)


@pytest.mark.parametrize("operation,distance,particles", [(break_bond, 2.5, ["H"]),
                                                           (form_bond, 1.0, ["H"])])
def test_separate_operations_never_fall_back_to_the_other(operation, distance, particles):
    atoms = Atoms("OH", positions=[[5, 5, 5], [5+distance, 5, 5]], tags=[1, 2], cell=[20]*3)
    event, _ = operation(atoms, np.random.default_rng(1), particles=particles, bond="O-H")
    assert event is None


def test_removed_combined_operator_has_a_migration_error():
    with pytest.raises(ValueError, match="replaced by bond_break and bond_form"):
        parse_operators([dict(method="bond_change", particles=["H"], bond="O-H")])
