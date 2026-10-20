"""Bond geometry, fragment identity and reversible proposal edits."""

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms

from gdpx.sampling import parse_operators
from gdpx.sampling.fragments import FragmentPolicy, fragment_policy_for
from gdpx.sampling.geometry import prepare_operators
from gdpx.utils.atoms_tags import get_tags_per_species


def water():
    frame = Atoms("OOHH", positions=[[5, 5, 3], [5, 5, 5], [6, 5, 5], [5, 6, 5]],
                  tags=[0, 1, 1, 1], cell=[20]*3, pbc=True)
    frame.set_constraint(FixAtoms(indices=[0, 1]))
    frame.calc = SinglePointCalculator(frame, energy=0., forces=np.zeros((4, 3)))
    return frame


def operator(frame=None, **kwargs):
    settings = dict(method="bond_break", particles=["H"], bond=["O", "H"],
                    allow_isolated=True)
    settings.update(kwargs)
    op = parse_operators([settings])[0][0]
    op._print = op._debug = lambda *args: None
    if frame is not None:
        prepare_operators([op], sorted(set(frame.numbers)))
    return op


def test_water_break_form_and_exact_rollback():
    frame = water()
    original = {name: array.copy() for name, array in frame.arrays.items()}
    calc = frame.calc
    frame.info["bond_change"] = {"previous": True}
    op = operator(frame, particles=["H"], cutoffs=[0., 1.2])
    proposal = op.propose(frame, np.random.default_rng(1))
    assert proposal.valid
    event = proposal.metadata["bond_change"]
    stationary, moving = event["pair"]
    assert stationary == 1 and frame[moving].symbol == "H"
    assert frame.get_distance(stationary, moving, mic=True) == pytest.approx(1.455)
    assert set(get_tags_per_species(frame)) == {"O", "HO", "H"}
    np.testing.assert_array_equal(frame.positions[:2], original["positions"][:2])
    proposal.rollback()
    for name, values in original.items():
        np.testing.assert_array_equal(frame.arrays[name], values)
    assert frame.calc is calc
    assert frame.info["bond_change"] == {"previous": True}
    op.propose(frame, np.random.default_rng(1)).commit()
    broken = frame.positions.copy()
    join = operator(frame, method="bond_form", particles=["H"], cutoffs=[1.2, 3.5])
    proposal = join.propose(frame, np.random.default_rng(3))
    assert proposal.valid
    assert frame.get_distance(*proposal.metadata["bond_change"]["pair"], mic=True) == pytest.approx(0.97)
    assert set(get_tags_per_species(frame)) == {"O", "H2O"}
    np.testing.assert_array_equal(frame.get_tags(), [0, 1, 1, 1])
    proposal.rollback()
    np.testing.assert_array_equal(frame.positions, broken)


def test_break_does_not_drag_a_multiatom_fragment():
    frame = Atoms("CCHH", positions=[[5, 5, 5], [6.5, 5, 5], [5, 6, 5], [6.5, 6, 5]],
                  tags=[1]*4, cell=[20]*3, pbc=True)
    original = {name: values.copy() for name, values in frame.arrays.items()}
    proposal = operator(frame, particles=["C"], bond="C-C").propose(frame, np.random.default_rng(2))
    assert not proposal.valid
    for name, values in original.items():
        np.testing.assert_array_equal(frame.arrays[name], values)


@pytest.mark.parametrize("mode", ["break", "form"])
def test_surface_hydrogen_keeps_separate_positive_tag(mode):
    distance = 1.0 if mode == "break" else 2.5
    frame = Atoms("OH", positions=[[5, 5, 5], [5+distance, 5, 5]], tags=[0, 4], cell=[20]*3)
    frame.set_constraint(FixAtoms(indices=[0]))
    original = frame.positions.copy()
    op = operator(frame, method=f"bond_{mode}", particles=["H"])
    proposal = op.propose(frame, np.random.default_rng(1))
    assert proposal.valid
    np.testing.assert_array_equal(frame.get_tags(), [0, 4])
    np.testing.assert_array_equal(frame.positions[0], original[0])
    proposal.rollback()
    np.testing.assert_array_equal(frame.positions, original)


def test_periodic_bond_uses_minimum_image():
    frame = water()
    frame.positions[1:] = [[19.7, 5, 5], [0.7, 5, 5], [19.7, 6, 5]]
    op = operator(frame, particles=["H"], cutoffs=[0., 1.2])
    proposal = op.propose(frame, np.random.default_rng(7))
    assert proposal.valid
    assert frame.get_distance(*proposal.metadata["bond_change"]["pair"], mic=True) == pytest.approx(1.455)
    proposal.rollback()
    np.testing.assert_array_equal(frame.positions[2], [0.7, 5, 5])


@pytest.mark.parametrize("mode,positions", [
    ("break", [[5, 5, 5], [6.5, 5, 5], [5.75, 6.299038106, 5]]),
    ("form", [[5, 5, 5], [6.5, 5, 5], [8, 5, 5]]),
])
def test_cycles_and_intracomponent_formation_are_skipped(mode, positions):
    frame = Atoms("C3", positions=positions, tags=[1]*3, cell=[20]*3)
    op = operator(frame, method=f"bond_{mode}", particles=["C"], bond=["C", "C"], cutoffs=([0., 1.8] if mode == "break" else [2., 3.5]))
    proposal = op.propose(frame, np.random.default_rng(1))
    assert not proposal.valid and proposal.closed
    np.testing.assert_array_equal(frame.positions, positions)


def test_all_fixed_is_skipped():
    frame = water()
    frame.set_constraint(FixAtoms(indices=range(len(frame))))
    proposal = operator(frame, cutoffs=[0., 1.2]).propose(frame, np.random.default_rng(2))
    assert not proposal.valid and proposal.closed


def test_collision_exhaustion_and_exception_restore_tags_positions_info(monkeypatch):
    frame = water()
    positions, tags, calc = frame.positions.copy(), frame.get_tags(), frame.calc
    op = operator(frame, cutoffs=[0., 1.2], max_random_attempts=2)
    import gdpx.structures.geometry.bond_change as module
    monkeypatch.setattr(module, "check_atomic_distances", lambda *args, **kwargs: False)
    proposal = op.propose(frame, np.random.default_rng(2))
    assert not proposal.valid and proposal.diagnostic == "BondChange_Failed"
    np.testing.assert_array_equal(frame.positions, positions)
    np.testing.assert_array_equal(frame.get_tags(), tags)
    assert frame.calc is calc and "bond_change" not in frame.info
    def fail(*args, **kwargs):
        raise RuntimeError("collision checker failed")
    monkeypatch.setattr(module, "check_atomic_distances", fail)
    with pytest.raises(RuntimeError, match="collision checker"):
        op.propose(frame, np.random.default_rng(2))
    np.testing.assert_array_equal(frame.positions, positions)
    np.testing.assert_array_equal(frame.get_tags(), tags)
    assert frame.calc is calc


def test_policy_tracks_recombination_and_other_water_transfer():
    policy = FragmentPolicy(bond_cutoffs={"O-H": 1.25})
    frame = water()
    frame.positions[2] = [7.5, 5, 5]
    policy.reconcile(frame)
    np.testing.assert_array_equal(frame.get_tags(), [0, 1, 2, 1])
    # Recombination merges the detached H into the oxygen-containing fragment.
    frame.positions[2] = [6, 5, 5]
    policy.reconcile(frame)
    np.testing.assert_array_equal(frame.get_tags(), [0, 1, 1, 1])
    # Transfer to surface oxygen does not join the substrate tag.
    frame.positions[2] = [6, 5, 3]
    policy.reconcile(frame)
    np.testing.assert_array_equal(frame.get_tags(), [0, 1, 2, 1])
    frame.extend(Atoms("OHH", positions=[[9, 5, 5], [10, 5, 5], [9, 6, 5]], tags=[7]*3))
    frame.positions[2] = [8, 5, 5]
    policy.reconcile(frame)
    assert frame.get_tags()[2] == frame.get_tags()[4] == 7
    once = frame.get_tags()
    policy.reconcile(frame)
    np.testing.assert_array_equal(frame.get_tags(), once)


def test_config_roundtrip_and_rng_replay():
    first, second = water(), water()
    op = operator(first)
    replay = parse_operators([op.as_dict()])[0][0]
    prepare_operators([replay], sorted(set(second.numbers)))
    replay._print = replay._debug = lambda *args: None
    assert replay.as_dict() == op.as_dict()
    op.propose(first, np.random.default_rng(11)).commit()
    replay.propose(second, np.random.default_rng(11)).commit()
    np.testing.assert_array_equal(first.positions, second.positions)
    np.testing.assert_array_equal(first.get_tags(), second.get_tags())


@pytest.mark.parametrize("settings", [
    {"particles": []}, {"particles": ["Q"]}, {"bond": ["Q", "H"]},
    {"cutoffs": [1., 0.]}, {"cutoffs": [-1., 3.]}, {"cutoffs": [0., float("nan")]},
    {"covalent_ratio": [0.8, 1.]}, {"covalent_ratio": [1., 1.2]},
    {"covalent_ratio": [0., 1.2]},
])
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        operator(**settings)


def test_conflicting_fragment_policies_are_rejected():
    with pytest.raises(ValueError, match="same fragment policy"):
        fragment_policy_for([operator(), operator(covalent_ratio=[0.8, 1.3])])


def test_hydrogen_bond_uses_its_own_covalent_radii():
    frame = Atoms("H2", positions=[[5, 5, 5], [5.7, 5, 5]], tags=[1, 1], cell=[20]*3)
    frame.set_constraint(FixAtoms(indices=[0]))
    op = operator(frame, particles=["H"], bond=["H", "H"])
    proposal = op.propose(frame, np.random.default_rng(1))
    assert proposal.valid
    assert frame.get_distance(0, 1) == pytest.approx(0.93)
    proposal.rollback()


def test_periodic_alias_of_target_distance_is_rejected():
    frame = Atoms("OH", positions=[[0.5, 1, 1], [1.5, 1, 1]], tags=[1, 1], cell=[2.5]*3, pbc=True)
    frame.set_constraint(FixAtoms(indices=[0]))
    positions = frame.positions.copy()
    op = operator(frame, cutoffs=[0., 1.2], max_random_attempts=3, skip_distance_check=True)
    proposal = op.propose(frame, np.random.default_rng(1))
    assert not proposal.valid
    np.testing.assert_array_equal(frame.positions, positions)


def test_fragment_policy_rejects_topology_changing_operator_mixtures():
    swap = parse_operators([dict(method="swap_type", particles=["Cu", "Ni"], chempots=[0., 0.])])[0][0]
    with pytest.raises(ValueError, match="atom-order-preserving"):
        fragment_policy_for([operator(), swap])
