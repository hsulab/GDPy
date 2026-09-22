import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.objective import evaluate_candidate


def _evaluated_atoms(symbols, energy):
    atoms = Atoms(symbols)
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    atoms.info["key_value_pairs"] = {}
    return atoms


def test_cohesive_energy_uses_elemental_chemical_potentials():
    atoms = _evaluated_atoms("CuO", 10.0)
    evaluate_candidate(atoms, "cohesive_energy", {"Cu": 1.0, "O": 2.0})

    assert atoms.info["key_value_pairs"]["target"] == pytest.approx(7.0)
    assert atoms.info["key_value_pairs"]["raw_score"] == pytest.approx(-7.0)


def test_formation_energy_uses_objective_chemical_potentials():
    atoms = _evaluated_atoms("CuO2", 12.0)
    atoms.info["identity_stats"] = {"Cu": 1, "O": 2}

    evaluate_candidate(
        atoms,
        objective_target="formation_energy",
        chemical_potentials={"Cu": 1.0, "O": 2.0},
    )

    assert atoms.info["key_value_pairs"]["target"] == pytest.approx(7.0)
    assert atoms.info["key_value_pairs"]["raw_score"] == pytest.approx(-7.0)


def test_energy_scoring_needs_no_forces_and_preserves_metadata():
    atoms = _evaluated_atoms('Cu', -3.)
    atoms.info['key_value_pairs']['generation'] = 2
    evaluate_candidate(atoms, 'energy')
    assert atoms.info['key_value_pairs'] == {'generation': 2, 'target': -3., 'raw_score': 3.}
    with pytest.raises(AssertionError, match='already has raw_score'):
        evaluate_candidate(atoms, 'energy')


def test_formation_energy_counts_molecular_fragments():
    atoms = _evaluated_atoms('Cu2CO', 10.)
    atoms.info['identity_stats'] = {'Cu': 2, 'CO': 1}
    evaluate_candidate(atoms, 'formation_energy', {'Cu': 1., 'CO': 3.})
    assert atoms.info['key_value_pairs']['target'] == 5.
    assert atoms.info['key_value_pairs']['raw_score'] == -5.


@pytest.mark.parametrize('target', ['formation_energy', 'cohesive_energy'])
def test_reference_objectives_require_chemical_potentials(target):
    atoms = _evaluated_atoms('Cu', 1.)
    with pytest.raises(AssertionError, match='chemical_potentials'):
        evaluate_candidate(atoms, target)
    assert atoms.info['key_value_pairs'] == {}


def test_formation_energy_requires_identity_counts():
    atoms = _evaluated_atoms('Cu', 1.)
    with pytest.raises(AssertionError, match='identity_stats'):
        evaluate_candidate(atoms, 'formation_energy', {'Cu': 0.})


def test_both_engines_use_the_shared_evaluator():
    from gdpx.exploration.basin_hopping import engine as bh
    from gdpx.exploration.genetic_algorithm import engine as ga
    assert bh.evaluate_candidate is ga.evaluate_candidate is evaluate_candidate
