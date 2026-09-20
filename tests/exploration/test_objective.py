import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

from gdpx.exploration.genetic_algorithm.engine import GeneticAlgorithmEngine
from gdpx.exploration.monte_carlo.concurrent_hopping import evaluate_candidate


def _evaluated_atoms(symbols, energy):
    atoms = Atoms(symbols)
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    atoms.info["key_value_pairs"] = {}
    return atoms


def test_ga_cohesive_energy_uses_objective_chemical_potentials():
    atoms = _evaluated_atoms("CuO", 10.0)
    engine = object.__new__(GeneticAlgorithmEngine)
    engine.target = "cohesive_energy"
    engine.objective = {
        "target": "cohesive_energy",
        "chemical_potentials": {"Cu": 1.0, "O": 2.0},
    }

    engine.evaluate_candidate(atoms)

    assert atoms.info["key_value_pairs"]["target"] == pytest.approx(7.0)
    assert atoms.info["key_value_pairs"]["raw_score"] == pytest.approx(-7.0)


def test_concurrent_formation_energy_uses_objective_chemical_potentials():
    atoms = _evaluated_atoms("CuO2", 12.0)
    atoms.info["identity_stats"] = {"Cu": 1, "O": 2}

    evaluate_candidate(
        atoms,
        objective_target="formation_energy",
        chemical_potentials={"Cu": 1.0, "O": 2.0},
    )

    assert atoms.info["key_value_pairs"]["target"] == pytest.approx(7.0)
    assert atoms.info["key_value_pairs"]["raw_score"] == pytest.approx(-7.0)
