"""Lossless accepted-state persistence without serializing live calculators."""

import pickle

from ase import Atoms
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator


def save_accepted_state(path, atoms, energy):
    # todict references arrays: pickle streams them without making an Atoms copy.
    results = {
        key: value for key, value in getattr(atoms.calc, "results", {}).items() if key in all_properties
    }
    results["energy"] = energy
    with path.open("wb") as stream:
        pickle.dump(dict(version=1, atoms=atoms.todict(), results=results), stream)


def load_accepted_state(path):
    with path.open("rb") as stream:
        data = pickle.load(stream)
    if data.get("version") != 1:
        raise ValueError("Unsupported accepted-state checkpoint; start a new run.")
    values = data["atoms"]
    constraints = values.pop("constraints", [])
    atoms = Atoms.fromdict(values)
    atoms.set_constraint(constraints)
    atoms.calc = SinglePointCalculator(atoms, **data["results"])
    return atoms
