"""Portable atomic structures without serializing live calculators."""
from ase import Atoms
from ase.constraints import dict2constraint
from ase.calculators.calculator import all_properties
from ase.calculators.singlepoint import SinglePointCalculator

from .checkpoint import load_data, save_data


def save_structure(path, atoms, results=None):
    values = atoms.todict()  # Borrow arrays; the codec streams them to NumPy storage.
    values['constraints'] = [constraint.todict() for constraint in atoms.constraints]
    save_data(path, dict(version=2, atoms=values, results=results))


def load_structure(path):
    data = load_data(path)
    if data.get('version') != 2:
        raise ValueError('Unsupported structure checkpoint version; start a new run.')
    values = data['atoms']
    constraints = values.pop('constraints', [])
    atoms = Atoms.fromdict(values)
    atoms.set_constraint([dict2constraint(item) for item in constraints])
    if data['results'] is not None:
        atoms.calc = SinglePointCalculator(atoms, **data['results'])
    return atoms


def save_accepted_state(path, atoms, energy):
    results = {key: value for key, value in getattr(atoms.calc, 'results', {}).items() if key in all_properties}
    results['energy'] = energy
    save_structure(path, atoms, results)


def load_accepted_state(path):
    return load_structure(path)
