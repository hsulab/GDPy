""" 

"""

import numpy as np

from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import Stationary
from ase import units

def force_temperature(atoms, temperature, unit="K"):
    """ force (nucl.) temperature to have a precise value

    Parameters:
    atoms: ase.Atoms
        the structure
    temperature: float
        nuclear temperature to set
    unit: str
        'K' or 'eV' as unit for the temperature
    """

    eps_temp = 1e-12

    if unit == "K":
        E_temp = temperature * units.kB
    elif unit == "eV":
        E_temp = temperature
    else:
        raise UnitError("'{}' is not supported, use 'K' or 'eV'.".format(unit))

    # check DOF
    ndof = 3*len(atoms)
    for constraint in atoms._constraints:
        ndof -= constraint.get_removed_dof(atoms)

    # calculate kinetic energy and get the scale
    if temperature > eps_temp:
        E_kin0 = atoms.get_kinetic_energy() / (0.5 * ndof)
        gamma = E_temp / E_kin0
    else:
        gamma = 0.0
    
    atoms.set_momenta(atoms.get_momenta() * np.sqrt(gamma))

    return

