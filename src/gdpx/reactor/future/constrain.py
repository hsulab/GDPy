#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" ase implementation of fort.188
    to sample a specific reactions,
    combine interplotation and overshooting,
    automatically sample IS, FS and intermediate points between them
"""

import numpy as np

import ase
from ase.io import read, write
from ase.constraints import FixBondLength, FixAtoms

from ase.neb import NEB
from ase.optimize import BFGS

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin

from sella import Sella, Constraints

atoms_IS = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-IS.xyz")
atoms_TS = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-TS.xyz")
atoms_FS = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-FS.xyz")

def add_188():
    cons = FixAtoms(indices=[a.index for a in atoms if a.position[2]<7.0])
    cons_bond = FixBondLength(0, 23)

    atoms.set_constraint([cons, cons_bond])

    return

from eann.interface.ase.calculator import Eann
calc = Eann(
    # for now, ase interface only supports double precision
    model = "/mnt/scratch2/users/40247882/catsign/ZnOx/P188/ZnO_200_DOUBLE.pt",
    type_map = {
        "C": 0, "O": 1, "Zn": 2
    }
)

def run_sella():
    """"""
    atoms = atoms_TS
    atoms.calc = calc
    dyn = Sella(
        atoms,
        order = 1,
        internal = False,
        trajectory="opt.traj"
    )
    dyn.run(0.05, 100)

    return

def run_single():
    maxforces = []
    energies = []
    for atoms in [atoms_IS, atoms_TS, atoms_FS]:
        calc.reset()
        atoms.calc = calc
        forces = atoms.get_forces()
        maxforce = np.max(np.fabs(forces))
        maxforces.append(maxforce)
        energies.append(atoms.get_potential_energy())
    energies = np.array(energies)
    print(energies)
    print(energies - energies[0])
    print(maxforces)

    return

def run_NEB():
    initial = atoms_IS
    final = atoms_FS

    nimages = 5
    images = [initial]
    images += [initial.copy() for i in range(nimages-2)]
    images.append(final)

    cons = FixAtoms(indices=[a.index for a in initial if a.position[2]<7.0])

    for atoms in images:
        calc = Eann(
            model = "/mnt/scratch2/users/40247882/catsign/ZnOx/P188/ZnO_200_DOUBLE.pt",
            type_map = {
                "C": 0, "O": 1, "Zn": 2
            }
        )
        atoms.calc = calc
        atoms.set_constraint(cons)

    neb = NEB(
        images, allow_shared_calculator=False,
        k=0.1
        # dynamic_relaxation = False
    )
    #neb = SingleCalculatorNEB(images)
    
    neb.interpolate()
    #print(neb.images)
    
    qn = BFGS(neb, trajectory="neb.traj")
    qn.run(fmax=0.05, steps=100)

    return

def run_opt():
    maxforces = []
    energies = []
    for i, atoms in enumerate([atoms_IS, atoms_TS, atoms_FS]):
        calc.reset()
        atoms.calc = calc
        qn = BFGS(atoms, trajectory=f"opt-{i}.traj")
        qn.run(fmax=0.05, steps=100)
        # info
        forces = atoms.get_forces()
        maxforce = np.max(np.fabs(forces))
        maxforces.append(maxforce)
        energies.append(atoms.get_potential_energy())
    energies = np.array(energies)
    print(energies)
    print(energies - energies[0])
    print(maxforces)

    return

# MD
def run_MD():
    timestep = 2.0
    temperature = 600
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    dyn = Langevin(
        atoms, 
        timestep = timestep*ase.units.fs, 
        temperature_K = temperature, 
        friction = 0.002, # TODO: what is the unit?
        fixcm = True
    )

    write("traj.xyz", atoms)
    for i in range(10):
        dyn.step()
        write("traj.xyz", atoms, append=True)

if __name__ == "__main__":
    #run_single()
    run_opt()
    #run_NEB()
    #run_sella()
    pass

