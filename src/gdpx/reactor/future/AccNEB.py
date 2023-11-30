#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from inspect import iscoroutine
import numpy as np

from ase.io import read, write
from ase.constraints import FixAtoms, FixBondLengths
from ase.calculators.emt import EMT
from ase.neb import NEB
#from ase.neb import SingleCalculatorNEB
from ase.optimize import BFGS
from ase.geometry import wrap_positions
from sklearn.preprocessing import scale
from torch import saddmm


class AccNEB():

    def __init__(self):

        return
    
    def __call__(self):

        return


def test_cons(atoms, calc):
    atoms.calc = calc
    atoms.set_constraint(
        FixBondLengths(
            pairs = [[0, 23]],
            bondlengths = [2.426]
        )
    )

    qn = BFGS(atoms, trajectory="cons.traj")
    qn.run(fmax=0.05, steps=50)

    return



if __name__ == "__main__":
    # test LASP-NEB
    from gdpx.computation.lasp import LaspNN
    pot_path = "/mnt/scratch2/users/40247882/catsign/lasp-main/ZnCrOCH.pot"
    pot = dict(
        C  = pot_path,
        O  = pot_path,
        Cr = pot_path,
        Zn = pot_path
    )
    calc = LaspNN(
        directory = "./LaspNN-Worker",
        command = "mpirun -n 4 lasp",
        pot=pot
    )


    initial = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-IS.xyz")
    transition = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-TS.xyz")
    final = read("/mnt/scratch2/users/40247882/catsign/ZnOx/P188/CO-FS.xyz")

    transition = read("/mnt/scratch2/users/40247882/catsign/ZnOx/cases/TStest.xyz")

    cons_inddices = list(range(1,13)) + list(range(24,36))
    constraint = FixAtoms(indices=cons_inddices) # first atom is O
    #constraint = FixAtoms(indices=[1,2,3,4,5,6,7,8]) # first atom is O

    transition.set_constraint(constraint)
    transition.calc = calc

    # test_cons(transition, calc)
    constrdyn = AccCons(transition, cpairs=[0,23])
    constrdyn()
    exit()

    nimages = 7
    images = [initial]
    images += [initial.copy() for i in range(nimages-2)]
    images.append(final)

    for atoms in images:
        calc = LaspNN(
            directory = "./LaspNN-Worker",
            command = "mpirun -n 4 lasp",
            pot=pot
        )
        atoms.calc = calc
        atoms.set_constraint(constraint)

    neb = NEB(
        images, 
        allow_shared_calculator=False,
        k = 0.1, # spring constant
        climb = False,
        # dynamic_relaxation = False
    )
    #neb = SingleCalculatorNEB(images)

    neb.interpolate()
    #print(neb.images)

    qn = BFGS(neb, trajectory="neb.traj")
    qn.run(fmax=0.05, steps=50)

    pass