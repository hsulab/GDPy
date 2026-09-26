#!/usr3/bin/env python3
# -*- coding: utf-8 -*


import numpy as np

from ase import units

try:
    import plumed
except Exception as e:
    print(e)


class AsePlumed(object):

    def __init__(self, atoms, timestep, inp_file="plumed.dat", out_file="plumed.out"):

        self.atoms = atoms
        self.timestep = timestep
        self.natoms = len(atoms)
        self.masses = self.atoms.get_masses().copy()  # masses cannot change

        self.inp_file = inp_file
        self.out_file = out_file

        self.worker = self.initialize()

        return

    def initialize(self):
        # init
        p_md = plumed.Plumed()

        # units
        energyUnits = units.mol / units.kJ  # eV to kJ/mol
        lengthUnits = 1.0 / units.nm  # angstrom to nm
        timeUnits = 1.0 / (1000.0 * units.fs)  # fs to ps

        p_md.cmd("setMDEnergyUnits", energyUnits)
        p_md.cmd("setMDLengthUnits", lengthUnits)
        p_md.cmd("setMDTimeUnits", timeUnits)

        # inp, out
        p_md.cmd("setPlumedDat", self.inp_file)
        p_md.cmd("setLogFile", self.out_file)

        # simulation details
        p_md.cmd("setTimestep", self.timestep)
        p_md.cmd("setNatoms", self.natoms)
        p_md.cmd("setMDEngine", "ase")

        # finally!
        p_md.cmd("init")

        return p_md

    def external_forces(
        self,
        istep,
        new_energy=None,
        new_forces=None,  # sometimes use forces not attached to self.atoms
        new_virial=None,
        delta_forces=False,
    ):
        """return external forces from plumed"""
        # structure info
        positions = self.atoms.get_positions().copy()
        cell = self.atoms.cell[:].copy()

        if new_forces is None:
            forces = self.atoms.get_forces().copy()
        else:
            forces = new_forces.copy()
        original_forces = forces.copy()

        if new_energy is None:
            energy = self.atoms.get_potential_energy()
        else:
            energy = new_energy

        # TODO: get virial
        virial = np.zeros((3, 3))

        self.worker.cmd("setStep", istep)
        self.worker.cmd("setMasses", self.masses)
        self.worker.cmd("setForces", forces)
        self.worker.cmd("setPositions", positions)
        self.worker.cmd("setEnergy", energy)
        self.worker.cmd("setBox", cell)
        self.worker.cmd("setVirial", virial)
        self.worker.cmd("calc", None)

        # implent plumed external forces into momenta
        if delta_forces:
            plumed_forces = forces - original_forces
        else:
            plumed_forces = forces

        return plumed_forces

    def finalize(self):
        self.worker.finalize()


if __name__ == "__main__":
    ...
