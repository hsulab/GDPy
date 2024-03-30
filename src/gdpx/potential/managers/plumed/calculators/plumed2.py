#!/usr3/bin/env python3
# -*- coding: utf-8 -*

import copy
import os
import pathlib
import shutil

from typing import List

import numpy as np

from ase import units
from ase.units import fs, mol, kJ, nm
from ase.calculators.calculator import Calculator, all_changes

try:
    import plumed
except Exception as e:
    print(e)


"""A plumed wrapper for ase.

There is an official plumed calculator in ase master branch but not in v3.22.1 that
can be installed from conda.

Units setup
warning: inputs and outputs of plumed will still be in
plumed units.

The change of Plumed units to ASE units is:
kjoule/mol to eV
nm to Angstrom
ps to ASE time units
ASE and plumed - charge unit is in e units
ASE and plumed - mass unit is in a.m.u units

Notes:

    conda install -c conda-forge plumed
    conda install -c conda-forge py-plumed

"""


def update_input_value(line: str, key: str, value, func: callable):
    """Update the given key with the new value."""
    shift = len(key) + 1  # key name and =
    if line.find(key) != -1:
        ini = line.find(key)
        end = line.find(" ", ini)
        if end == -1:
            prev = line[ini + shift :]
            line = line[: ini + shift] + func(prev, value)
        else:
            prev = line[ini + shift : end]
            line = line[: ini + shift] + func(prev, value) + line[end:]

    return line


def update_stride_and_file(input_lines: List[str], wdir: str, stride: int) -> List[str]:
    """"""
    input_lines, parsed_lines = copy.deepcopy(input_lines), []
    for line in input_lines:
        parsed_line = update_input_value(
            line, "FILE", wdir, func=lambda x, y: os.path.join(y, x)
        )
        parsed_line = update_input_value(
            parsed_line, "STRIDE", stride, func=lambda x, y: str(y)
        )
        parsed_lines.append(parsed_line)

    return parsed_lines


class Plumed(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        input: List[str],
        atoms=None,
        kT=1.0,
        restart=False,
        use_charge=False,
        update_charge=False,
    ):
        """
        Plumed calculator is used for simulations of enhanced sampling methods
        with the open-source code PLUMED (plumed.org).

        [1] The PLUMED consortium, Nat. Methods 16, 670 (2019)
        [2] Tribello, Bonomi, Branduardi, Camilloni, and Bussi,
        Comput. Phys. Commun. 185, 604 (2014)

        Parameters
        ----------
        input: List of strings
            It contains the setup of plumed actions

        atoms: Atoms
            Atoms object to be attached

        kT: float. Default 1.
            Value of the thermal energy in eV units. It is important for
            some methods of plumed like Well-Tempered Metadynamics.

        restart: boolean. Default False
            True if the simulation is restarted.

        use_charge: boolean. Default False
            True if you use some collective variable which needs charges. If
            use_charges is True and update_charge is False, you have to define
            initial charges and then this charge will be used during all
            simulation.

        update_charge: boolean. Default False
            True if you want the charges to be updated each time step. This
            will fail in case that calc does not have 'charges' in its
            properties.

        .. note:: For this case, the calculator is defined strictly with the
            object atoms inside. This is necessary for initializing the
            Plumed object. For conserving ASE convention, it can be initialized
            as atoms.calc = (..., atoms=atoms, ...)


        .. note:: In order to guarantee a proper restart, the user has to fix
            momenta, positions and Plumed.istep, where the positions and
            momenta corresponds to the last configuration in the previous
            simulation, while Plumed.istep is the number of timesteps
            performed previously. This can be done using
            ase.calculators.plumed.restart_from_trajectory.
        """

        # if atoms is None:
        #    raise TypeError('plumed calculator has to be defined with the \
        #                     object atoms inside.')

        self.istep = 0
        Calculator.__init__(self, atoms=atoms)

        self.input = input
        self.use_charge = use_charge
        self.update_charge = update_charge

        self.kT = kT
        self.restart = restart

        self._timestep = None
        self._stride = 1

        return

    @property
    def timestep(self):
        """"""

        return self._timestep

    @timestep.setter
    def timestep(self, timestep):
        """"""
        self._timestep = timestep

        return

    @property
    def stride(self):

        return self._stride

    @stride.setter
    def stride(self, stride):
        """"""
        self._stride = stride

        return self._stride

    def _prepare(
        self,
        natoms: int,
        input_lines: List[str],
        timestep: float,
        restart: bool,
        kT: float,
    ):
        """"""
        self.plumed = plumed.Plumed()

        # - basic configuration
        ps = 1000 * fs
        self.plumed.cmd("setMDEnergyUnits", mol / kJ)
        self.plumed.cmd("setMDLengthUnits", 1 / nm)
        self.plumed.cmd("setMDTimeUnits", 1 / ps)
        self.plumed.cmd("setMDChargeUnits", 1.0)
        self.plumed.cmd("setMDMassUnits", 1.0)
        self.plumed.cmd("setMDEngine", "ASE")

        self.plumed.cmd("setNatoms", natoms)
        self.plumed.cmd("setLogFile", os.path.join(self.directory, "plumed.out"))
        self.plumed.cmd("setTimestep", float(timestep))
        self.plumed.cmd("setRestart", restart)
        self.plumed.cmd("setKbT", float(kT))
        self.plumed.cmd("init")

        # - parse lines, update FILE and STRIDE
        input_lines, parsed_lines = copy.deepcopy(input_lines), []
        for line in input_lines:
            parsed_line = update_input_value(
                line, "FILE", self.directory, func=lambda x, y: os.path.join(y, x)
            )
            parsed_line = update_input_value(
                parsed_line, "STRIDE", self.stride, func=lambda x, y: str(y)
            )
            parsed_line = update_input_value(
                parsed_line, "PACE", self.stride, func=lambda x, y: str(y)
            )
            parsed_lines.append(parsed_line)

        for line in parsed_lines:
            self.plumed.cmd("readInputLine", line)

        # - save input lines
        inp_fpath = pathlib.Path(self.directory, "plumed.inp")
        with open(inp_fpath, "w") as fopen:
            fopen.write("\n".join(input_lines))

        return

    def _save_checkpoint(self, ckpt_wdir: pathlib.Path):
        """"""
        calc_wdir = pathlib.Path(self.directory)
        if (ckpt_wdir / calc_wdir.name).exists():
            shutil.rmtree(ckpt_wdir / calc_wdir.name)
        _ = shutil.copytree(calc_wdir, ckpt_wdir / calc_wdir.name)

        return

    def _load_checkpoint(
        self, ckpt_wdir: pathlib.Path, dst_wdir=None, start_step: int = 0
    ):
        """"""
        calc_wdir = list(ckpt_wdir.glob(f"*Plumed"))[0]
        if dst_wdir is None:
            dst_wdir = ckpt_wdir.parent
        else:
            dst_wdir = pathlib.Path(dst_wdir)
        _ = shutil.copytree(calc_wdir, dst_wdir / calc_wdir.name)

        if hasattr(self, "plumed"):
            self.plumed.finalize()
            delattr(self, "plumed")
        self.istep = int(start_step)
        self.input = ["RESTART"] + self.input
        self.restart = True

        return

    def calculate(
        self, atoms=None, properties=["energy", "forces"], system_changes=all_changes
    ):
        """"""
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.timestep is None:
            raise RuntimeError(
                "Plumed needs a valid timestep set by an external class."
            )

        if not hasattr(self, "plumed"):
            self._prepare(len(atoms), self.input, self.timestep, self.restart, self.kT)

        energy_bias, forces_bias = self.compute_bias(
            self.atoms.get_positions(), self.istep
        )

        self.results["energy"], self.results["forces"] = energy_bias, forces_bias
        self.istep += 1

        return

    def compute_bias(self, pos, istep):
        """"""
        self.plumed.cmd("setStep", istep)

        # Box for functions with PBC in plumed
        if np.any(self.atoms.pbc):
            cell = self.atoms.get_cell(complete=True).array
            self.plumed.cmd("setBox", cell)

        self.plumed.cmd("setPositions", pos)
        self.plumed.cmd("setEnergy", 0.0)
        self.plumed.cmd("setMasses", self.atoms.get_masses())
        forces_bias = np.zeros((self.atoms.get_positions()).shape)
        self.plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalc")
        energy_bias = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias)

        return [energy_bias, forces_bias]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plumed.finalize()


if __name__ == "__main__":
    ...
