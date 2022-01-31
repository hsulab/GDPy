#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from re import A
import numpy as np

from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError

"""
output files by lasp NVE-MD
    allfor.arc  allkeys.log  allstr.arc  firststep.restart
    md.arc  md.restart  vel.arc
"""


class LaspDynamics:

    def __init__(self):

        return


class LaspNN(FileIOCalculator):

    name = "LaspNN"
    implemented_properties = ["energy", "forces"]
    # implemented_propertoes = ["energy", "forces", "stress"]
    command = "lasp"

    def __init__(self, *args, label="LASP", **kwargs):
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

        self._directory_path = Path(self.directory)

        return
    
    def calculate(self, *args, **kwargs):

        FileIOCalculator.calculate(self, *args, **kwargs)

        return
    
    def write_input(self, atoms, properties=None, system_changes=None):
        # create calc dir
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # structure
        write(
            self._directory_path / "lasp.str", atoms, format="dmol-arc",
            parallel=False
        )

        # check symbols and corresponding potential file
        atomic_types = set(self.atoms.get_chemical_symbols()) # TODO: sort by 

        # input
        content  = "potential NN\n"
        content += "%block netinfo\n"
        #content += "  O  ZnCrO.pot\n"
        #content += "  Cr ZnCrO.pot\n"
        #content += "  Zn ZnCrO.pot\n"
        for atype in atomic_types:
            # write path
            pot_path = Path(self.parameters["pot"][atype]).resolve()
            content += "  {:<4s} {:s}\n".format(atype, pot_path.name)
            # creat potential link
            pot_link = self._directory_path / pot_path.name
            if not pot_link.exists():
                pot_link.symlink_to(pot_path)

        content += "%endblock netinfo\n"

        content += "explore_type nve\n"
        content += "MD.dt           1.0\n"
        content += "MD.ttotal       0\n"
        content += "MD.target_T     300\n"
        content += "MD.realmass     .true.\n"
        content += "MD.print_freq      10\n"

        with open(self._directory_path / "lasp.in", "w") as fopen:
            fopen.write(content)


        return
    
    def read_results(self):
        """read LASP results"""
        natoms = len(self.atoms)

        with open(self._directory_path / "allfor.arc", "r") as fopen:
            lines = fopen.readlines()
        
        energy = float(lines[0].split()[3])
        stress = np.array(lines[1].split()) # TODO: what is the format of stress

        forces = []
        for line in lines[2:2+natoms]:
            forces.append(line.split())
        forces = np.array(forces, dtype=float)

        self.results["energy"] = energy
        self.results["forces"] = forces

        return


if __name__ == "__main__":
    atoms = read("/mnt/scratch2/users/40247882/catsign/lasp-main/xxx.xyz")
    pot = dict(
        O  = "./ZnCrO.pot",
        Cr = "./ZnCrO.pot",
        Zn = "./ZnCrO.pot"
    )
    calc = LaspNN(
        directory = "./LaspNN-Worker",
        command = "mpirun -n 4 lasp",
        pot=pot
    )

    atoms.calc = calc
    print("initial energy: ", atoms.get_potential_energy())
    # print(atoms.get_forces())

    from GDPy.calculator.ase_interface import AseDynamics
    worker = AseDynamics(calc, directory=calc.directory)
    #worker.run(atoms, fmax=0.05, steps=10)
    _ = worker.minimise(atoms, fmax=0.05, steps=10)
    print(_)
    print("opt energy: ", atoms.get_potential_energy())