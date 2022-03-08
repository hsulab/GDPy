#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np

import xml.etree.ElementTree as ET
from xml.dom import minidom 

from ase import Atoms 
from ase.io import read, write
from ase.constraints import FixAtoms

from pathlib import Path

from GDPy.calculator.vasp import VaspMachine
from GDPy.utils.command import run_command

def read_xsd2(fd):
    """ read xsd file by Material Studio
    """
    tree = ET.parse(fd)
    root = tree.getroot()

    atomtreeroot = root.find('AtomisticTreeRoot')
    # if periodic system
    if atomtreeroot.find('SymmetrySystem') is not None:
        symmetrysystem = atomtreeroot.find('SymmetrySystem')
        mappingset = symmetrysystem.find('MappingSet')
        mappingfamily = mappingset.find('MappingFamily')
        system = mappingfamily.find('IdentityMapping')

        coords = list()
        cell = list()
        formula = str()

        names = list()
        restrictions = list() 

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                if xyz:
                    coord = [float(coord) for coord in xyz.split(',')]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)

                name = atom.get('Name') 
                if name:
                    pass # find name 
                else: 
                    name = symbol + str(len(names)+1) # None due to copy atom 
                names.append(name)

                restriction = atom.get('RestrictedProperties', None)
                if restriction:
                    if restriction.startswith("FractionalXYZ"):  # TODO: may have 1-3 flags
                        restrictions.append(True)
                    else: 
                        raise ValueError('unknown RestrictedProperties')
                else: 
                    restrictions.append(False)
            elif atom.tag == 'SpaceGroup':
                avec = [float(vec) for vec in atom.get('AVector').split(',')]
                bvec = [float(vec) for vec in atom.get('BVector').split(',')]
                cvec = [float(vec) for vec in atom.get('CVector').split(',')]

                cell.append(avec)
                cell.append(bvec)
                cell.append(cvec)

        atoms = Atoms(formula, cell=cell, pbc=True)
        atoms.set_scaled_positions(coords)

        # add constraints 
        fixed_indices = [idx for idx, val in enumerate(restrictions) if val]
        if fixed_indices:
            atoms.set_constraint(FixAtoms(indices=fixed_indices))

        # add two atoms constrained optimisation 
        constrained_indices = [
            idx for idx, name in enumerate(names) if name.endswith('_c') 
        ]
        if constrained_indices:
            assert len(constrained_indices) == 2
            atoms.info["copt"] = constrained_indices

        return atoms
        # if non-periodic system
    elif atomtreeroot.find('Molecule') is not None:
        system = atomtreeroot.find('Molecule')

        coords = list()
        formula = str()

        for atom in system:
            if atom.tag == 'Atom3d':
                symbol = atom.get('Components')
                formula += symbol

                xyz = atom.get('XYZ')
                coord = [float(coord) for coord in xyz.split(',')]
                coords.append(coord)

        atoms = Atoms(formula, pbc=False)
        atoms.set_scaled_positions(coords)
        return atoms


def vasp_main(
    pstru, # path, a single file or a directory with many files
    choice,
    incar_template,
    cinidices,
    is_sort,
    is_submit
):
    """"""
    # parse structures
    pstru = Path(pstru)

    # vasp machine
    gdpconfig = Path.home() / ".gdp"
    if gdpconfig.exists() and gdpconfig.is_dir():
        # find vasp config
        vasprc = gdpconfig / "vasprc.json"
        with open(vasprc, "r") as fopen:
            input_dict = json.load(fopen)
    else:
        input_dict = {}
    input_dict["incar"] = incar_template
    input_dict["isinteractive"] = True


    if choice == "create":
        stru_files = []
        if pstru.is_file():
            stru_files.append(pstru)
        elif pstru.is_dir():
            for p in pstru.glob("*.xsd"): # TODO: change this
                stru_files.append(p)
            stru_files.sort()

        vasp_machine = VaspMachine(**input_dict)
    
        # create calculation files
        cwd = Path.cwd()
        for struct_path in stru_files:
            if cwd != Path.cwd(): 
                directory = cwd.resolve() / struct_path.stem
            else:
                directory = Path.cwd().resolve() / struct_path.stem

            if directory.exists(): 
                raise ValueError(f"{directory} exists.")

            atoms = read_xsd2(struct_path)
            #atoms.set_constraint(FixAtoms(indices=range(len(atoms))))

            # sort atoms by symbols and z-positions especially for supercells 
            if is_sort: 
                numbers = atoms.numbers 
                xposes = atoms.positions[:, 0].tolist()
                yposes = atoms.positions[:, 1].tolist()
                zposes = atoms.positions[:, 2].tolist()
                sorted_indices = np.lexsort((xposes,yposes,zposes,numbers))
                atoms = atoms[sorted_indices]

                map_indices = dict(zip(sorted_indices, range(len(atoms))))
                copt = atoms.info.get("copt", None)
                if copt: 
                    new_copt = [map_indices[key] for key in atoms.info["copt"]]
                    atoms.info["copt"] = new_copt 

            vasp_machine.create(atoms, directory)

    elif choice == "freq":
        assert pstru.is_dir(), "input path is not a directory"
        vasprun = pstru / "vasprun.xml"
        if not vasprun.exists():
            raise RuntimeError("vasp calculation may not exist...")
        
        input_dict["incar"] = pstru / "INCAR"

        vasp_machine = VaspMachine(**input_dict)

        frames = read(vasprun, ":")
        atoms = frames[-1]

        assert len(cinidices) > 0, "at least one atom should be free for freq..."

        indices = [i for i in range(len(atoms)) if i not in cinidices]
        cons = FixAtoms(indices=indices)
        atoms.set_constraint(cons)

        directory = Path.cwd().resolve() / (pstru.name + "-freq")
        vasp_machine.create(atoms, directory, task="freq")


    # submit job automatically 
    if is_submit: 
        vasp_machine.submit(directory)

    return

if __name__ == "__main__":
    pass