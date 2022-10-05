#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import Union, List


from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms

from GDPy.builder.builder import StructureGenerator

def read_xsd2(fd) -> Atoms:
    """read xsd file by Material Studio

    NOTE:
        support reading constraints...

    """
    import xml.etree.ElementTree as ET
    from xml.dom import minidom 

    tree = ET.parse(fd)
    root = tree.getroot()

    atomtreeroot = root.find("AtomisticTreeRoot")
    # if periodic system
    if atomtreeroot.find("SymmetrySystem") is not None:
        symmetrysystem = atomtreeroot.find("SymmetrySystem")
        mappingset = symmetrysystem.find("MappingSet")
        mappingfamily = mappingset.find("MappingFamily")
        system = mappingfamily.find("IdentityMapping")

        coords = list()
        cell = list()
        formula = str()

        names = list()
        restrictions = list() 

        for atom in system:
            if atom.tag == "Atom3d":
                symbol = atom.get("Components")
                formula += symbol

                xyz = atom.get("XYZ")
                if xyz:
                    coord = [float(coord) for coord in xyz.split(",")]
                else:
                    coord = [0.0, 0.0, 0.0]
                coords.append(coord)

                name = atom.get("Name") 
                if name:
                    pass # find name 
                else: 
                    name = symbol + str(len(names)+1) # None due to copy atom 
                names.append(name)

                restriction = atom.get("RestrictedProperties", None)
                if restriction:
                    if restriction.startswith("FractionalXYZ"):  # TODO: may have 1-3 flags
                        restrictions.append(True)
                    else: 
                        raise ValueError("unknown RestrictedProperties")
                else: 
                    restrictions.append(False)
            elif atom.tag == "SpaceGroup":
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
    elif atomtreeroot.find("Molecule") is not None:
        system = atomtreeroot.find("Molecule")

        coords = list()
        formula = str()

        for atom in system:
            if atom.tag == "Atom3d":
                symbol = atom.get("Components")
                formula += symbol

                xyz = atom.get("XYZ")
                coord = [float(coord) for coord in xyz.split(",")]
                coords.append(coord)

        atoms = Atoms(formula, pbc=False)
        atoms.set_scaled_positions(coords)
        return atoms


class DirectGenerator(StructureGenerator):
    """This generator directly returns structures that it stores.
    """

    #: stored structures.
    _frames: List[Atoms] = []

    def __init__(
        self, frames: Union[str,pathlib.Path,List[Atoms]], 
        directory: Union[str,pathlib.Path]="./", *args, **kwargs
    ):
        """Create a direct generator.

        Args:
            frames: The structure file path or a list of ase atoms.
            directory: Working directory.

        """
        super().__init__(directory, *args, **kwargs)

        frames_ = frames
        if isinstance(frames_, (str,pathlib.Path)):
            if frames_.endswith(".xsd"):
                frames_ = read_xsd2(frames_)
            else:
                frames_ = read(frames_, ":")
        # check whether its a single Atoms
        if isinstance(frames_, Atoms):
            frames_ = [frames_]

        self._frames = frames_

        return
    
    @property
    def frames(self) -> List[Atoms]:
        """Return stored structures."""
        return self._frames
    
    def run(self, *args, **kwargs) -> List[Atoms]:
        """Return stored structures."""
        return self.frames


if __name__ == "__main__":
    pass