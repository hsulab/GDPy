#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import NoReturn, Optional, Union, List

from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms

from .builder import StructureBuilder

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

class ReadBuilder(StructureBuilder):

    def __init__(
        self, fname, index=":", format=None, use_tags=False, 
        directory="./", random_seed=None, *args, **kwargs
    ):
        """"""
        super().__init__(use_tags=use_tags, directory=directory, random_seed=random_seed, *args, **kwargs)

        self.fname = pathlib.Path(fname)
        self.index = index
        self.format = format
        #self.kwargs = kwargs

        return
    
    def run(self, *args, **kwargs):
        """"""
        frames = read(self.fname, self.index, self.format)

        return frames
    
    def as_dict(self) -> dict:
        """"""
        params = {}
        params["method"] = "reader"
        params["fname"] = str(self.fname.resolve())
        params["index"] = self.index
        params["format"] = self.format

        return params


class DirectBuilder(StructureBuilder):
    """This generator directly returns structures that it stores.
    """

    #: Builder's name.
    name: str = "direct"

    default_parameters: dict= {

    }

    #: Stored structures.
    _frames: Optional[List[Atoms]] = None

    #: The file path of stored structures.
    _fpath: Optional[Union[str,pathlib.Path]] = None

    #: Selected structure indices.
    _indices: Union[str,List[int]] = None

    def __init__(
        self, frames: Union[str,pathlib.Path,List[Atoms]], 
        indices: Union[str,List[int]] = None, directory: Union[str,pathlib.Path]="./", 
        *args, **kwargs
    ):
        """Create a direct generator.

        Args:
            frames: The structure file path or a list of ase atoms.
            directory: Working directory.

        """
        super().__init__(directory, *args, **kwargs)

        if isinstance(frames, (str,pathlib.Path)):
            self._fpath = pathlib.Path(frames).resolve()
        else:
            assert all(isinstance(x,Atoms) for x in frames), "Input should be a list of atoms."
            self._frames = frames

        self._indices = indices

        return
    
    @property
    def fpath(self) -> Union[str,pathlib.Path]:
        """Return the file path of stored structures."""
        return self._fpath
    
    @property
    def indices(self) -> Union[str,List[int]]:
        """Return selected indices."""
        return self._indices
    
    def run(self, indices: Union[str,List[int]]=[], *args, **kwargs) -> List[Atoms]:
        """Return stored structures.

        Args:
            indices: Selected frames.

        """
        # - check indices
        if indices:
            indices_ = indices
        else:
            indices_ = self.indices

        assert (bool(self._frames) ^ bool(self.fpath)), "Cant have frames and fpath at the same time."

        # - read frames if it is a path
        if self.fpath:
            # NOTE: custom read_xsd can retrieve stored constraints
            fpath_ = str(self.fpath)
            if fpath_.endswith(".xsd"):
                frames_ = read_xsd2(fpath_)
            else:
                frames_ = read(fpath_, ":")

            # - check whether its a single Atoms object
            if isinstance(frames_, Atoms):
                frames_ = [frames_]
        else:
            if self._frames:
                frames_ = self._frames
            else:
                # NOTE: should go to here
                pass

        # - get structures
        if indices_:
            ret_frames = [frames_[i] for i in indices_]
        else:
            ret_frames = frames_
        return ret_frames
    
    def as_dict(self) -> dict:
        """Return generator parameters"""
        params = dict(
            method = "direct",
            frames = str(self._fpath.resolve()), # TODO: if not exists, and only have _frames
            indices = self.indices
        )

        return params


if __name__ == "__main__":
    ...