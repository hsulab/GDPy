#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import pathlib
from typing import NoReturn, List
import warnings

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.data.dataset import XyzDataloader
from GDPy.data.array import AtomsArray2D

@registers.operation.register
class end_session(Operation):

    def __init__(self, *args) -> NoReturn:
        super().__init__(args)
    
    def forward(self, *args):
        """"""
        return super().forward()

@registers.operation.register
class chain(Operation):

    """Merge arbitrary nodes' outputs into one list.
    """

    status = "finished" # Always finished since it is not time-consuming

    def __init__(self, nodes, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(nodes)

        # - some operation parameters

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()
        print("chain outputs: ", outputs)

        return list(itertools.chain(*outputs))

@registers.operation.register
class map(Operation):

    """Give each input node a name and construct a dict.

    This is useful when validating structures from different systems.

    """

    status = "finished"

    def __init__(self, nodes, names, directory="./") -> NoReturn:
        """"""
        super().__init__(nodes, directory)

        assert len(nodes) == len(names), "Numbers of nodes and names are inconsistent."
        self.names = names

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()

        ret = {}
        for k, v in zip(self.names, outputs):
            ret[k] = v
        
        print("map ret: ", ret)

        return ret


@registers.operation.register
class transfer(Operation):

    """Transfer worker results to target destination.
    """

    def __init__(self, structure, target_dir, version, system="mixed", directory="./") -> NoReturn:
        """"""
        input_nodes = [structure]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.target_dir = pathlib.Path(target_dir).resolve()
        self.version = version

        self.system = system # molecule/cluster, surface, bulk

        return
    
    def forward(self, frames: List[Atoms]):
        """"""
        super().forward()

        if isinstance(frames, AtomsArray2D):
            frames = frames.get_marked_structures()

        self.pfunc(f"target dir: {str(self.target_dir)}")

        # - check chemical symbols
        system_dict = {} # {formula: [indices]}

        formulae = [a.get_chemical_formula() for a in frames]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            system_dict[k] = [x[0] for x in v]
        
        # - transfer data
        for formula, curr_indices in system_dict.items():
            # -- TODO: check system type
            system_type = self.system # currently, use user input one
            # -- name = description+formula+system_type
            dirname = "-".join([self.directory.parent.name, formula, system_type])
            target_subdir = self.target_dir/dirname
            target_subdir.mkdir(parents=True, exist_ok=True)

            # -- save frames
            curr_frames = [frames[i] for i in curr_indices]
            curr_nframes = len(curr_frames)

            strname = self.version + ".xyz"
            target_destination = self.target_dir/dirname/strname
            if not target_destination.exists():
                write(target_destination, curr_frames)
                self.pfunc(f"nframes {curr_nframes} -> {target_destination.name}")
            else:
                warnings.warn(f"{target_destination} exists.", UserWarning)
        
        dataset = XyzDataloader(self.target_dir)
        self.status = "finished"

        return dataset

if __name__ == "__main__":
    ...