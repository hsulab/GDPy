#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
from typing import List

from ase import Atoms

from GDPy.core.node import AbstractNode
from GDPy.builder import create_generators

class SystemToExplore(AbstractNode):

    """An object that helps the exploration of the system.
    """

    _curr_structures: List[Atoms] = None
    _curr_composition: dict = None
    _curr_builder_index = 0

    _all_structures: List[List[Atoms]] = []

    def __init__(
        self, generators: List[dict], prefix=None, composition=None, 
        directory="./", *args, **kwargs
    ):
        """"""
        super().__init__(directory=directory)

        builders_ = []
        for params in generators:
            builder = create_generators(params)
            builders_.extend(builder)
        self.builders = builders_

        for i, builder in enumerate(self.builders):
            builder.directory = self.directory/f"b{i}"

        return

    def prefix(self):

        return

    @AbstractNode.directory.setter 
    def directory(self, directory_) -> pathlib.Path:
        self._directory = pathlib.Path(directory_)
        if hasattr(self, "builders"):
            for i, builder in enumerate(self.builders):
                builder.directory = self.directory/f"b{i}"

        return
    
    def get_structures(self):
        """"""
        if self._curr_structures is None:
            # - the first builder needs generate structures
            curr_builder = self.builders[0]
            curr_frames = curr_builder.run() # TODO: run params: number?
            self._curr_structures = curr_frames
            self._curr_builder_index += 1
        else:
            curr_builder = self.builders[self._curr_builder_index]
            if True: # update parameters to _curr_structures
                curr_builder.substrates = self._curr_structures # TODO: ...
                ...
            curr_frames = curr_builder.run() # TODO: run params: number?
            self._curr_structures = curr_frames
            self._curr_builder_index += 1

        return self._curr_structures
    
    def update_structures(self, input_frames_: List[Atoms]):
        """Update current structures with input ones.

        This helps when extra modifications are applied.
        
        """
        self._curr_structures = input_frames_

        return
    
    def get_composition(self):
        """Get the composition of this system."""

        return
    
    def get_specific_parameters(self):
        """Specific parameters for the system.
        
        Some for geometric minimisation and some for ab-initio calculations.

        """

        return


if __name__ == "__main__":
    import yaml
    p = "/mnt/scratch2/users/40247882/pbe-oxides/eann-main/r12/expeditions/CO+O/vads/sys.yaml"

    with open(p, "r") as fopen:
        data = yaml.safe_load(fopen)
    print(data)

    sys = SystemToExplore(**data["fcc111s22_1CO+1O"])
    sys.directory = "./xxx"

    frames = sys.get_structures()
    nframes = len(frames)
    print(nframes)
    
    sys.update_structures(frames[:3])

    frames = sys.get_structures()
    nframes = len(frames)
    print(nframes)

    ...