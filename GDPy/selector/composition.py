#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import Union, List, NoReturn

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.selector.selector import AbstractSelector


class ComposedSelector(AbstractSelector):
    
    """Perform several selections consecutively.
    """

    name = "composed"

    default_parameters = dict(
        selectors = []
    )

    def __init__(self, selectors: List[AbstractSelector], directory="./", *args, **kwargs):
        """"""
        super().__init__(directory, *args, **kwargs)

        # - re-init selectors and their directories
        self.selectors = selectors
        self.directory = directory
        #self.name = "-".join([s.name for s in self.selectors])
        #self.fname = self.name+"-info.txt"

        self._check_convergence()

        return
    
    def _check_convergence(self) -> NoReturn:
        """Check if there is a convergence selector.

        If it has, selections will be performed on converged ones and 
        others separately.

        """
        conv_i = None
        selectors_ = self.selectors
        for i, s in enumerate(selectors_):
            if s.name == "convergence":
                conv_i = i
                break
        
        if conv_i is not None:
            self.conv_selection = selectors_.pop(conv_i)
            self.selectors = selectors_
        else:
            self.conv_selection = None

        return
    
    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, directory_: Union[str,Path]):
        self._directory = directory_
        self.info_fpath = self._directory/self._fname
        for s in self.selectors:
            s.directory = self._directory

        return
    
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """Return selected indices."""
        # - initial index stuff
        nframes = len(frames)
        cur_index_map = list(range(nframes))

        # NOTE: find converged ones and others
        # convergence selector is standalone
        frame_index_groups = {}
        if self.conv_selection is not None:
            selector = self.conv_selection
            selector.indent = 2
            selector.directory = self.directory
            converged_indices = selector.select(frames, ret_indices=True)
            traj_indices = [m for m in cur_index_map if m not in converged_indices] # means not converged
            frame_index_groups = dict(
                converged = converged_indices,
                traj = traj_indices
            )
            # -- converged
            converged_frames = [frames[x] for x in converged_indices]
            if converged_frames:
                write(self.directory/f"{selector.name}-frames.xyz", converged_frames)
        else:
            frame_index_groups = dict(
                traj = cur_index_map
            )

        # - run selectors
        merged_final_indices = []

        for sys_name, global_indices in frame_index_groups.items():
            # - init 
            metadata = [] # selected indices

            # - prepare index map
            cur_frames = [frames[i] for i in global_indices]
            cur_index_map = global_indices.copy()

            metadata.append(cur_index_map)

            self.pfunc(f"@@@ start subgroup {sys_name} @@@")
            ncandidates = len(cur_frames)
            self.pfunc(f"  ncandidates: {ncandidates}")
            for isele, selector in enumerate(self.selectors):
                self.pfunc(f"  @{selector.name}-s{isele}")
                selector.fname = f"{sys_name}-{selector.name}-info-s{isele}.txt"
                selector.indent = 4
                # - map indices
                cur_indices = selector.select(cur_frames, index_map=cur_index_map, ret_indices=True)
                # - create index_map for next use
                metadata.append(cur_indices)
                cur_frames = [frames[x] for x in cur_indices]
                cur_index_map = cur_indices.copy()

                # TODO: should print-out intermediate results?
                #write(self.directory/f"{sys_name}-{selector.name}-selection-{isele}.xyz", cur_frames)
            
            merged_final_indices += copy.deepcopy(cur_indices)

            # - ouput data
            # TODO: write selector parameters to metadata file
            maxlength = np.max([len(m) for m in metadata])
            data = -np.ones((maxlength,len(metadata)))
            for i, m in enumerate(metadata):
                data[:len(m),i] = m
            # - flip
            data = data[:,::-1]
            header = "".join(["init "+"{:<24s}".format(s.name) for s in self.selectors[::-1]])
        
            np.savetxt((self.directory/(sys_name+"-"+self.fname)), data, fmt="%24d", header=header)

        return merged_final_indices


if __name__ == "__main__":
    pass