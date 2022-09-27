#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from typing import Union, List

import numpy as np

import abc 
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from GDPy import config

""" Various Selection Protocols
"""


class AbstractSelector(abc.ABC):

    default_parameters = dict(
        number = [4, 0.2] # number & ratio
    )

    prefix = "structure"
    _directory = None

    logger = None
    pfunc = print

    def __init__(self, directory=Path.cwd(), *args, **kwargs) -> None:
        """"""
        self.directory = directory
        
        self.parameters = copy.deepcopy(self.default_parameters)
        for k in self.parameters:
            if k in kwargs.keys():
                self.parameters[k] = kwargs[k]
        
        if "random_seed" in self.parameters:
            self.set_rng(seed=self.parameters["random_seed"])

        self.njobs = config.NJOBS

        return

    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, directory_):
        self._directory = Path(directory_)
        return 
    
    def set(self, *args, **kwargs):
        """"""
        for k, v in kwargs.items():
            if k in self.parameters:
                self.parameters[k] = v

        return

    def __getattr__(self, key):
        """ Corresponding getattribute-function 
        """
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]
        return object.__getattribute__(self, key)

    def set_rng(self, seed=None):
        """"""
        # - assign random seeds
        if seed is None:
            self.rng = np.random.default_rng()
        elif isinstance(seed, int):
            self.rng = np.random.default_rng(seed)

        return

    @abc.abstractmethod
    def select(self, index_map=None, ret_indices: bool=False, *args, **kargs):
        """"""
        if self.logger is not None:
            self.pfunc = self.logger.info
        self.pfunc(f"@@@{self.__class__.__name__}")

        # - check if finished

        return

    def _parse_selection_number(self, nframes):
        """ nframes - number of frames
            sometimes maybe zero
        """
        default_number, default_ratio = self.default_parameters["number"]
        number_info = self.parameters["number"]
        if isinstance(number_info, int):
            num_fixed, num_percent = number_info, default_ratio
        elif isinstance(number_info, float):
            num_fixed, num_percent = default_number, number_info
        else:
            assert len(number_info) == 2, "Cant parse number for selection..."
            num_fixed, num_percent = number_info
        
        if num_fixed is not None:
            if num_fixed > nframes:
                num_fixed = int(nframes*num_percent)
        else:
            num_fixed = int(nframes*num_percent)

        return num_fixed
    
    def as_dict(self):
        """"""
        params = dict(
            name = self.__class__.__name__
        )
        params.update(**copy.deepcopy(self.parameters))

        return params

class ComposedSelector(AbstractSelector):
    
    """ perform a list of selections on input frames
    """

    name = None
    verbose = True

    def __init__(self, selectors, directory=Path.cwd()):
        """"""
        self.selectors = selectors
        self._directory = directory

        self._check_convergence()

        # - set namd and directory
        self.name = "-".join([s.name for s in self.selectors])

        return
    
    def _check_convergence(self):
        """ check if there is a convergence selector
            if so, selections will be performed on converged ones and 
            others separately
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
        """"""
        self._directory = directory_

        for s in self.selectors:
            s.directory = self._directory

        return
    
    def select(self, frames, index_map=None, ret_indces: bool=False, *args, **kwargs):
        """"""
        super().select(*args, **kwargs)
        # - initial index stuff
        nframes = len(frames)
        cur_index_map = list(range(nframes))

        # NOTE: find converged ones and others
        # convergence selector is standalone
        frame_index_groups = {}
        if self.conv_selection is not None:
            selector = self.conv_selection
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
                self.pfunc(f"  nconverged: {len(converged_frames)}")
        else:
            frame_index_groups = dict(
                traj = cur_index_map
            )

        # - run selectors
        merged_final_indices = []

        for sys_name, global_indices in frame_index_groups.items():
            # - init 
            metadata = [] # selected indices
            final_indices = []

            # - prepare index map
            cur_frames = [frames[i] for i in global_indices]
            cur_index_map = global_indices.copy()

            metadata.append(cur_index_map)

            self.pfunc(f"--- start subgroup {sys_name} ---")
            ncandidates = len(cur_frames)
            self.pfunc(f"  ncandidates: {ncandidates}")
            for isele, selector in enumerate(self.selectors):
                # TODO: add info to selected frames
                # TODO: use cached files
                if ncandidates <= 0:
                    self.pfunc("  skip this subgroup...")
                    break
                self.pfunc(f"  @{selector.name}-s{isele}")
                ncandidates = len(cur_frames)
                cur_indices = selector.select(cur_frames, ret_indices=True)
                mapped_indices = sorted([cur_index_map[x] for x in cur_indices])
                metadata.append(mapped_indices)
                final_indices = mapped_indices.copy()
                cur_frames = [frames[x] for x in mapped_indices]
                nselected = len(cur_frames)
                self.pfunc(f"  ncandidates: {ncandidates} nselected: {nselected}")
                # - create index_map for next use
                cur_index_map = mapped_indices.copy()

                # TODO: should print-out intermediate results?
                write(self.directory/f"{sys_name}-{selector.name}-selection-{isele}.xyz", cur_frames)
            
            merged_final_indices += final_indices

            # TODO: map selected indices
            # is is ok?
            if index_map is not None:
                final_indices = [index_map[s] for s in final_indices]

            # - ouput data
            # TODO: write selector parameters to metadata file
            maxlength = np.max([len(m) for m in metadata])
            data = -np.ones((maxlength,len(metadata)))
            for i, m in enumerate(metadata):
                data[:len(m),i] = m
            header = "".join(["init "+"{:<24s}".format(s.name) for s in self.selectors])
        
            np.savetxt(self.directory/(f"{sys_name}-{self.name}_metadata.txt"), data, fmt="%24d", header=header)

        if ret_indces:
            return merged_final_indices
        else:
            selected_frames = [frames[i] for i in merged_final_indices]
            write(self.directory/f"merged-{selector.name}-selection-final.xyz", selected_frames)

            return selected_frames

class ConvergenceSelector(AbstractSelector):

    """ find geometrically converged frames
    """

    name = "convergence"

    def __init__(self, fmax=0.05, directory=Path.cwd()):
        """"""
        self.fmax = fmax # eV

        self.directory = directory

        return
    
    def select(self, frames, index_map = None, ret_indices: bool=False, *args, **kwargs):
        """"""
        super().select(*args, **kwargs)
        # NOTE: input atoms should have constraints attached
        selected_indices = []
        for i, atoms in enumerate(frames):
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            if maxforce < self.fmax:
                selected_indices.append(i)

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]

        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
            return selected_indices


if __name__ == "__main__":
    pass