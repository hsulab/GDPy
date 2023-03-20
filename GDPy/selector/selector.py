#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc 
import copy

import pathlib
from pathlib import Path
from typing import Union, List, Callable, NoReturn

import numpy as np

from ase import Atoms

from GDPy import config
from GDPy.core.node import AbstractNode
from GDPy.core.datatype import isAtomsFrames, isTrajectories
from GDPy.computation.worker.drive import DriverBasedWorker


"""Define an AbstractSelector that is the base class of any selector.
"""


class AbstractSelector(AbstractNode):

    """The base class of any selector."""

    #: Selector name.
    name: str = "abstract"

    #: Default parameters.
    default_parameters: dict = dict(
        number = [4, 0.2] # number & ratio
    )

    #: A worker for potential computations.
    worker: DriverBasedWorker = None

    #: Distinguish structures when using ComposedSelector.
    prefix: str = "selection"

    #: Output file name.
    _fname: str = "info.txt"

    logger = None #: Logger instance.

    _pfunc: Callable = print #: Function for outputs.
    indent: int = 0 #: Indent of outputs.

    #: Input data format (frames or trajectories).
    _inp_fmt: str = "stru"

    #: Output data format (frames or trajectories).
    _out_fmt: str = "stru"

    def __init__(self, directory=Path.cwd(), *args, **kwargs) -> NoReturn:
        """Create a selector.

        Args:
            directory: Working directory.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(directory=directory, *args, **kwargs)

        self.fname = self.name+"-info.txt"
        
        if "random_seed" in self.parameters:
            self.set_rng(seed=self.parameters["random_seed"])

        #: Number of parallel jobs for joblib.
        self.njobs = config.NJOBS

        return

    @AbstractNode.directory.setter
    def directory(self, directory_) -> NoReturn:
        self._directory = Path(directory_)
        self.info_fpath = self._directory/self._fname

        return 
    
    @property
    def fname(self):
        """"""
        return self._fname
    
    @fname.setter
    def fname(self, fname_):
        """"""
        self._fname = fname_
        self.info_fpath = self._directory/self._fname
        return
    
    def attach_worker(self, worker=None) -> NoReturn:
        """Attach a worker to this node."""
        self.worker = worker

        return

    def select(
        self, inp_dat: Union[List[Atoms],List[List[Atoms]]], 
        index_map: List[int]=None, ret_indices: bool=False, 
        *args, **kargs
    ) -> Union[List[Atoms],List[int]]:
        """Select frames or trajectories.

        Based on used selction protocol

        Args:
            frames: A list of ase.Atoms or a list of List[ase.Atoms].
            index_map: Global indices of frames.
            ret_indices: Whether return selected indices or frames.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            List[Atoms] or List[int]: selected results

        """
        if self.logger is not None:
            self._pfunc = self.logger.info
        self.pfunc(f"@@@{self.__class__.__name__}")

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        # - check whether frames or trajectories
        #   if trajs, flat it to frames for further selection
        _is_trajs = False
        if isAtomsFrames(inp_dat):
            frames = inp_dat
            nframes = len(inp_dat)
            mapping_indices = list(range(nframes))
            self.pfunc(f"find {nframes} nframes...")
        else:
            if isTrajectories(inp_dat):
                ntrajs = len(inp_dat)
                frames, mapping_indices = [], []
                for i, traj in enumerate(inp_dat):
                    for j, atoms in enumerate(traj):
                        # TODO: sometimes only last frame is needed
                        #       e.g. selection based on minima
                        frames.append(atoms)
                        mapping_indices.append([i,j])
                nframes = len(frames)
                self.pfunc(f"find {nframes} nframes from {ntrajs} ntrajs...")
                _is_trajs = True
            else:
                raise TypeError("Selection needs either Frames or Trajectories.")

        # - check if it is finished
        if not (self.info_fpath).exists():
            self.pfunc("run selection...")
            selected_indices = self._select_indices(frames)
        else:
            # -- restart
            self.pfunc("use cached...")
            data = np.loadtxt(self.info_fpath)
            if len(data.shape) == 1:
                data = data[np.newaxis,:]
            if not np.all(np.isnan(data.flatten())):
                data = data.tolist()
                selected_indices = [int(row[0]) for row in data]
            else:
                selected_indices = []
        self.pfunc(f"{self.name} nframes {len(frames)} -> nselected {len(selected_indices)}")

        # - add info
        for i, s in enumerate(selected_indices):
            atoms = frames[s]
            selection = atoms.info.get("selection", "")
            atoms.info["selection"] = selection+f"->{self.name}"
        
        # - save cached results for restart
        self._write_cached_results(frames, selected_indices, index_map)

        # - map indices to frames or trajectories
        #   TODO: return entire traj if minima is selected?
        #global_indices = [mapping_indices[i] for i in selected_indices]

        #print("nframes: ", len(frames), self.__class__.__name__)

        # -
        if not ret_indices:
            # -- TODO: return frames or trajs
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames # mix trajs into one frames
        else:
            # - map selected indices
            #   always List[int] for either frames or trajs
            if index_map is not None:
                selected_indices = [index_map[s] for s in selected_indices]

            return selected_indices

    @abc.abstractmethod
    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """Select structures and return selected indices."""

        return

    def _parse_selection_number(self, nframes: int) -> int:
        """Compute number of selection based on the input number.

        Args:
            nframes: Number of input frames, sometimes maybe zero.

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
    
    def pfunc(self, content, *args, **kwargs):
        """Write outputs to file."""
        content = self.indent*" " + content
        self._pfunc(content)

        return

    def _write_cached_results(self, frames: List[Atoms], selected_indices: List[int], index_map: List[int]=None, *args, **kwargs) -> NoReturn:
        """Write selection results into file that can be used for restart."""
        # - output
        data = []
        for i, s in enumerate(selected_indices):
            atoms = frames[s]
            # - gather info
            confid = atoms.info.get("confid", -1)
            step = atoms.info.get("step", -1) # step number in the trajectory
            natoms = len(atoms)
            try:
                ene = atoms.get_potential_energy()
                ae = ene / natoms
            except:
                ene, ae = np.NaN, np.NaN
            try:
                maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            except:
                maxforce = np.NaN
            score = atoms.info.get("score", np.NaN)
            if index_map is not None:
                s = index_map[s]
            data.append([s, confid, step, natoms, ene, ae, maxforce, score])

        if data:
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  %8d  %12.4f  %12.4f  %12.4f  %12.4f",
                #fmt="{:>8d}  {:>8d}  {:>8d}  {:>12.4f}  {:>12.4f}",
                header="{:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
                    *"index confid step natoms ene aene maxfrc score".split()
                ),
                footer=f"random_seed {self.random_seed}"
            )
        else:
            np.savetxt(
                self.info_fpath, [[np.NaN]*8],
                header="{:>6s}  {:>8s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
                    *"index confid step natoms ene aene maxfrc score".split()
                ),
                footer=f"random_seed {self.random_seed}"
            )

        return


if __name__ == "__main__":
    pass