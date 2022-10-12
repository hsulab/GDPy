#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc 
import copy
from pathlib import Path
from typing import Union, List, Callable, NoReturn

import numpy as np

from ase import Atoms

from GDPy import config


"""Define an AbstractSelector that is the base class of any selector.
"""


class AbstractSelector(abc.ABC):

    """The base class of any selector."""

    #: Selector name.
    name: str = "abstract"

    #: Default parameters.
    default_parameters: dict = dict(
        number = [4, 0.2] # number & ratio
    )

    #: Working directory.
    _directory: Path = None

    #: Distinguish structures when using ComposedSelector.
    prefix: str = "selection"

    #: Output file name.
    _fname: str = "info.txt"

    logger = None #: Logger instance.
    _pfunc: Callable = print #: Function for outputs.
    indent: int = 0 #: Indent of outputs.

    def __init__(self, directory=Path.cwd(), *args, **kwargs) -> NoReturn:
        """Create a selector.

        Args:
            directory: Working directory.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        self.parameters = copy.deepcopy(self.default_parameters)
        for k in self.parameters:
            if k in kwargs.keys():
                self.parameters[k] = kwargs[k]

        self.directory = directory
        self.fname = self.name+"-info.txt"
        
        if "random_seed" in self.parameters:
            self.set_rng(seed=self.parameters["random_seed"])

        #: Number of parallel jobs for joblib.
        self.njobs = config.NJOBS

        return

    @property
    def directory(self) -> Path:
        """Working directory.

        Note:
            When setting directory, info_fpath would be set as well.

        """
        return self._directory
    
    @directory.setter
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

    def select(
        self, frames: List[Atoms], index_map: List[int]=None, 
        ret_indices: bool=False, *args, **kargs
    ) -> Union[List[Atoms],List[int]]:
        """Seelect frames.

        Based on used selction protocol

        Args:
            frames: A list of ase.Atoms.
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

        # - check if it is finished
        if not (self.info_fpath).exists():
            self.pfunc("run selection...")
            selected_indices = self._select_indices(frames)
        else:
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
            selection = atoms.info.get("selection","")
            atoms.info["selection"] = selection+f"->{self.name}"

        # - map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]

        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
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
    
    def as_dict(self) -> dict:
        """Return a dict of selector parameters."""
        params = dict(
            name = self.__class__.__name__
        )
        params.update(**copy.deepcopy(self.parameters))

        return params


if __name__ == "__main__":
    pass