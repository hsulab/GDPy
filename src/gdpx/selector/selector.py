#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc 
import copy
import itertools
import pathlib
from typing import Union, List, Callable, NoReturn, Optional

import numpy as np

from ase import Atoms

from ..core.node import AbstractNode
from ..worker.drive import DriverBasedWorker
from ..data.array import AtomsNDArray


"""Define an AbstractSelector that is the base class of any selector.
"""

def save_cache(fpath, data, random_seed: int=None):
    """"""
    header = ("#{:>11s}  {:>8s}  {:>8s}  {:>8s}  "+"{:>12s}"*4+"\n").format(
        *"index confid step natoms ene aene maxfrc score".split()
    )
    footer = f"random_seed {random_seed}"

    content = header
    for x in data:
        content += ("{:>12s}  {:>8d}  {:>8d}  {:>8d}  "+"{:>12.4f}"*4+"\n").format(*x)
    content += footer

    with open(fpath, "w") as fopen:
        fopen.write(content)

    return

def load_cache(fpath, random_seed: int=None):
    """"""
    with open(fpath, "r") as fopen:
        lines = fopen.readlines()

    # - header
    header = lines[0]

    # - data
    data = lines[1:-1] # TODO: test empty data

    raw_markers = []
    if data:
        # NOTE: new_markers looks like [(0,1),(0,2),(1,0)]
        #       If no structures are selected, the info file should only contain
        #       the header and the footer
        new_markers =[
            [int(x) for x in (d.strip().split()[0]).split(",")] for d in data
        ]
        #new_markers = []
        #for d in data:
        #    curr_marker = []
        #    for x in d.strip().split()[0].split(","):
        #        if x.isdigit(): # MUST BE AN INTEGER
        #            curr_marker.append(int(x))
        #        else:
        #            curr_marker.append(np.nan)
        #    new_markers.append(curr_marker)
        raw_markers = new_markers

    # - footer
    footer = lines[-1]
    cache_random_seed = int(footer.strip().split()[-1]) # TODO: random state
    #assert cache_random_seed == random_seed

    return raw_markers

def group_markers(new_markers_unsorted):
    """"""
    new_markers = sorted(new_markers_unsorted, key=lambda x: x[0])
    raw_markers_unsorted = []
    for k, v in itertools.groupby(new_markers, key=lambda x: x[0]):
        raw_markers_unsorted.append([k,[x[1] for x in v]])

    # traj markers are sorted when set
    raw_markers = [[x[0],sorted(x[1])] for x in sorted(raw_markers_unsorted, key=lambda x:x[0])]

    return raw_markers


class AbstractSelector(AbstractNode):

    """The base class of any selector."""

    #: Selector name.
    name: str = "abstract"

    #: Target axis to select.
    axis: Optional[int] = None

    #: Default parameters.
    default_parameters: dict = dict(
        number = [4, 0.2], # number & ratio
        verbose = False
    )

    #: A worker for potential computations.
    worker: DriverBasedWorker = None

    #: Distinguish structures when using ComposedSelector.
    prefix: str = "selection"

    #: Output file name.
    _fname: str = "info.txt"

    #: Output data format (frames or trajectories).
    _out_fmt: str = "stru"

    def __init__(self, directory="./", axis=None, *args, **kwargs) -> None:
        """Create a selector.

        Args:
            directory: Working directory.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        super().__init__(directory=directory, *args, **kwargs)

        self.axis = axis

        self.fname = self.name+"-info.txt"
        
        # - update parameters from kwargs
        self.parameters = copy.deepcopy(self.default_parameters)
        for k in self.parameters:
            if k in kwargs.keys():
                self.parameters[k] = kwargs[k]

        return

    def set(self, *args, **kwargs):
        """Set parameters."""
        for k, v in kwargs.items():
            if k in self.parameters:
                self.parameters[k] = v

        return

    def __getattr__(self, key):
        """Corresponding getattribute-function."""
        if key != "parameters" and key in self.parameters:
            return self.parameters[key]

        return object.__getattribute__(self, key)

    @AbstractNode.directory.setter
    def directory(self, directory_) -> NoReturn:
        self._directory = pathlib.Path(directory_)
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

    def select(self, inp_dat: AtomsNDArray, *args, **kargs) -> List[Atoms]:
        """Select trajectories.

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
        self._print(f"@@@{self.__class__.__name__}")

        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        # NOTE: input structures should always be the AtomsNDArray type
        # TODO: if inp_dat is already a AtomsNDArray, a new object is created,
        #       which makes markers immutable... Need fix this!!
        if isinstance(inp_dat, AtomsNDArray):
            ...
        else:
            inp_dat = AtomsNDArray(inp_dat)

        frames = inp_dat
        inp_nframes = len(frames.markers)

        # - check if it is finished
        if not (self.info_fpath).exists():
            # -- mark structures
            self._print("run selection...")
            self._mark_structures(frames)
            # -- save cached results for restart
            self._write_cached_results(frames)
        else:
            # -- restart
            self._print("use cached...")
            raw_markers = load_cache(self.info_fpath)
            #print(f"raw_markers: {raw_markers}")
            frames.markers = raw_markers
        
        out_nframes = len(frames.markers)
        self._print(f"{self.name} nstructures {inp_nframes} -> nselected {out_nframes}")

        # - add history
        #   get_marked_structures return reference to Atoms objects
        #for a in inp_dat.get_marked_structures():
        #    print(a.info.get("selection"))

        marked_structures = inp_dat.get_marked_structures()
        for atoms in marked_structures:
            selection = atoms.info.get("selection", "")
            atoms.info["selection"] = selection+f"->{self.name}"
        
        #for a in inp_dat.get_marked_structures():
        #    print(a.info["selection"])

        return marked_structures
    
    @abc.abstractmethod
    def _mark_structures(self, data, *args, **kwargs) -> None:
        """Mark structures subject to selector's conditions."""

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

    def _write_cached_results(self, aa: AtomsNDArray, *args, **kwargs) -> None:
        """Write selection results into file that can be used for restart."""
        # - 
        markers = aa.markers

        # - output
        data = []
        for ind in markers:
            atoms = aa[tuple(ind)]
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
            score = atoms.info.get("score", np.nan)
            # - add info
            ind_str = ",".join([str(x) for x in ind])
            data.append([f"{ind_str}", confid, step, natoms, ene, ae, maxforce, score])
        
        if data:
            save_cache(self.info_fpath, data, self.random_seed)
        else:
            #np.savetxt(
            #    self.info_fpath, [[np.NaN]*8],
            #    header="{:>11s}  {:>8s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
            #        *"index confid step natoms ene aene maxfrc score".split()
            #    ),
            #    footer=f"random_seed {self.random_seed}"
            #)
            content ="{:>11s}  {:>8s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}  {:>12s}".format(
                *"index confid step natoms ene aene maxfrc score".split()
            )
            content += f"random_seed {self.random_seed}"
            with open(self.info_fpath, "w") as fopen:
                fopen.write(content)

        return


if __name__ == "__main__":
    ...
