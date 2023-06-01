#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import NoReturn, List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.geometry import find_mic

from GDPy.core.operation import Operation
from GDPy.computation.worker.worker import AbstractWorker
from GDPy.computation.worker.drive import DriverBasedWorker
from GDPy.potential.register import create_potter
from GDPy.selector.interface import create_selector
from GDPy.utils.command import parse_input_file
from GDPy.utils.command import CustomTimer

DEFAULT_MAIN_DIRNAME = "MyWorker"


class drive(Operation):

    """Drive structures.
    """

    def __init__(
        self, frames, worker, 
        read_traj=True, traj_period: int=1,
    ):
        """"""
        super().__init__([frames])

        self.worker = worker

        self.read_traj = read_traj
        self.traj_period = traj_period

        return
    
    @Operation.directory.setter
    def directory(self, directory_) -> NoReturn:
        """"""
        super(drive, drive).directory.__set__(self, directory_)

        self.worker.directory = self._directory

        return
    
    def forward(self, frames):
        """"""
        super().forward()

        new_frames = []

        cached_fpath = self.directory/"output.xyz"
        if not cached_fpath.exists():
            self.worker.run(frames)
            self.worker.inspect(resubmit=True)
            # TODO: check whether finished this operation...
            if self.worker.get_number_of_running_jobs() == 0:
                trajectories = self.worker.retrieve(
                    read_traj = self.read_traj, traj_period = self.traj_period,
                    include_first = True, include_last = True,
                    ignore_retrieved=False # TODO: ignore misleads...
                )
                for traj in trajectories:
                    new_frames.append(traj[-1])
                write(cached_fpath, new_frames)
        else:
            new_frames = read(cached_fpath, ":")

        return new_frames

class label(Operation):

    """Drive structures.
    """

    def __init__(
        self, nodes: list, worker, 
        read_traj=True, traj_period: int=1,
    ):
        """"""
        super().__init__(nodes)

        self.worker = worker

        self.read_traj = read_traj
        self.traj_period = traj_period

        return
    
    @Operation.directory.setter
    def directory(self, directory_) -> NoReturn:
        """"""
        super(drive, drive).directory.__set__(self, directory_)

        self.worker.directory = self._directory

        return
    
    def forward(self, *args):
        """"""
        super().forward()

        for node, frames in zip(self.input_nodes, args):
            _ = self._irun(self.directory/node.directory.name, frames)

        return
    
    def _irun(self, sub_wdir, sub_frames: List[Atoms]):
        """"""
        cached_fpath = sub_wdir/"output.xyz"
        if not cached_fpath.exists():
            self.worker = sub_wdir
            self.worker.run(sub_frames)
            self.worker.inspect(resubmit=True)
            trajectories = self.worker.retrieve(
                read_traj = self.read_traj, traj_period = self.traj_period,
                include_first = True, include_last = True,
                ignore_retrieved=False # TODO: ignore misleads...
            )
            new_frames = []
            for traj in trajectories:
                new_frames.append(traj[-1])
            write(cached_fpath, new_frames)
        else:
            new_frames = read(cached_fpath, ":")

        return new_frames

class extract_trajectories(Operation):

    """Extract dynamics trajectories from a drive-node's worker.
    """

    def __init__(
        self, drive_node, read_traj=True, traj_period: int=1,
        include_first=True, include_last=True
    ) -> NoReturn:
        """"""
        super().__init__([drive_node])

        self.read_traj = read_traj
        self.traj_period = traj_period

        self.include_first = include_first
        self.include_last = include_last

        return
    
    def forward(self, frames: List[Atoms]):
        """
        Args:
            frames: Structures from drive-node's output (end frame of each traj).
            
        """
        super().forward()

        # - save ALL end frames
        cached_conv_fpath = self.directory/"last.xyz" # converged frames
        if not cached_conv_fpath.exists():
            write(cached_conv_fpath, frames)

        # - extract trajetories
        cached_trajs_fpath = self.directory/"trajs.xyz"
        if not cached_trajs_fpath.exists():
            worker = self.input_nodes[0].worker

            trajectories = worker.retrieve(
                read_traj = self.read_traj, traj_period = self.traj_period,
                include_first = self.include_first, include_last = self.include_last,
                ignore_retrieved=False # TODO: ignore misleads...
            )
            flatten_traj_frames = []
            for traj in trajectories:
                flatten_traj_frames.extend(traj)
            write(cached_trajs_fpath, flatten_traj_frames)
        else:
            flatten_traj_frames = read(cached_trajs_fpath, ":")
        
        # TODO: reconstruct trajs to List[List[Atoms]]

        return flatten_traj_frames

class extract_selected_trajectories(Operation):

    """Extract dynamics trajectories from a drive-node's worker.
    """

    def __init__(
        self, drive_node, selector_node, read_traj=True, traj_period: int=1,
        include_first=True, include_last=True
    ) -> NoReturn:
        """"""
        super().__init__([drive_node, selector_node])

        self.read_traj = read_traj
        self.traj_period = traj_period

        self.include_first = include_first
        self.include_last = include_last

        return
    
    def forward(self, frames: List[Atoms], selected_minima: List[Atoms]):
        """
        Args:
            frames: Structures from drive-node's output (end frame of each traj).
            
        """
        super().forward()

        # - save selected minima (end) frames
        cached_conv_fpath = self.directory/"selected.xyz" # converged frames
        if not cached_conv_fpath.exists():
            write(cached_conv_fpath, selected_minima)

        # - find selected minima's wdirs
        given_wdirs = []
        for i, atoms in enumerate(selected_minima):
            wdir = atoms.info.get("wdir", None)
            if wdir is not None:
                given_wdirs.append(wdir)
            else:
                raise RuntimeError(f"There is no wdir for atoms {i}.")

        # - extract trajetories
        cached_trajs_fpath = self.directory/"trajs.xyz"
        if not cached_trajs_fpath.exists():
            worker = self.input_nodes[0].worker

            trajectories = worker.retrieve(
                ignore_retrieved=False, # TODO: ignore misleads...
                given_wdirs = given_wdirs,
                read_traj = self.read_traj, traj_period = self.traj_period,
                include_first = self.include_first, include_last = self.include_last
            )
            flatten_traj_frames = []
            for traj in trajectories:
                flatten_traj_frames.extend(traj)
            write(cached_trajs_fpath, flatten_traj_frames)
        else:
            flatten_traj_frames = read(cached_trajs_fpath, ":")
        
        # TODO: reconstruct trajs to List[List[Atoms]]

        return flatten_traj_frames


def create_extract(method: str, params: dict):
    """"""    
    # TODO: check if params are valid
    if method == "trajs":
        from GDPy.computation.worker.interface import extract_trajectories as op_cls
    elif method == "selected_trajs":
        from GDPy.computation.worker.interface import extract_selected_trajectories as op_cls
    else:
        raise NotImplementedError(f"Unimplemented modifier {method}.")

    return op_cls


def run_driver(structure: str, directory="./", worker=None, o_fname=None):
    """"""
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    generator.directory = directory/"init"
    frames = generator.run()
    #nframes = len(frames)
    #print("nframes: ", nframes)

    wdirs = []
    for i, atoms in enumerate(frames):
        wdir = atoms.info.get("wdir", f"cand{i}")
        wdirs.append(wdir)
    assert len(wdirs) == len(frames), "Have duplicated wdir names..."
    
    driver = worker.driver
    for wdir, atoms in zip(wdirs, frames):
        driver.reset()
        driver.directory = directory/wdir
        print(driver.directory)
        driver.run(atoms, read_exists=True, extra_info=None)
    
    ret_frames = []
    for wdir, atoms in zip(wdirs, frames):
        driver.directory = directory/wdir
        atoms = driver.read_converged()
        ret_frames.append(atoms)
    
    if o_fname is not None:
        write(directory/o_fname, ret_frames)

    return


def run_worker(
    structure: str, directory=pathlib.Path.cwd()/DEFAULT_MAIN_DIRNAME,
    worker: DriverBasedWorker=None, output: str=None, selection=None, 
    nostat: bool=False, batch: int=None
):
    """"""
    directory = pathlib.Path(directory)
    if not directory.exists():
        directory.mkdir()

    # - read structures
    from GDPy.builder import create_generator
    generator = create_generator(structure)
    generator.directory = directory/"init"
    frames = generator.run()
    nframes = len(frames)
    print("nframes: ", nframes)

    #wdirs = params.pop("wdirs", None)
    #if wdirs is None:
    #    wdirs = [f"cand{i}" for i in range(nframes)]

    # - find input frames
    worker.directory = directory
    print("wdir: ", directory)

    _ = worker.run(generator, batch=batch)
    worker.inspect(resubmit=True)
    if worker.get_number_of_running_jobs() == 0:
        # - report
        res_dir = directory/"results"
        res_dir.mkdir(exist_ok=True)
        data = {}
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            if output == "last":
                ret = worker.retrieve(ignore_retrieved=False)
                data["last"] = ret
            elif output == "traj":
                ret = worker.retrieve(
                    ignore_retrieved=False, read_traj=True, traj_period=1,
                )
                data["last"] = [x[-1] for x in ret]
                data["traj"] = [x[:] for x in ret]
                traj_frames = []
                for x in ret:
                    traj_frames.extend(x)
                write(res_dir/"traj.xyz", traj_frames)
            else:
                raise NotImplementedError(f"Output {output} is not implemented.")
            write(res_dir/"last.xyz", data["last"])
    
        if not nostat:
            # - last
            last_frames = data.get("last", None)
            if last_frames is not None:
                energies = [a.get_potential_energy() for a in last_frames]
                numbers = list(range(len(last_frames)))
                sorted_numbers = sorted(numbers, key=lambda i: energies[i])

                content = "{:<24s}  {:<12s}  {:<12s}\n".format("#wdir", "ene", "rank")
                for i, (a, ene) in enumerate(zip(last_frames, energies)):
                    content += "{:<24s}  {:<12.4f}  {:<12d}\n".format(
                        a.info["wdir"], a.get_potential_energy(), sorted_numbers.index(i)
                    )

                with open(res_dir/"last.txt", "w") as fopen:
                    fopen.write(content)

            # - traj
            trajectories = data.get("traj", None)
            if trajectories is not None:
                #first_energies = [x[0].get_potential_energy() for x in trajectories]
                #end_energies = [x[-1].get_potential_energy() for x in trajectories]
                content = ("{:<24s}  "+"{:<12s}  "*5+"\n").format(
                    "#wdir", "nframes", "ene0", "ene-1", "ediff", "grmse"
                )
                for traj_frames in data["traj"]:
                    first_atoms, end_atoms = traj_frames[0], traj_frames[-1]
                    first_ene, end_ene = first_atoms.get_potential_energy(), end_atoms.get_potential_energy()
                    first_pos, end_pos = first_atoms.positions, end_atoms.positions
                    vmin, vlen = find_mic(end_pos-first_pos, first_atoms.get_cell())
                    drmse = np.sqrt(np.var(vlen)) # RMSE of displacement
                    content += ("{:<24s}  "+"{:<12d}  "+"{:<12.4f}  "*4+"\n").format(
                        first_atoms.info["wdir"], len(traj_frames), first_ene, end_ene, end_ene-first_ene,
                        drmse
                    )
                with open(res_dir/"traj.txt", "w") as fopen:
                    fopen.write(content)
                ...

        # - perform further analysis and selection on retrieved results
        if selection is not None:
            params = parse_input_file(selection)["selection"]
            if isinstance(params, list):
                conv_params = copy.deepcopy(params) # for converged or last frames
                traj_params = copy.deepcopy(params) # for trajectory frames
            else: # should be a dict
                conv_params = params.get("last", None)
                assert conv_params is not None, "At least provide selection params for last frames."
                traj_params = params.get("traj", None)
                if traj_params is None:
                    traj_params = copy.deepcopy(conv_params)
            # -- create selectors
            conv_selector = create_selector(conv_params, directory=directory/"results"/"select"/"conv")
            traj_selector = create_selector(traj_params, directory=directory/"results"/"select"/"traj")

            # -- run selections
            conv_frames = data.get("last", None)
            if conv_frames is not None:
                _ = conv_selector.select(conv_frames)

            trajectories = data.get("traj", None)
            if trajectories is not None:
                _ = traj_selector.select(trajectories)

    return

if __name__ == "__main__":
    pass
