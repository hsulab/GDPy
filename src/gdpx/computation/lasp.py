#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
import copy
import dataclasses
import time
import shutil
import warnings
import pathlib
import tarfile
import tempfile
import traceback
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.calculators.calculator import FileIOCalculator, EnvironmentError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic

from ..builder.constraints import parse_constraint_info
from .driver import AbstractDriver, DriverSetting


"""Driver and calculator of LaspNN.

Output files by LASP NVE-MD are 
    allfor.arc  allkeys.log  allstr.arc  firststep.restart 
    md.arc  md.restart  vel.arc

"""

def is_number(s):
    """"""
    try:
        float(s)
        return True
    except ValueError:
        ...

    return False

def compare_trajectory_continuity(t0, t1):
    """Compare positions."""
    a0, a1 = t0[-1], t1[0]
    cell = a0.get_cell(complete=True)
    shift = a0.positions - a1.positions
    curr_vectors, curr_distances = find_mic(shift, cell, pbc=True)

    # Due to the floating point precision, arc -> atoms may lead to
    # position inconsistent after 8 decimals...
    return np.allclose(
        curr_vectors, np.zeros(curr_vectors.shape), 
        #rtol=1e-05, atol=1e-08, equal_nan=False
        rtol=1e-04, atol=1e-06, equal_nan=False
    )


class LaspEnergyError(Exception):
    """Error due to the failure of LASP energy.

    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def read_laspset(train_structures):
    """Read LASP TrainStr.txt and TrainFor.txt files."""
    train_structures = Path(train_structures)
    frames = []

    all_energies, all_forces, all_stresses = [], [], []

    # - TrainStr.txt
    # TODO: use yield
    with open(train_structures, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - energy
                line = fopen.readline()
                energy = float(line.strip().split()[-2])
                all_energies.append(energy)
                # - natoms
                line = fopen.readline()
                natoms = int(line.strip().split()[-1])
                # skip 5 lines, symbol info and training weights
                #skipped_lines = [fopen.readline() for i in range(5)]
                # - cell
                cell = []
                for _ in range(1000):
                    lat_line = fopen.readline()
                    if lat_line.strip().startswith("lat"):
                        cell.append(lat_line.strip().split()[1:])
                    if len(cell) == 3:
                        break
                else:
                    raise RuntimeError("Failed to read lattice.")
                cell = np.array(cell, dtype=float)
                # - symbols, positions, and charges
                anumbers, positions, charges = [], [], []
                for i in range(natoms):
                    data = fopen.readline().strip().split()[1:]
                    anumbers.append(int(data[0]))
                    positions.append([float(x) for x in data[1:4]])
                    charges.append(float(data[-1]))
                atoms = Atoms(numbers=anumbers, positions=positions, cell=cell, pbc=True)
                assert fopen.readline().strip().startswith("End one structure")
                frames.append(atoms)
                #break
            if not line:
                break
    
    # - TrainFor.txt
    train_forces = train_structures.parent / "TrainFor.txt"
    with open(train_forces, "r") as fopen:
        while True:
            line = fopen.readline()
            if line.strip().startswith("Start one structure"):
                # - stress, voigt order
                stress = np.array(fopen.readline().strip().split()[1:], dtype=float)
                # - symbols, forces
                anumbers, forces = [], []
                line = fopen.readline()
                while True:
                    if line.strip().startswith("force"):
                        data = line.strip().split()[1:]
                        anumbers.append(int(data[0]))
                        forces.append([float(x) for x in data[1:4]])
                    else:
                        all_forces.append(forces)
                        assert line.strip().startswith("End one structure")
                        break
                    line = fopen.readline()
                #break
            if not line:
                break
    
    for i, atoms in enumerate(frames):
        calc = SinglePointCalculator(
            atoms, energy=all_energies[i], forces=all_forces[i]
        )
        atoms.calc = calc
    write(train_structures.parent / "dataset.xyz", frames)

    return frames

def read_lasp_structures(
        mdir: pathlib.Path, wdir: pathlib.Path, archive_path: pathlib.Path=None, 
        *args, **kwargs
    ) -> List[Atoms]:
    """Read simulation trajectory."""
    wdir = pathlib.Path(wdir)

    # - check if output file exists...
    if (not (wdir/"allstr.arc").exists()) and archive_path is None:
        return []

    # - get IO
    if archive_path is None:
        with open(wdir/"allstr.arc", "r") as fopen:
            stru_io = io.StringIO(fopen.read())
        afrc_io = open(wdir/"allfor.arc", "r") # atomic forces in arc format
        lout_io = open(wdir/"lasp.out", "r")
    else:
        rpath = wdir.relative_to(mdir.parent)
        stru_tarname = str(rpath/"allstr.arc")
        afrc_tarname = str(rpath/"allfor.arc")
        lout_tarname = str(rpath/"lasp.out")
        with tarfile.open(archive_path, "r:gz") as tar:
            for tarinfo in tar:
                if tarinfo.name.startswith(wdir.name):
                    if tarinfo.name == stru_tarname:
                        stru_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    elif tarinfo.name == afrc_tarname:
                        afrc_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    elif tarinfo.name == lout_tarname:
                        lout_io = io.StringIO(tar.extractfile(tarinfo.name).read().decode())
                    else:
                        ...
                else:
                    ...
            else: # TODO: if not find target traj?
                ...

    # - parse data
    # NOTE: ASE does not support read dmol-arc from stringIO
    with tempfile.NamedTemporaryFile(mode="w", suffix=".arc") as tmp:
        tmp.write(stru_io.getvalue())
        tmp.seek(0)
        traj_frames = read(tmp.name, ":", format="dmol-arc")
    natoms = len(traj_frames[-1])

    traj_steps = []
    traj_energies = []
    traj_forces = []

    # NOTE: for lbfgs opt, steps+2 frames would be output?

    # have to read last structure
    while True:
        line = afrc_io.readline()
        if line.strip().startswith("For"):
            step = int(line.split()[1])
            traj_steps.append(step)
            # NOTE: need check if energy is a number
            #       some ill structure may result in large energy of ******
            energy_data = line.split()[3]
            try:
                energy = float(energy_data)
            except ValueError:
                energy = np.inf
                msg = "Energy is too large at {}. The structure maybe ill-constructed.".format(wdir)
                warnings.warn(msg, UserWarning)
            traj_energies.append(energy)
            # stress
            line = afrc_io.readline()
            stress = np.array(line.split()) # TODO: what is the format of stress
            # forces
            forces = []
            for j in range(natoms):
                line = afrc_io.readline()
                force_data = line.strip().split()
                if len(force_data) == 3: # expect three numbers
                    force_data_ = []
                    for x in force_data:
                        if not is_number(x):
                            force_data_.append(np.inf)
                        else:
                            force_data_.append(float(x))
                    force_data = force_data_
                else: # too large forces make out become ******
                    force_data = [np.inf]*3
                forces.append(force_data)
            forces = np.array(forces, dtype=float)
            traj_forces.append(forces)
        if line.strip() == "":
            pass
        if not line: # if line == "":
            break
    assert len(traj_frames) == len(traj_steps), "Output number is inconsistent."
    
    # - create traj
    for i, atoms in enumerate(traj_frames):
        calc = SinglePointCalculator(
            atoms, energy=traj_energies[i], forces=traj_forces[i]
        )
        atoms.calc = calc

    # check if the structure is too bad... LASP v3.3.4
    TOO_SHORT_BOND_TAG = "Warning: Minimum Structure with too short bond"
    is_badstru = False
    lines = lout_io.readlines()
    for line in lines:
        if TOO_SHORT_BOND_TAG in line:
            is_badstru = True
            break
        else:
            ...
    traj_frames[-1].info["is_badstru"] = is_badstru

    # - close IO
    stru_io.close()
    afrc_io.close()
    lout_io.close()

    return traj_frames

@dataclasses.dataclass
class LaspDriverSetting(DriverSetting):

    def __post_init__(self):
        """"""
        if self.task == "min":
            self._internals.update(
                **{
                    "explore_type": "ssw",
                    "SSW.SSWsteps": 1, # BFGS
                    "SSW.ftol": self.fmax
                }
            )

            assert self.dump_period == 1, "LaspDriver must have dump_period ==1."
        
        if self.task == "md":
            if self.tend is None:
                self.tend = self.temp
            self._internals.update(
                **{
                    "explore_type": self.md_style,
                    "Ranseed": self.velocity_seed,
                    "MD.dt": self.timestep,
                    "MD.initial_T": self.temp,
                    "MD.target_T": self.tend,
                    "nhmass": self.Tdamp,
                    "MD.target_P": self.press,
                    "MD.prmass": self.Pdamp,
                }
            )
            ...

        # - shared params

        # - special params

        return
    
    def get_run_params(self, *args, **kwargs):
        """"""
        steps_ = kwargs.get("steps", self.steps)
        fmax_ = kwargs.get("fmax", self.fmax)

        run_params = {
            "constraint": kwargs.get("constraint", self.constraint),
            "SSW.MaxOptstep": steps_
        }

        if self.task == "min":
            run_params.update(
                **{"SSW.ftol": fmax_}
            )
        
        if self.task == "md":
            timestep = self._internals["MD.dt"]
            run_params.update(
                **{
                    "MD.ttotal": timestep*steps_,
                    "MD.print_freq": self.dump_period*timestep, # freq has unit fs
                    "MD.print_strfreq": self.dump_period*timestep
                }
            )

        # - add extra parameters
        run_params.update(
            **kwargs
        )

        return run_params

class LaspDriver(AbstractDriver):

    """Driver for LASP."""

    name = "lasp"

    #: Whether accepct the bad structure due to crashed FF or SCF-unconverged DFT.
    accept_bad_structure: bool = True

    # - defaults
    default_task = "min"
    supported_tasks = ["min", "md"]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        super().__init__(calc, params, directory=directory, *args, **kwargs)

        self.setting = LaspDriverSetting(**params)

        return
    
    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            laspstr = self.directory / "allstr.arc"
            if (laspstr.exists() and laspstr.stat().st_size != 0):
                verified = True
            else:
                verified = False
        else:
            ...

        return verified

    def _irun(self, atoms: Atoms, ckpt_wdir=None, cache_traj: List[Atoms]=None, *args, **kwargs) -> None:
        """"""
        try:
            if ckpt_wdir is None: # start from the scratch
                # - init params
                run_params = self.setting.get_init_params()
                run_params.update(**self.setting.get_run_params(**kwargs))

                self.calc.set(**run_params)
                atoms.calc = self.calc

                _ = atoms.get_forces()
            else:
                # TODO: velocities?
                if cache_traj is None:
                    traj = self.read_trajectory()
                else:
                    self._debug("use cache trajectory to restart...")
                    traj = cache_traj
                nframes = len(traj)
                assert nframes > 0, "LaspDriver restarts with a zero-frame trajectory."
                atoms = traj[-1]
                target_steps = self.setting.steps
                dump_period = self.setting.dump_period
                if target_steps > 0:
                    steps = target_steps + dump_period - nframes*dump_period
                assert steps > 0, "Steps should be greater than 0."
                run_params = self.setting.get_init_params()
                run_params.update(**self.setting.get_run_params(steps=steps))

                self.calc.set(**run_params)
                atoms.calc = self.calc

                _ = atoms.get_forces()

        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return
    
    def read_force_convergence(self, *args, **kwargs) -> bool:
        """"""
        return self.calc._is_converged()
    
    def _read_a_single_trajectory(self, wdir: pathlib.Path, archive_path: pathlib.Path, *args, **kwargs) -> List[Atoms]:
        """"""
        curr_frames = read_lasp_structures(self.directory, wdir, archive_path=archive_path)

        return curr_frames
    
    def read_trajectory(self, archive_path: pathlib.Path=None, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        self._debug(f"prev_wdirs: {prev_wdirs}")

        traj_list = []
        for w in prev_wdirs:
            curr_frames = self._read_a_single_trajectory(w, archive_path)
            traj_list.append(curr_frames)
        
        # Even though arc file may be empty, the read can give a empty list...
        laspstr = self.directory / "allstr.arc"
        traj_list.append(self._read_a_single_trajectory(self.directory, archive_path))

        # -- concatenate
        traj_frames, ntrajs = [], len(traj_list)
        if ntrajs > 0:
            traj_frames.extend(traj_list[0])
            for i in range(1, ntrajs):
                if traj_list[i]: # check if the traj is a empty list
                    #assert np.allclose(traj_list[i-1][-1].positions, traj_list[i][0].positions), f"Traj {i-1} and traj {i} are not consecutive."
                    assert compare_trajectory_continuity(traj_list[i-1], traj_list[i]), f"Traj {i-1} and traj {i} are not consecutive."
                    traj_frames.extend(traj_list[i][1:])
                else:
                    ...
        else:
            ...
        
        nframes = len(traj_frames)
        self._debug(f"LASP read_trajectory nframes: {nframes}")

        # NOTE: LASP only save step info in MD simulations...
        for i in range(nframes):
            traj_frames[i].info["step"] = i*self.setting.dump_period

        return traj_frames


class LaspNN(FileIOCalculator):

    #: Calculator name.
    name: str = "LaspNN"

    #: Implemented properties.
    implemented_properties: List[str] = ["energy", "forces"]
    # implemented_propertoes = ["energy", "forces", "stress"]

    #: LASP command.
    command = "lasp"

    #: Default calculator parameters, NOTE which have ase units.
    default_parameters = {
        # built-in parameters
        "potential": "NN",
        # - general settings
        "explore_type": "ssw", # ssw, nve nvt npt rigidssw train
        "Ranseed": None,
        # - ssw
        "SSW.internal_LJ": True,
        "SSW.ftol": 0.05, # fmax
        "SSW.SSWsteps": 0, # 0 sp 1 opt >1 ssw search
        "SSW.Bfgs_maxstepsize": 0.2,
        "SSW.MaxOptstep": 0, # lasp default 300
        "SSW.output": "T",
        "SSW.printevery": "T",
        # md
        "MD.dt": 1.0, # fs
        "MD.ttotal": 0, 
        "MD.initial_T": None,
        "MD.equit": 0,
        "MD.target_T": 300, # K
        "MD.nhmass": 1000, # eV*fs**2
        "MD.target_P": 300, # 1 bar = 1e-4 GPa
        "MD.prmass": 1000, # eV*fs**2
        "MD.realmass": ".true.",
        "MD.print_freq": 10,
        "MD.print_strfreq": 10,
        # calculator-related
        "constraint": None, # str, lammps-like notation
    }

    def __init__(self, *args, label="LASP", **kwargs):
        """Init calculator.

        The potential path would be resolved.

        """
        FileIOCalculator.__init__(self, *args, label=label, **kwargs)

        # NOTE: need resolved pot path
        pot_ = {}
        pot = self.parameters.get("pot", None)
        for k, v in pot.items():
            pot_[k] = Path(v).resolve()
        self.set(pot=pot_)

        return
    
    def calculate(self, *args, **kwargs):
        """Perform the calculation."""
        FileIOCalculator.calculate(self, *args, **kwargs)

        return
    
    def write_input(self, atoms, properties=None, system_changes=None):
        """Write LASP inputs."""
        # create calc dir
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # structure
        write(
            os.path.join(self.directory, "lasp.str"), atoms, format="dmol-arc",
            parallel=False
        )

        # check symbols and corresponding potential file
        atomic_types = set(self.atoms.get_chemical_symbols()) # TODO: sort by 

        # - potential choice
        # NOTE: only for LaspNN now
        content  = "potential {}\n".format(self.parameters["potential"])
        assert self.parameters["potential"] == "NN", "Lasp calculator only support NN now."

        content += "%block netinfo\n"
        for atype in atomic_types:
            # write path
            pot_path = Path(self.parameters["pot"][atype]).resolve()
            content += "  {:<4s} {:s}\n".format(atype, pot_path.name)
            # creat potential link
            pot_link = Path(os.path.join(self.directory,pot_path.name))
            if not pot_link.is_symlink(): # false if not exists
                pot_link.symlink_to(pot_path)
        content += "%endblock netinfo\n"

        # - atom constraint
        constraint = self.parameters["constraint"]
        mobile_text, frozen_text = parse_constraint_info(atoms, constraint)

        if frozen_text is not None:
            content += "%block fixatom\n"
            frozen_block = frozen_text.strip().split()
            for block in frozen_block:
                info = block.split(":")
                if len(info) == 2:
                    s, e = info
                else:
                    s, e = info[0], info[0]
                content += "  {} {} xyz\n".format(s, e)
            content += "%endblock fixatom\n"
        
        # - general settings
        seed = self.parameters["Ranseed"]
        if seed is None:
            seed = np.random.randint(1,10000)
        content += "{}  {}".format("Ranseed", seed)

        # - simulation task
        explore_type = self.parameters["explore_type"]
        content += "\nexplore_type {}\n".format(explore_type)
        if explore_type == "ssw":
            for key, value in self.parameters.items():
                if key.startswith("SSW."):
                    content += "{}  {}\n".format(key, value)
        elif explore_type in ["nve", "nvt", "npt"]:
            required_keys = [
                "MD.dt", "MD.ttotal", "MD.realmass", 
                "MD.print_freq", "MD.print_strfreq"
            ]
            if explore_type == "nvt":
                required_keys.extend(["MD.initial_T", "MD.target_T", "MD.equit"])
            if explore_type == "npt":
                required_keys.extend(["MD.target_P"])
            
            self.parameters["MD.ttotal"] = (
                self.parameters["MD.dt"] * self.parameters["SSW.MaxOptstep"]
            )

            for k, v in self.parameters.items():
                if k == "MD.target_P":
                    v *= 1e4 # from bar to GPa
                if k in required_keys:
                    content += "{}  {}\n".format(k, v)
        else:
            # TODO: should check explore_type in init
            pass

        with open(os.path.join(self.directory, "lasp.in"), "w") as fopen:
            fopen.write(content)

        return
    
    def read_results(self):
        """Read LASP results."""
        # have to read last structure
        traj_frames = read_lasp_structures(self.directory, self.directory)

        energy = traj_frames[-1].get_potential_energy()
        forces = traj_frames[-1].get_forces().copy()

        self.results["energy"] = energy
        self.results["forces"] = forces

        return
    
    def _is_converged(self) -> bool:
        """Check whether LASP simulation is converged."""
        converged = False
        lasp_out = Path(os.path.join(self.directory, "lasp.out" ))
        if lasp_out.exists():
            with open(lasp_out, "r") as fopen:
                lines = fopen.readlines()
            if (# NOTE: its a typo in LASP!!
                lines[-1].strip().startswith("elapse_time") or # v3.3.4
                lines[-1].strip().startswith("Elapse_time")    # v3.4.5
            ):
                converged = True

        return converged


if __name__ == "__main__":
    ...
