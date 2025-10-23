#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import os
import pathlib
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.calculator import FileIOCalculator
from ase.io import write

from gdpx.backend.lasp import compare_trajectory_continuity, read_lasp_structures
from gdpx.group import evaluate_constraint_expression
from gdpx.utils.strconv import integers_to_string

from .driver import BaseDriver, DriverSetting


"""Driver and calculator of LaspNN.

Output files by LASP NVE-MD are 
    allfor.arc  allkeys.log  allstr.arc  firststep.restart 
    md.arc  md.restart  vel.arc

"""


class LaspEnergyError(Exception):
    """Error due to the failure of LASP energy."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@dataclasses.dataclass
class LaspDriverSetting(DriverSetting):

    #: MD ensemble.
    ensemble: str = "nve"

    #: Temperature damping factor.
    Tdamp: float = 100.0  # fs

    #: Pressure damping factor.
    Pdamp: float = 100.0  # fs

    #: Stress tolerance in minimisation.
    smax: Optional[float] = 0.10  # GPa

    def __post_init__(self):
        """"""
        if self.task == "min":
            self._internals.update(
                **{
                    "explore_type": "ssw",
                    "SSW.SSWsteps": 1,  # BFGS
                    "SSW.ftol": self.fmax,
                }
            )
            assert self.dump_period == 1, "LaspDriver/min must have dump_period ==1."
        elif self.task == "cmin":
            self._internals.update(
                **{
                    "explore_type": "ssw",
                    "Run_Type": 15,
                    "SSW.SSWsteps": 1,  # BFGS
                    "SSW.ftol": self.fmax,
                    "SSW.strtol": self.smax,  # GPa
                }
            )
            assert self.dump_period == 1, "LaspDriver/cmin must have dump_period ==1."
        elif self.task == "md":
            if self.tend is None:
                self.tend = self.temp
            self._internals.update(
                **{
                    "explore_type": self.ensemble,
                    "Ranseed": self.velocity_seed,
                    "MD.dt": self.timestep,
                    "MD.initial_T": self.temp,
                    "MD.target_T": self.tend,
                    "nhmass": self.Tdamp,
                    "MD.target_P": self.press,
                    "MD.prmass": self.Pdamp,
                }
            )
        else:
            raise RuntimeError(f"Unknown task `{self.task}` for LaspDriver.")

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        steps_ = kwargs.get("steps", self.steps)
        fmax_ = kwargs.get("fmax", self.fmax)
        smax_ = kwargs.get("smax", self.smax)

        run_params = {
            "constraint": kwargs.get("constraint", self.constraint),
            "SSW.MaxOptstep": steps_,
        }

        if self.task == "min":
            run_params.update(**{"SSW.ftol": fmax_, "SSW.strtol": smax_})

        if self.task == "md":
            timestep = self._internals["MD.dt"]
            run_params.update(
                **{
                    "MD.ttotal": timestep * steps_,
                    "MD.print_freq": self.dump_period * timestep,  # freq has unit fs
                    "MD.print_strfreq": self.dump_period * timestep,
                }
            )

        # - add extra parameters
        run_params.update(**kwargs)

        return run_params


class LaspDriver(BaseDriver):
    """Driver for LASP."""

    name = "lasp"

    default_task = "min"
    supported_tasks = ["min", "md"]

    #: Whether accepct the bad structure due to crashed FF or SCF-unconverged DFT.
    accept_bad_structure: bool = True

    #: Class for setting.
    setting_cls: type[DriverSetting] = LaspDriverSetting

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            laspstr = self.directory / "allstr.arc"
            if laspstr.exists() and laspstr.stat().st_size != 0:
                verified = True
            else:
                verified = False
        else:
            ...

        return verified

    def _irun(
        self,
        atoms: Atoms,
        ckpt_wdir=None,
        cache_traj: list[Atoms] = None,
        *args,
        **kwargs,
    ) -> None:
        """"""
        try:
            if ckpt_wdir is None:  # start from the scratch
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
                    steps = target_steps + dump_period - nframes * dump_period
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

    def _read_a_single_trajectory(
        self, wdir: pathlib.Path, archive_path: pathlib.Path, *args, **kwargs
    ) -> list[Atoms]:
        """"""
        curr_frames = read_lasp_structures(self.directory, wdir, archive_path=archive_path)

        return curr_frames

    def read_trajectory(self, archive_path: pathlib.Path = None, *args, **kwargs) -> list[Atoms]:
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
                if traj_list[i]:  # check if the traj is a empty list
                    # assert np.allclose(traj_list[i-1][-1].positions, traj_list[i][0].positions), f"Traj {i-1} and traj {i} are not consecutive."
                    assert compare_trajectory_continuity(
                        traj_list[i - 1], traj_list[i]
                    ), f"Traj {i-1} and traj {i} are not consecutive."
                    traj_frames.extend(traj_list[i][1:])
                else:
                    ...
        else:
            ...

        nframes = len(traj_frames)
        self._debug(f"LASP read_trajectory nframes: {nframes}")

        # NOTE: LASP only save step info in MD simulations...
        for i in range(nframes):
            traj_frames[i].info["step"] = i * self.setting.dump_period

        return traj_frames


class LaspNN(FileIOCalculator):

    #: Implemented properties.
    implemented_properties: list[str] = ["energy", "forces", "stress"]

    #: Default calculator parameters, NOTE which have ase units.
    default_parameters = {
        # built-in parameters
        "potential": "NN",
        # - general settings
        "explore_type": "ssw",  # ssw, nve nvt npt rigidssw train
        "Run_Type": 5,  # 5 fixed-cell, 15 variable-cell
        "Ranseed": None,
        # - ssw
        "SSW.internal_LJ": True,
        "SSW.ftol": 0.05,  # fmax
        "SSW.strtol": 0.10,  # smax
        "SSW.SSWsteps": 0,  # 0 sp 1 opt >1 ssw search
        "SSW.Bfgs_maxstepsize": 0.2,
        "SSW.MaxOptstep": 0,  # lasp default 300
        "SSW.output": "T",
        "SSW.printevery": "T",
        # md
        "MD.dt": 1.0,  # fs
        "MD.ttotal": 0,
        "MD.initial_T": None,
        "MD.equit": 0,
        "MD.target_T": 300,  # K
        "MD.nhmass": 1000,  # eV*fs**2
        "MD.target_P": 300,  # 1 bar = 1e-4 GPa
        "MD.prmass": 1000,  # eV*fs**2
        "MD.realmass": ".true.",
        "MD.print_freq": 10,
        "MD.print_strfreq": 10,
        # calculator-related
        "constraint": None,  # str, lammps-like notation
    }

    def __init__(self, command="lasp", label="LaspNN", **kwargs):
        """Init calculator.

        The potential path would be resolved.

        """
        FileIOCalculator.__init__(self, command=command, label=label, **kwargs)

        # Complete command
        command_ = self.profile.command
        self.profile.command = command_

        # Resolve potential paths
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
            os.path.join(self.directory, "lasp.str"),
            atoms,
            format="dmol-arc",
            parallel=False,
        )

        # Check symbols and corresponding potential file
        atomic_types = sorted(list(set(self.atoms.get_chemical_symbols())))

        # Check potential choice and must be NN
        content = "potential {}\n".format(self.parameters["potential"])
        assert self.parameters["potential"] == "NN", "Lasp calculator only support NN now."

        content += "%block netinfo\n"
        for atype in atomic_types:
            # write path
            pot_path = Path(self.parameters["pot"][atype]).resolve()
            content += "  {:<4s} {:s}\n".format(atype, pot_path.name)
            # creat potential link
            pot_link = Path(os.path.join(self.directory, pot_path.name))
            if not pot_link.is_symlink():  # false if not exists
                pot_link.symlink_to(pot_path)
        content += "%endblock netinfo\n"

        # Add atomic constraints
        cons_expr = self.parameters["constraint"]
        _, frozen_indices = evaluate_constraint_expression(atoms, cons_expr)
        if frozen_indices:
            frozen_text = integers_to_string(frozen_indices, inp_convention="ase")
        else:
            frozen_text = None

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

        # General parameters
        seed = self.parameters["Ranseed"]
        if seed is None:
            seed = np.random.randint(1, 10000)
        content += "{}  {}".format("Ranseed", seed)

        # Simulation parameters
        explore_type = self.parameters["explore_type"]
        content += f"\nexplore_type {explore_type}\n"
        content += f"Run_Type {self.parameters['Run_Type']}\n"
        if explore_type == "ssw":
            for key, value in self.parameters.items():
                if key.startswith("SSW."):
                    content += "{}  {}\n".format(key, value)
        elif explore_type in ["nve", "nvt", "npt"]:
            required_keys = [
                "MD.dt",
                "MD.ttotal",
                "MD.realmass",
                "MD.print_freq",
                "MD.print_strfreq",
            ]
            if explore_type == "nvt":
                required_keys.extend(["MD.initial_T", "MD.target_T", "MD.equit"])
            if explore_type == "npt":
                required_keys.extend(["MD.target_P"])

            self.parameters["MD.ttotal"] = self.parameters["MD.dt"] * self.parameters["SSW.MaxOptstep"]

            for k, v in self.parameters.items():
                if k == "MD.target_P":
                    v *= 1e4  # from bar to GPa
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
        # Read the entire trajectory but only store the last frame
        wdir = pathlib.Path(self.directory)
        traj_frames = read_lasp_structures(wdir, wdir)

        energy = traj_frames[-1].get_potential_energy()
        forces = traj_frames[-1].get_forces().copy()

        self.results["energy"] = energy
        self.results["forces"] = forces

        return

    def _is_converged(self) -> bool:
        """Check whether LASP simulation is converged."""
        converged = False
        lasp_out = Path(os.path.join(self.directory, "lasp.out"))
        if lasp_out.exists():
            with open(lasp_out, "r") as fopen:
                lines = fopen.readlines()
            end_line = lines[-1].strip()
            # Test on v3.3.4 (has a typo), v3.4.5, and v3.7.3
            if end_line.startswith("elapse_time") or end_line.startswith("Elapse_time"):
                converged = True

        return converged


if __name__ == "__main__":
    ...
