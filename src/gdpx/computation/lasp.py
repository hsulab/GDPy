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

from .driver import BaseDriver, Controller, DriverSetting

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
class SinglePointController(Controller):

    name: str = "spc"

    def __post_init__(self):
        """"""
        self.conv_params = {
            "explore_type": "ssw",
            "SSW.SSWsteps": 0,  # spc
        }

        return


@dataclasses.dataclass
class BFGSMinimiser(Controller):

    name: str = "bfgs"

    def __post_init__(self):
        """"""
        maxstep = self.params.get("maxstep", 0.2)

        self.conv_params = {
            "explore_type": "ssw",
            "SSW.SSWsteps": 1,  # BFGS
            "SSW.Bfgs_maxstepsize": maxstep,
        }

        return


@dataclasses.dataclass
class CellBFGSMinimiser(Controller):

    name: str = "bfgs"

    def __post_init__(self):
        """"""
        maxstep = self.params.get("maxstep", 0.2)

        self.conv_params = {
            "explore_type": "ssw",
            "Run_Type": 15,
            "SSW.SSWsteps": 1,  # BFGS
            "SSW.Bfgs_maxstepsize": maxstep,
        }

        return


@dataclasses.dataclass
class MDController(Controller):

    #: Controller name.
    name: str = "md"

    #: Timestep in fs.
    timestep: float = 1.0

    #: Temperature in Kelvin.
    temperature: float = 300.0

    #: Temperature in Kelvin.
    temperature_end: Optional[float] = None

    #: Pressure in bar.
    pressure: float = 1.0

    #: Pressure in Kelvin.
    pressure_end: Optional[float] = None

    #: Whether fix center of mass.
    fix_com: bool = True

    def __post_init__(self):
        """"""
        basic_params = {
            "MD.dt": self.timestep,
            "MD.initial_T": self.temperature,
            "MD.target_T": self.temperature_end,
        }

        # We need keywords: TEBEG and TEEND.
        if self.temperature_end is not None:
            basic_params.update(teend=self.temperature_end)

        self.conv_params = basic_params

        return


@dataclasses.dataclass
class VerletMD(MDController):

    name: str = "verlet"

    def __post_init__(self):
        """"""
        super().__post_init__()

        more_params = dict(explore_type="nve")

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class NoseHooverThermostat(MDController):

    name: str = "nose_hoover"

    def __post_init__(self):
        """"""
        super().__post_init__()

        nhmass = self.params.get("Tdamp", 1000)  # eV*fs**2

        more_params = dict(explore_type="nvt", nhmass=nhmass)

        self.conv_params.update(**more_params)

        return


@dataclasses.dataclass
class ParrinelloRahmanBarostat(MDController):

    name: str = "parrinello_rahman"

    def __post_init__(self):
        """"""
        super().__post_init__()

        nhmass = self.params.get("Tdamp", 1000)  # eV*fs**2

        prmass = self.params.get("Pdamp", 1000)  # eV*fs**2

        pressure = self.pressure * 1e-4  # from bar to GPa

        more_params = {"explore_type": "npt", "nhmass": nhmass, "prmass": prmass, "MD.target_P": pressure}

        if self.pressure_end is not None:
            raise RuntimeError("LASP does not support NPT with changing pressure.")

        self.conv_params.update(**more_params)

        return


controllers = dict(
    # - spc
    single_point_spc=SinglePointController,
    # - min
    bfgs_min=BFGSMinimiser,
    # - cmin
    bfgs_cmin=CellBFGSMinimiser,
    # - md
    verlet_nve=VerletMD,
    nose_hoover_nvt=NoseHooverThermostat,
    parrinello_rahman_npt=ParrinelloRahmanBarostat,
)

default_controllers = dict(
    spc=SinglePointController,
    min=BFGSMinimiser,
    cmin=CellBFGSMinimiser,
    nve=VerletMD,
    nvt=NoseHooverThermostat,
    npt=ParrinelloRahmanBarostat,
)


@dataclasses.dataclass
class LaspDriverSetting(DriverSetting):

    #: Simulation task.
    task: str = "spc"

    #: MD ensemble.
    ensemble: str = "nve"

    #: Driver detailed controller setting.
    controller: dict = dataclasses.field(default_factory=dict)

    #: Whether initialise velocities by LASP itself.
    use_lasp_vinit: bool = True

    #: Force tolerance in minimisation.
    fmax: Optional[float] = 0.05  # eV/Ang

    #: Stress tolerance in minimisation.
    smax: Optional[float] = 0.10  # GPa

    def __post_init__(self):
        """"""
        _init_params = {}
        if self.task == "spc":
            suffix = self.task
        elif self.task == "min":
            suffix = self.task
        elif self.task == "cmin":
            suffix = self.task
        elif self.task == "md":
            suffix = self.ensemble
            _init_params.update(
                timestep=self.timestep,
                temperature=self.temp,
                temperature_end=self.temp,
                pressure=self.press,
                pressure_end=self.pend,
            )
        elif self.task == "freq":
            raise NotImplementedError("")
        else:
            raise RuntimeError(f"Unknown LASP task `{self.task}`.")

        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]

        _init_params.update(**self.controller)
        cont = cont_cls(**_init_params)

        self._internals.update(**cont.conv_params)

        if self.task == "md" and not self.use_lasp_vinit:
            raise Exception("LASP Driver only supports initialising velocities by LASP itself.")

        return

    def get_run_params(self, *args, **kwargs):
        """"""
        # convergence criteria
        steps_ = kwargs.get("steps", self.steps)
        fmax_ = kwargs.get("fmax", self.fmax)
        smax_ = kwargs.get("smax", self.smax)

        run_params = {
            "constraint": kwargs.get("constraint", self.constraint),
            "SSW.MaxOptstep": steps_,
        }

        if self.task == "spc":
            ...
        elif self.task == "min":
            run_params.update(**{"SSW.ftol": fmax_})
        elif self.task == "cmin":
            run_params.update(**{"SSW.ftol": fmax_, "SSW.strtol": smax_})
        elif self.task == "md":
            timestep = self._internals["MD.dt"]
            print_freq = self.dump_period * timestep  # freq has unit fs
            run_params.update(
                **{
                    "MD.ttotal": timestep * steps_,
                    "MD.print_freq": print_freq,
                    "MD.print_strfreq": print_freq,
                    "MD.print_velfreq": print_freq,
                    # "MD.printevery": print_freq,  # default: equal to print_freq
                }
            )
        else:
            raise NotImplementedError(f"LASP driver task `{self.task}` not implemented in get_run_params.")

        # add extra parameters
        run_params.update(**kwargs)

        return run_params


class LaspDriver(BaseDriver):
    """Driver for LASP."""

    name = "lasp"

    default_task = "spc"
    supported_tasks = ["spc", "min", "cmin", "md"]

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
        cache_traj: Optional[list[Atoms]] = None,
        *args,
        **kwargs,
    ) -> None:
        """"""
        assert isinstance(self.calc, LaspNN), "LaspDriver needs LaspNN calculator."
        if ckpt_wdir is None:  # start from the scratch
            run_params = self.setting.get_init_params()
            run_params.update(**self.setting.get_run_params(**kwargs))

            if self.setting.task == "md":
                lasp_random_seed = self.random_seed
                self._print(f"MD Driver's rng: lasp-{lasp_random_seed}")
                run_params.update(Ranseed=lasp_random_seed)
                if self.setting.use_lasp_vinit:
                    if not self.setting.ignore_atoms_velocities:
                        raise Exception(
                            "Cannot use atoms' velocties (ignore_atoms_velocities is false) when use_lasp_vinit is true."
                        )
                else:
                    raise Exception("LASP Driver only supports use_lasp_vinit = True.")

            self.calc.set(**run_params)
        else:
            if cache_traj is None:
                traj = self.read_trajectory()
            else:
                traj = cache_traj
            nframes = len(traj)
            assert nframes > 0, "LaspDriver restarts with a zero-frame trajectory."

            # TODO: velocities?
            atoms = traj[-1]

            target_steps = self.setting.steps
            dump_period = self.setting.dump_period
            if target_steps > 0:
                steps = target_steps + dump_period - nframes * dump_period
            else:
                raise Exception("LaspDriver restart needs a positive target steps.")
            assert steps > 0, "Steps should be greater than 0."

            run_params = self.setting.get_init_params()
            run_params.update(**self.setting.get_run_params(steps=steps))

            self.calc.set(**run_params)

        try:
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
        self, wdir: pathlib.Path, archive_path: Optional[pathlib.Path], *args, **kwargs
    ) -> list[Atoms]:
        """"""
        curr_frames = read_lasp_structures(self.directory, wdir, archive_path=archive_path)

        return curr_frames

    def read_trajectory(self, archive_path: Optional[pathlib.Path] = None, *args, **kwargs) -> list[Atoms]:
        """Read trajectory in the current working directory."""
        # Find all previous working directories
        prev_wdirs = sorted(self.directory.glob(r"[0-9][0-9][0-9][0-9][.]run"))
        self._debug(f"prev_wdirs: {prev_wdirs}")

        # Even though arc file may be empty, the read can give a empty list.
        traj_list = []
        for w in prev_wdirs:
            curr_frames = self._read_a_single_trajectory(w, archive_path)
            traj_list.append(curr_frames)

        curr_Frames = self._read_a_single_trajectory(self.directory, archive_path)
        if not curr_Frames:
            ...
        else:
            traj_list.append(curr_Frames)

        # Concatenate trajectories
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

        # Add step info, and LASP only save step info in md
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
        # general settings
        "explore_type": "ssw",  # ssw, nve nvt npt rigidssw train
        "Run_Type": 5,  # 5 fixed-cell, 15 variable-cell
        "Ranseed": None,
        # ssw
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
        "MD.target_P": 1e-4,  # 1 bar = 1e-4 GPa
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
