import copy
import io
import pathlib
import traceback
from typing import Optional

import numpy as np
from ase import Atoms
from ase.calculators.mixing import LinearCombinationCalculator

from gdpx.backend.plumed import (
    add_colvar_to_atoms_info,
    clap_plumed_file_by_number,
    clap_plumed_file_by_simulations,
    find_input_key_value,
    write_plumed_input_file,
)
from gdpx.utils.archive import open_archive

from ..driver import BaseDriver
from ..observer import create_an_observer
from .calculator import Lammps, _read_a_single_trajectory
from .constants import ASELMPCONFIG
from .settings import LmpDriverSetting


class LmpDriver(BaseDriver):
    """Use lammps to perform dynamics.

    Minimisation and/or molecular dynamics.

    """

    name = "lammps"
    default_task = "min"
    supported_tasks = ["min", "md"]
    setting_cls = LmpDriverSetting

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        calc, params = self._canonicalise_calculator(calc=calc, params=params)
        super().__init__(calc, params, directory=directory, *args, **kwargs)

    def _canonicalise_calculator(self, calc, params: dict):
        units = "metal"
        new_calc, new_params = calc, params
        try:
            from gdpx.potential.plumed.calculators.plumed2 import Plumed

            if isinstance(calc, LinearCombinationCalculator):
                assert len(calc.calcs) == 2, "Number of calculators should be 2."
                if isinstance(calc.calcs[0], Lammps) and isinstance(calc.calcs[1], Plumed):
                    new_calc = calc.calcs[0]
                    new_params = copy.deepcopy(params)
                    new_params["plumed"] = "".join(calc.calcs[1].input)
                    units = calc.calcs[0].units
        except ImportError:
            units = calc.units
        new_params.update(units=units)
        return new_calc, new_params

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        verified = super()._verify_checkpoint(*args, **kwargs)
        if verified:
            if self.setting.ckpt_period >= self.setting.steps:
                verified = self.read_convergence_from_logfile()
            else:
                checkpoints = list(self.directory.glob("restart.*"))
                self._debug(f"checkpoints: {checkpoints}")
                if not checkpoints:
                    verified = False
        return verified

    def _create_dynamics(self, atoms: Atoms, *args, **kwargs):
        lines = []
        if self.setting.task == "md":
            if self.setting.use_lammps_vinit:
                velocity_seed = self.setting.velocity_seed
                if velocity_seed is None:
                    velocity_seed = self.random_seed
                self._print(f"MD Driver's velocity_seed: {velocity_seed}")
                line = f"velocity        mobile create {self.setting.temp} {velocity_seed} dist gaussian "
                if self.setting.remove_translation:
                    line += "mom yes "
                if self.setting.remove_rotation:
                    line += "rot yes "
                if atoms.get_kinetic_energy() > 0.0:
                    if self.setting.ignore_atoms_velocities:
                        atoms.set_momenta(np.zeros(atoms.positions.shape))
                        lines.append(line)
                    else:
                        lines.append("# Use atoms' velocities.")
                else:
                    atoms.set_momenta(np.zeros(atoms.positions.shape))
                    lines.append(line)
            else:
                self._prepare_velocities(
                    atoms,
                    self.setting.velocity_seed,
                    self.setting.ignore_atoms_velocities,
                )
        dynamics = self.setting.get_simulation_inputs(random_seed=self.random_seed, group="mobile")
        lines.extend(dynamics)
        return lines

    def _write_plumed(self, ckpt_wdir: Optional[pathlib.Path], finish_steps: int, curr_temperature: float):
        if self.calc.plumed is None:
            return

        is_continue = ckpt_wdir is not None
        if is_continue:
            assert finish_steps > 0
            required_num_lines = int(finish_steps / self.setting.dump_period)
            assert required_num_lines - finish_steps / self.setting.dump_period == 0, (
                "The finished steps must be multiple of dump_period."
            )
            self._print(f"{required_num_lines=}")
            colvar_files = list(ckpt_wdir.glob("COLVAR*"))
            for fpath in colvar_files:
                clap_plumed_file_by_number(fpath, self.directory / fpath.name, required_num_lines, 0)
            hills_fpath = ckpt_wdir / "HILLS"
            if hills_fpath.exists():
                clap_plumed_file_by_number(hills_fpath, self.directory / "HILLS", required_num_lines, 0)
            kernels_fpath = ckpt_wdir / "KERNELS"
            if kernels_fpath.exists():
                opes_sigma = find_input_key_value(self.calc.plumed, "OPES_METAD", "SIGMA")
                adaptive_sigma_stride = find_input_key_value(self.calc.plumed, "OPES_METAD", "ADAPTIVE_SIGMA_STRIDE")
                if opes_sigma is None or opes_sigma == "ADAPTIVE":
                    adaptive_sigma_stride = (
                        int(adaptive_sigma_stride) if adaptive_sigma_stride is not None else 10
                    )
                    kernel_offset = -int(adaptive_sigma_stride / self.setting.dump_period) + 1
                else:
                    assert adaptive_sigma_stride is None, (
                        "If SIGMA is initialised, ADAPTIVE_SIGMA_STRIDE should not be set."
                    )
                    kernel_offset = 0
                self._print(f"{adaptive_sigma_stride=} {kernel_offset=}")
                clap_plumed_file_by_number(
                    kernels_fpath, self.directory / "KERNELS", required_num_lines, kernel_offset
                )
            state_fpath = ckpt_wdir / "STATE"
            if state_fpath.exists():
                finished_time = finish_steps * self.setting.timestep / 1000.0
                clap_plumed_file_by_simulations(state_fpath, self.directory / "STATE", finished_time)
        write_plumed_input_file(
            pathlib.Path(self.directory),
            copy.deepcopy(self.calc.plumed),
            dict(
                dump_period=self.setting.dump_period,
                ckpt_period=self.setting.ckpt_period,
                temperature=curr_temperature,
            ),
            is_continue=is_continue,
        )

    def _prepare_restart(self, ckpt_wdir, run_params):
        is_continue = ckpt_wdir is not None
        if not is_continue:
            curr_temperature = self.setting.temp
            curr_pressure = self.setting.press
            finish_steps = 0
        else:
            checkpoints = sorted(
                list(ckpt_wdir.glob("restart.*")),
                key=lambda x: int(x.name.split(".")[1]),
            )
            self._debug(f"checkpoints to restart: {checkpoints}")
            target_steps = run_params["steps"]
            finish_steps = int(checkpoints[-1].name.split(".")[1])
            remain_steps = target_steps - finish_steps
            run_params.update(read_restart=str(checkpoints[-1].resolve()), steps=remain_steps)
            if self.setting.tend is not None:
                curr_temperature = (
                    self.setting.temp + (self.setting.tend - self.setting.temp) / target_steps * finish_steps
                )
            else:
                curr_temperature = self.setting.temp
            if self.setting.pend is not None:
                curr_pressure = (
                    self.setting.press + (self.setting.pend - self.setting.press) / target_steps * finish_steps
                )
            else:
                curr_pressure = self.setting.press
        return curr_temperature, curr_pressure, finish_steps

    def _setup_observers(self):
        halt = ""
        if self.setting.observers is not None:
            observers = [create_an_observer(ob_params) for ob_params in self.setting.observers]
            for i, ob in enumerate(observers):
                if hasattr(ob, "get_lammps_command"):
                    halt += ob.get_lammps_command(
                        f"observer_{i:>02d}", "all", dump_period=self.setting.dump_period
                    )
        return halt

    def _irun(self, atoms: Atoms, ckpt_wdir=None, *args, **kwargs):
        run_params = self.setting.get_init_params()
        run_params.update(**self.setting.get_run_params(**kwargs))

        prev_temperature, prev_pressure = self.setting.temp, self.setting.press

        curr_temperature, curr_pressure, finish_steps = self._prepare_restart(ckpt_wdir, run_params)

        self.setting.temp = curr_temperature
        self.setting.press = curr_pressure

        dynamics = self._create_dynamics(atoms, *args, **kwargs)
        self._write_plumed(ckpt_wdir, finish_steps, curr_temperature)

        self.setting.temp = prev_temperature
        self.setting.press = prev_pressure

        halt = self._setup_observers()

        self.calc.set(
            task=self.setting.task,
            dump_period=self.setting.dump_period,
            ckpt_period=self.setting.ckpt_period,
            dynamics=dynamics,
            steps=run_params["steps"],
            constraint=run_params["constraint"],
            etol=run_params["etol"],
            ftol=run_params["ftol"],
            read_restart=run_params.get("read_restart", None),
            extra_fix=run_params["extra_fix"],
            neighbor=run_params["neighbor"],
            neigh_modify=run_params["neigh_modify"],
            halt=halt,
        )
        atoms.calc = self.calc

        try:
            _ = atoms.get_forces()
        except Exception:
            self._debug(traceback.format_exc())

    def _read_a_single_trajectory(self, *args, **kwargs):
        return _read_a_single_trajectory(*args, **kwargs, print_func=self._print, debug_func=self._debug)

    def read_trajectory(
        self,
        type_list=None,
        archive_path: Optional[pathlib.Path] = None,
        *args,
        **kwargs,
    ) -> list[Atoms]:
        if type_list is not None:
            self.calc.type_list = type_list
        curr_units = self.calc.units

        traj_frames = self._aggregate_trajectories(
            units=curr_units,
            mdir=self.directory,
            check_energy=True,
            archive_path=archive_path,
        )

        colvar_io = None
        if archive_path is None:
            colvar_path = self.directory / "COLVAR"
            if colvar_path.exists():
                colvar_io = open(colvar_path)
        else:
            rpath = self.directory.relative_to(self.directory.parent)
            colvar_tarname = str(rpath / "COLVAR")
            with open_archive(archive_path) as tar:
                for tarinfo in tar:
                    if tarinfo.name.startswith(self.directory.name):
                        if tarinfo.name == colvar_tarname:
                            colvar_io = io.StringIO(tar.extractfile(tarinfo).read().decode())

        dump_period_in_ps = self.setting.dump_period * self.setting.timestep / 1000.0

        if colvar_io is not None:
            add_colvar_to_atoms_info(
                colvar_io,
                traj_frames,
                dump_period_in_ps=dump_period_in_ps,
                ignored_columns=["time"],
            )
            colvar_io.close()

        return traj_frames

    def read_convergence_from_logfile(self, *args, **kwargs):
        converged = False
        log_fpath = self.directory / ASELMPCONFIG.log_filename
        if log_fpath.exists() and log_fpath.stat().st_size != 0:
            with open(log_fpath) as fopen:
                lines = fopen.readlines()
            end_line = lines[-1].strip()
            if end_line.startswith("Total wall time:"):
                converged = True
            elif end_line.startswith("Last command: run"):
                with open(self.directory / "EARLYSTOP", "w") as fopen:
                    fopen.write("")
                converged = True
            else:
                self._print(f"LAMMPS ENDLINE: {end_line}")
        return converged
