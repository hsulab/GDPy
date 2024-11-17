#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import json
import pathlib
import shutil
from typing import List, Optional, Union

import jax_md
import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
from deepmd_jax.md import Simulation

from .driver import AbstractDriver, Controller, DriverSetting


def save_trajectory_and_checkpoint(
    chemical_symbols,
    data: dict,
    start: int,
    end: int,
    dump_period: int,
    end_step: int,
    wdir: pathlib.Path,
    append: bool = False,
    ckpt_number: int = 3,
):
    """"""
    step_indices = list(range(start, end))

    frames = []
    for s_i, c_i in enumerate(step_indices):
        if (c_i % dump_period == 0) or (c_i == start) or (c_i == end_step):
            box = data["box"][s_i]
            if box.size == 3:
                box = np.diag(box)
            else:
                box = box.reshape(3, 3)
            atoms = Atoms(
                chemical_symbols,
                positions=data["position"][s_i],
                velocities=data["velocity"][s_i] / units.fs,
                cell=box,
                pbc=True,  # TODO: Check pbc?
            )
            atoms.info["step"] = c_i
            frames.append(atoms)

    write(wdir / "traj.xyz", frames, append=append)

    ckpt_wdir = wdir / f"checkpoint.{step_indices[-1]}"
    ckpt_wdir.mkdir(exist_ok=True)
    write(ckpt_wdir / "structures.xyz", frames[-1])

    # remove checkpoints if the number is over ckpt_number
    ckpt_wdirs = sorted(wdir.glob("checkpoint*"), key=lambda x: int(x.name[11:]))
    num_ckpts = len(ckpt_wdirs)
    if num_ckpts > ckpt_number:
        for w in ckpt_wdirs[:-ckpt_number]:
            shutil.rmtree(w)

    return


def run_dynamics(
    chemical_symbols,
    dynamics,
    start: int,
    steps: int,
    dump_period: int,
    ckpt_period: int,
    ckpt_number: int,
    wdir: pathlib.Path,
):
    """"""
    num_chunks = int(steps / ckpt_period)

    init_steps = ckpt_period
    if num_chunks < 1:
        init_steps = steps

    # run the first chunk
    traj = dynamics.run(steps=init_steps)
    save_trajectory_and_checkpoint(
        chemical_symbols,
        traj,
        start,
        start + ckpt_period + 1,
        dump_period,
        end_step=start+steps,
        wdir=wdir,
        append=False,
        ckpt_number=ckpt_number,
    )

    # run intermediate chunks
    for i in range(1, num_chunks):
        traj = dynamics.run(steps=ckpt_period)
        save_trajectory_and_checkpoint(
            chemical_symbols,
            traj,
            start + i * ckpt_period + 1,
            start + (i + 1) * ckpt_period + 1,
            dump_period,
            end_step=start+steps,
            wdir=wdir,
            append=True,
            ckpt_number=ckpt_number,
        )

    # run the last chunk
    remaining_steps = steps - ckpt_period * num_chunks
    if remaining_steps > 0:
        traj = dynamics.run(steps=remaining_steps)
        save_trajectory_and_checkpoint(
            chemical_symbols,
            traj,
            start + num_chunks * ckpt_period + 1,
            start + num_chunks * ckpt_period + remaining_steps + 1,
            dump_period,
            end_step=start+steps,
            wdir=wdir,
            append=True,
            ckpt_number=ckpt_number,
        )

    return

@dataclasses.dataclass
class Verlet(Controller):

    name: str = "verlet"

    timestep: float = 1.0

    temperature: float = 300.0

    def __post_init__(self):
        """"""
        self.conv_params = dict(
            routine="nve".upper(),
            dt=self.timestep,
            temperature=self.temperature,
        )

        return


@dataclasses.dataclass
class NoseHooverChainThermostat(Controller):

    #: Controller name.
    name: str = "nose_hoover"

    #: Timestep in fs.
    timestep: float = 1.0

    #: Temperature in Kelvin.
    temperature: float = 300.0

    #: Pressure in bar.
    pressure: float = 1.0

    def __post_init__(self):
        """"""
        taut = self.params.get("Tdamp", 100.0)  # fs
        assert taut is not None

        chain_length = self.params.get("chain_length", 3)
        assert chain_length is not None

        chain_steps = self.params.get("chain_steps", 1)
        assert chain_steps is not None

        self.conv_params = dict(
            routine="nvt".upper(),
            dt=self.timestep,
            temperature=self.temperature,
            tau_t=taut,
            chain_length_t=chain_length,
            chain_steps_t=chain_steps,
            sy_steps_t=1,
        )

        return

@dataclasses.dataclass
class NoseHooverChainBarostat(NoseHooverChainThermostat):

    def __post_init__(self):
        """"""
        super().__post_init__()

        taup = self.params.get("Pdamp", 1000.0)  # fs
        assert taup is not None

        chain_length_p = self.params.get("chain_length_p", 3)
        assert chain_length_p is not None

        chain_steps_p = self.params.get("chain_steps_P", 1)
        assert chain_steps_p is not None

        self.conv_params.update(
            routine="npt".upper(),
            pressure=self.pressure,
            tau_p=taup,
            chain_length_p=chain_length_p,
            chain_steps_p=chain_steps_p,
            sy_steps_p=1
        )

        return


controllers = dict(
    verlet_nve=Verlet,
    nose_hoover_chain_nvt=NoseHooverChainThermostat,
    nose_hoover_chain_npt=NoseHooverChainBarostat,
)

default_controllers = dict(
    nve=Verlet,
    nvt=NoseHooverChainThermostat,
    npt=NoseHooverChainBarostat,
)


@dataclasses.dataclass
class DeepmdJaxDriverSetting(DriverSetting):

    #: MD ensemble.
    ensemble: str = "nve"

    #: Driver detailed controller setting.
    controller: dict = dataclasses.field(default_factory=dict)

    #: Whether fix com to the its initial position.
    fix_com: bool = False

    def __post_init__(self):
        """"""
        if self.fix_com:
            raise RuntimeError("DeepmdJaxDriver does not suppotr fix_com dynamics.")

        _init_params = {}
        if self.task == "md":
            suffix = self.ensemble
            _init_params.update(
                temperature=self.temp, pressure=self.press,
                timestep=self.timestep
            )
        else:
            raise RuntimeError(f"Failed to parse task {self.task}.")
        _init_params.update(**self.controller)

        if self.controller:
            cont_cls_name = self.controller["name"] + "_" + suffix
            if cont_cls_name in controllers:
                cont_cls = controllers[cont_cls_name]
            else:
                raise RuntimeError(f"Unknown controller {cont_cls_name}.")
        else:
            cont_cls = default_controllers[suffix]

        cont = cont_cls(**_init_params)

        self._internals.update(**cont.conv_params)

        return

    def get_run_params(self, *args, **kwargs) -> dict:
        """"""
        run_params = dict(
            steps=kwargs.get("steps", self.steps),
            constraint=kwargs.get("constraint", self.constraint),
        )

        return run_params


class DeepmdJaxDriver(AbstractDriver):

    #: Driver's name.
    name: str = "jax"

    #: Default Task.
    default_task: str = "spc"

    #: Supported tasks.
    supported_tasks: List[str] = ["spc", "min", "md"]

    #: Class for setting.
    setting_cls: type[DriverSetting] = DeepmdJaxDriverSetting

    def _irun(
        self,
        atoms: Atoms,
        ckpt_wdir: Optional[Union[str, pathlib.Path]] = None,
        *args,
        **kwargs,
    ):
        """Run the simulation."""
        # get things not changed from scratch or restart
        type_list = self.calc.type_list
        assert type_list == list(
            sorted(type_list)
        ), f"deepmd_jax must use a type_list in the alphabetical order, not `{type_list}`."
        type_map = {k: v for v, k in enumerate(type_list)}

        chemical_symbols = atoms.get_chemical_symbols()
        type_indices = np.array([type_map[s] for s in chemical_symbols])
        masses = atomic_masses[[atomic_numbers[s] for s in type_list]]

        # update steps
        init_params = self.setting.get_init_params()
        run_params = self.setting.get_run_params(**kwargs)
        if ckpt_wdir is None:
            self._prepare_velocities(
                atoms,
                velocity_seed=self.setting.velocity_seed,
                ignore_atoms_velocities=self.setting.ignore_atoms_velocities,
            )
            start = 0
            steps = run_params["steps"]
            with open(self.directory/"params.json", "w") as fopen:
                json.dump(dict(init=init_params, run=run_params), fopen, indent=2)
        else:
            assert ckpt_wdir is not None
            prev_ckpt_wdir = self._find_latest_checkpoint(pathlib.Path(ckpt_wdir))
            atoms = self._load_checkpoint(prev_ckpt_wdir)
            self._print(f"Load checkpoint `{prev_ckpt_wdir.name}`.")
            start = atoms.info["step"]
            steps = run_params["steps"] - start

        # TODO: get seed in ckpt if we have...
        dyn_seed = self.random_seed
        self._print(f"MD Driver's dynamics_seed: {dyn_seed}")

        # The three things below will be different from scratch or restart
        box = atoms.get_cell(complete=True)
        positions = atoms.get_positions()
        velocities = atoms.get_velocities() * units.fs

        dynamics = Simulation(
            self.calc.model_fpath,
            box=box,
            type_idx=type_indices,
            mass=masses,
            initial_position=positions,
            initial_velocity=velocities,
            report_interval=self.setting.dump_period,
            debug=False,
            seed=dyn_seed,
            use_neighbor_list_when_possible=True,
            **init_params,
        )

        run_dynamics(
            chemical_symbols,
            dynamics,
            start=start,
            steps=steps,
            dump_period=self.setting.dump_period,
            ckpt_period=self.setting.ckpt_period,
            ckpt_number=self.setting.ckpt_number,
            wdir=self.directory,
        )

        return

    def _find_latest_checkpoint(self, wdir: pathlib.Path) -> pathlib.Path:
        """"""
        ckpt_dirs = sorted(
            wdir.glob("checkpoint.*"), key=lambda x: int(x.name.split(".")[-1])
        )
        latest_ckpt_dir = ckpt_dirs[-1]

        return latest_ckpt_dir

    def _load_checkpoint(self, ckpt_dir: pathlib.Path) -> Atoms:
        """"""
        atoms = read(ckpt_dir / "structures.xyz", ":")[-1]

        return atoms  # type: ignore

    def _read_a_single_trajectory(
        self,
        wdir: pathlib.Path,
        archive_path: Optional[Union[str, pathlib.Path]] = None,
        *args,
        **kwargs,
    ):
        """"""
        frames = []
        if archive_path is None:
            if (wdir / "traj.xyz").exists():
                frames = read(wdir / "traj.xyz", ":")
            else:
                ...
        else:
            raise RuntimeError("DeepmdJaxDriver does not support archived results.")

        return frames

    def read_trajectory(
        self, archive_path: Optional[Union[str, pathlib.Path]] = None, *args, **kwargs
    ) -> List[Atoms]:
        """"""
        traj_frames = self._aggregate_trajectories(archive_path=archive_path)

        # validate trajectory
        num_frames = len(traj_frames)
        if num_frames > 0:
            ...
        else:  # Calculation did not start properly.
            ...

        return traj_frames


if __name__ == "__main__":
    ...
