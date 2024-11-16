#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import functools
import pathlib
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from ase import Atoms, units
from ase.io import read, write
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from .driver import AbstractDriver, Controller, DriverSetting

#: This makes the kinetic energy has a unit of eV and velocity Ang/fs.
MASS_CONVERTOR: float = 1.036427e2


def convert_data_to_atoms(symbols, box, coord, velocity, pbc, rsort) -> Atoms:
    """"""
    velocity = velocity / units.fs
    atoms = Atoms(symbols, positions=coord, velocities=velocity, cell=box)
    atoms.pbc = pbc
    atoms = atoms[rsort]

    return atoms


@functools.partial(jax.jit, static_argnames="ene_fn")
def get_quantity(state, ene_fn, **kwargs):
    """"""
    potential_energy = ene_fn(state.position, **kwargs)
    kinetic_energy = jax_md.quantity.temperature(
        velocity=state.velocity, mass=state.mass
    )  # eV
    temperature = (
        jax_md.quantity.temperature(velocity=state.velocity, mass=state.mass)  # eV
        / 1.380649e-23
        * 1.602176634e-19
    )

    return temperature, potential_energy, kinetic_energy


@dataclasses.dataclass
class NoseHooverChainThermostat(Controller):

    name: str = "nose_hoover"

    timestep: float = 1.0

    temperature: float = 300.0

    def __post_init__(self):
        """"""
        taut = self.params.get("Tdamp", 100.0)  # fs
        assert taut is not None

        chain_length = self.params.get("chain_length", 1)
        assert chain_length is not None

        from jax_md.simulate import nvt_nose_hoover as driver_cls

        self.conv_params = dict(
            driver_cls=driver_cls,
            dt=self.timestep,
            kT=self.temperature * 8.61733e-5,  # Kelvin -> kbT in eV
            tau=taut,
            chain_length=chain_length,
        )

        return


controllers = dict(nose_hoover_nvt=NoseHooverChainThermostat)

default_controllers = dict(nvt=NoseHooverChainThermostat)


@dataclasses.dataclass
class JarexDriverSetting(DriverSetting):

    ensemble: str = "nve"

    controller: dict = dataclasses.field(default_factory=dict)

    driver_cls: Optional[jax_md.simulate.Simulator] = None

    fix_com: bool = False

    def __post_init__(self):
        """"""
        _init_params = {}
        if self.task == "md":
            suffix = self.ensemble
            _init_params.update(
                timestep=self.timestep, temperature=self.temp
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
        self.driver_cls = cont.conv_params.pop("driver_cls")

        _init_params.update(**cont.conv_params)
        self._internals.update(**_init_params)

        return

    def get_run_params(self, *args, **kwargs) -> dict:
        """"""
        run_params = dict(
            steps=kwargs.get("steps", self.steps),
            constraint=kwargs.get("constraint", self.constraint),
        )

        return run_params


class JarexDriver(AbstractDriver):

    #: Driver's name.
    name: str = "jax"

    #: Default Task.
    default_task: str = "spc"

    #: Supported tasks.
    supported_tasks: List[str] = ["spc", "min", "md"]

    #: Class for setting.
    setting_cls: type[DriverSetting] = JarexDriverSetting

    def _irun(
        self,
        atoms: Atoms,
        ckpt_wdir: Optional[Union[str, pathlib.Path]] = None,
        *args,
        **kwargs,
    ):
        """Run the simulation."""
        if ckpt_wdir is None:
            if self.setting.task != "md":
                raise RuntimeError("Driver backend jax only supports md for now.")
            # Load flax-based prepre energy_fn
            ene_fn, a_sort, a_rsort = self.calc.prepare_jaxmd_simulation(atoms)

            symbols = (np.array(atoms.get_chemical_symbols())[a_sort]).tolist()
            coord = atoms.get_positions()[a_sort]

            # convert masses to make velocty'unit Ang/fs
            masses = atoms.get_masses()[a_sort] * MASS_CONVERTOR

            # Simluation box
            box = np.array(atoms.get_cell(complete=True))

            # NOTE: periodic wrap??
            # dispalce, shift = jax_md.space.periodic(jnp.diag(box))
            dispalce, shift = jax_md.space.periodic(box, wrapped=False)

            # Thermostate
            init_params = self.setting.get_init_params()

            if self.setting.driver_cls is not None:
                init_fn, apply_fn = self.setting.driver_cls(
                    ene_fn, shift, **init_params
                )
            else:
                raise RuntimeError()  # This should not happen.

            # As jax-md init velocities by Maxwell-Boltzmann and
            # center the momenta, we redo everything by ourselves.
            state = init_fn(jax.random.PRNGKey(0), coord, mass=masses, nbrs_nm=None)

            self._prepare_velocities(
                atoms,
                velocity_seed=self.setting.velocity_seed,
                ignore_atoms_velocities=self.setting.ignore_atoms_velocities,
            )
            print(f"{atoms.get_temperature() =}")

            # velocities = (atoms.get_velocities() * units.fs)[a_sort]
            # state = state.set(momentum=state.mass*velocities)
            momenta = (atoms.get_momenta() * MASS_CONVERTOR * units.fs)[a_sort]
            state = state.set(momentum=momenta)

            with open(self.directory / "traj.xyz", "w") as fopen:
                fopen.write("")
            with open(self.directory / "dyn.log", "w") as fopen:
                fopen.write(
                    f"# {'Time[ps]':<10s}  {'Etot[eV]':>12s}  {'Epot[eV]':>12s}  {'Ekin[eV]':>12s}  {'T[K]':>12s}\n"
                )

            run_params = self.setting.get_run_params()
            for i in range(run_params["steps"]):
                # check quantity
                temp, pot_ene, kin_ene = get_quantity(state, ene_fn, nbrs_nm=None)
                print(f"{i}  {temp =} {pot_ene =} {kin_ene =}")
                # save structure
                curr_atoms = convert_data_to_atoms(
                    symbols, box, state.position, state.velocity, atoms.pbc, a_rsort
                )
                curr_atoms.info["step"] = i
                write(self.directory / "traj.xyz", curr_atoms, append=True)
                with open(self.directory / "dyn.log", "a") as fopen:
                    fopen.write(
                        f"{i*self.setting.timestep/1000.:<12.4f}  {pot_ene+kin_ene:>12.4f}  {pot_ene:>12.4f}  {kin_ene:>12.4f}  {temp:>12.4f}\n"
                    )
                # perform dynamics
                state = apply_fn(state, nbrs_nm=None)
        else:
            raise NotImplementedError("Driver backend Jax does not support restart.")

        return

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
            ...

        return frames

    def read_trajectory(
        self, archive_path: Optional[Union[str, pathlib.Path]] = None, *args, **kwargs
    ) -> List[Atoms]:
        """"""
        traj_frames = self._aggregate_trajectories(archive_path=archive_path)

        # add extra atoms info

        # validate trajectory
        num_frames = len(traj_frames)
        if num_frames > 0:
            ...
        else:  # Calculation did not start properly.
            ...

        return traj_frames


if __name__ == "__main__":
    ...
