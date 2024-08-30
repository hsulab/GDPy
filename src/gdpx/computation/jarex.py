#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
from typing import List, Optional

import jax
import jax.numpy as jnp
import jax_md
import numpy as np
from ase import Atoms

from .driver import AbstractDriver, Controller, DriverSetting


@dataclasses.dataclass
class JarexDriverSetting(DriverSetting):

    ensemble: str = "nve"

    controller: dict = dataclasses.field(default_factory=dict)

    driver_cls: Optional[jax_md.simulate.Simulator] = None

    fix_cm: bool = False

    fmax: float = 0.05

    def __post_init__(self):
        """"""
        if self.task == "spc":
            ...
        elif self.task == "md":
            self.__parse_molecular_dynamics__()
        else:
            raise RuntimeError(f"Failed to parse task {self.task}.")

        return

    def __parse_molecular_dynamics__(self):
        """"""
        self._internals.update(dt=self.timestep)
        if self.ensemble == "nvt":
            temperature = self.temp * 8.61733e-5  # Kelvin -> kbT in eV
            self._internals.update(kT=temperature)
            if self.controller:
                thermo_cls_name = self.controller["name"] + "_" + self.ensemble
                thermo_cls = controllers[thermo_cls_name]
            else:
                thermo_cls = NoseHooverChainThermostat
            thermostat = thermo_cls(**self.controller)
            if thermostat.name == "nose_hoover":
                from jax_md.simulate import nvt_nose_hoover as driver_cls
            else:
                raise RuntimeError(f"Unknown thermostat {thermostat}.")
            thermo_params = thermostat.conv_params
            _init_md_params = dict()
            _init_md_params.update(**thermo_params)
        else:
            raise NotImplementedError(f"Unimplemented ensemble {self.ensemble}")

        self.driver_cls = driver_cls
        self._internals.update(**_init_md_params)

        return

    def get_run_params(self, *args, **kwargs) -> dict:
        """"""
        run_params = dict(
            steps=kwargs.get("steps", self.steps),
            constraint=kwargs.get("constraint", self.constraint),
        )

        return run_params


@dataclasses.dataclass
class NoseHooverChainThermostat(Controller):

    name: str = "nose_hoover"

    def __post_init__(self):
        """"""
        taut = self.params.get("Tdamp", 100.0)  # fs
        assert taut is not None

        chain_length = self.params.get("chain_length", 1)
        assert chain_length is not None

        self.conv_params = dict(tau=taut, chain_length=chain_length)

        return


controllers = dict(nose_hoover_nvt=NoseHooverChainThermostat)


class JarexDriver(AbstractDriver):

    #: Driver's name.
    name: str = "jax"

    #: Default Task.
    default_task: str = "spc"

    #: Supported tasks.
    supported_tasks: List[str] = ["spc", "min", "md"]

    def __init__(self, calc, params: dict, *args, **kwargs):
        """"""
        super().__init__(calc=calc, params=params, *args, **kwargs)

        self.setting: JarexDriverSetting = JarexDriverSetting(**params)

        return

    def _irun(
        self,
        atoms: Atoms,
        ckpt_wdir: Optional[None] = None,
        *args,
        **kwargs,
    ):
        """Run the simulation."""
        if ckpt_wdir is None:
            # Load flax-based prepre energy_fn
            ene_fn = self.calc.prepare_jaxmd_simulation(atoms)

            coord = atoms.get_positions()[self.calc._atype_sort]
            masses = atoms.get_masses()[self.calc._atype_sort] * 1.036427e2

            # Simluation box
            box = np.array(atoms.get_cell(complete=True))

            # NOTE: periodic wrap??
            # dispalce, shift = jax_md.space.periodic(jnp.diag(box))
            dispalce, shift = jax_md.space.periodic(box, wrapped=False)

            # Thermostate
            init_params = self.setting.get_init_params()

            if self.setting.driver_cls is not None:
                init_fn, apply_fn = self.setting.driver_cls(ene_fn, shift, **init_params)
            else:
                raise RuntimeError()  # This should not happen.

            state = init_fn(jax.random.PRNGKey(0), coord, mass=masses, nbrs_nm=None)

            @jax.jit
            def get_quantity(state, nbrs_nm=None):
                """"""
                potential_energy = ene_fn(state.position, nbrs_nm)
                kinetic_energy = jax_md.quantity.temperature(
                    velocity=state.velocity, mass=state.mass
                )
                temperature = (
                    jax_md.quantity.temperature(
                        velocity=state.velocity, mass=state.mass
                    )
                    / 1.380649e-23
                    * 1.602176634e-19
                )

                return temperature, potential_energy, kinetic_energy

            for i in range(10):
                temp, pot_ene, kin_ene = get_quantity(state, nbrs_nm=None)
                print(f"{i}  {temp =} {pot_ene =} {kin_ene =}")
                # print(f"{state.position.shape =}")
                # print(f"{state.momentum.shape =}")
                # print(f"{state.mass.shape =}")
                state = apply_fn(state, nbrs_nm=None)
        else:
            ...

        return

    def read_trajectory(self, *args, **kwargs) -> List[Atoms]:
        """"""
        traj_frames = []

        return traj_frames


if __name__ == "__main__":
    ...
