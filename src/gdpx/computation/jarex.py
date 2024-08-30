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


def prepare_jaxmd_simulation(model, variables, box, lattice_args):
    """Prepare a jax_md simulation with a flax-based NN potential."""

    return


@dataclasses.dataclass
class JarexDriverSetting(DriverSetting):

    ensemble: str = "nve"

    controllder: dict = dataclasses.field(default_factory=dict)

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
        if self.ensemble == "nvt":
            ...
        else:
            ...

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
            # Load flax model and variables, prepre energy_fn
            self.calc.__init_model__()
            # self._print(f"{self.calc._model =}")

            ene_fn = self.calc.get_energy_fn(atoms)

            coord = atoms.get_positions()[self.calc._atype_sort]
            masses = atoms.get_masses()[self.calc._atype_sort] * 1.036427e2
            print(f"{masses.shape =}")

            # Simluation box
            box = np.array(atoms.get_cell(complete=True))

            # NOTE: periodic wrap??
            # dispalce, shift = jax_md.space.periodic(jnp.diag(box))
            dispalce, shift = jax_md.space.periodic(box, wrapped=False)

            # Thermostate
            dt = 0.48
            temp = 350 * 8.61733e-5
            chain_length = 1
            tau = 2000 * dt
            init_fn, apply_fn = jax_md.simulate.nvt_nose_hoover(
                ene_fn, shift, dt, temp, chain_length=chain_length, tau=tau
            )
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
                print(f"{state.position.shape =}")
                print(f"{state.momentum.shape =}")
                print(f"{state.mass.shape =}")
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
