#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
import pathlib
import traceback
from typing import List

from ase import Atoms

from .driver import AbstractDriver, Controller, DriverSetting


@dataclasses.dataclass
class AbacusDriverSetting(DriverSetting):

    #: Driver detailed controller setting.
    controller: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """"""

        return

    def get_run_params(self, *args, **kwargs):
        """"""

        run_params = dict()

        return run_params


class AbacusDriver(AbstractDriver):

    name = "abacus"

    default_task = "min"
    supported_tasks = ["scf", "min", "md"]

    def __init__(self, calc, params: dict, directory="./", *args, **kwargs):
        """"""
        super().__init__(calc, params, directory=directory, *args, **kwargs)

        self.setting = AbacusDriverSetting(**params)

        return

    def _verify_checkpoint(self, *args, **kwargs) -> bool:
        """"""
        verified = True
        if self.directory.exists():
            ...
        else:
            verified = False

        return verified

    def _irun(self, atoms: Atoms, ckpt_wdir=None, cache_traj=None, *args, **kwargs):
        """"""
        # NOTE: abacus uses GenericFileIO
        self.calc.directory = pathlib.Path(self.calc.directory)

        try:
            if ckpt_wdir is None:  # start from the scratch
                run_params = self.setting.get_run_params(**kwargs)
                run_params.update(**self.setting.get_init_params())

                self._preprocess_constraints(atoms, run_params)
                ...
            else:
                ...
            atoms.calc = self.calc
            _ = atoms.get_forces()
        except Exception as e:
            self._debug(e)
            self._debug(traceback.print_exc())

        return

    def read_trajectory(self, archive_path=None, *args, **kwargs) -> List[Atoms]:
        """Read trajectory in the current working directory."""
        super().read_trajectory(*args, **kwargs)

        traj_frames = self._aggregate_trajectories(archive_path=archive_path)

        return traj_frames

    def _read_a_single_trajectory(self, wdir, archive_path=None, *args, **kwargs):
        """"""
        from ase.io.abacus import read_abacus_out

        frames = read_abacus_out(
            open(self.directory / "OUT.ABACUS" / "running_scf.log", "r"),
            index=-1,
            non_convergence_ok=True,
        )

        for i, atoms in enumerate(frames):
            atoms.info["step"] = i

        return frames


if __name__ == "__main__":
    ...
