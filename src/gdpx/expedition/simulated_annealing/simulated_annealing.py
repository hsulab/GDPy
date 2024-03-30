#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import pathlib
import time

from typing import Union, List

from .. import registers

from ..expedition import AbstractExpedition


class SimulatedAnnealing(AbstractExpedition):

    name: str = "simulated_annealing"

    #: Name of the folder that stores all worker computations.
    comput_dirname: str = "tmp_folder"

    def __init__(
        self,
        builder,
        temperatures: List[float],
        directory: Union[str, pathlib.Path] = "./",
        random_seed: Union[int, dict] = None,
        *args,
        **kwargs,
    ):
        super().__init__(directory, random_seed, *args, **kwargs)

        # - check system type
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            builder = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else:
            builder = builder
        self.builder = builder

        # -
        self.temperatures = temperatures

        self.num_slices = len(self.temperatures)

        return

    def run(self, *args, **kwargs):
        """"""
        super().run()

        # ---
        structures = self.builder.run()
        nstructures = len(structures)
        self._print(f"Number of input structures: {nstructures}.")

        num_slices = len(self.temperatures)

        # ---
        rng_states = []
        for istep in range(num_slices):
            curr_temperature = self.temperatures[istep]
            if istep > 0:
                # - get each candidates' final rng_states
                prev_structures, prev_rng_states = [], []
                for icand in range(nstructures):
                    curr_wdir = (
                        self.directory
                        / self.comput_dirname
                        / f"gen{istep-1}"
                        / f"cand{icand}"
                    )
                    ckpt_wdir = self.worker.driver._find_latest_checkpoint(curr_wdir)
                    prev_atoms, prev_rng_state = self.worker.driver._load_checkpoint(
                        ckpt_wdir
                    )
                    self._print(f"{prev_atoms =}")
                    self._print(f"{prev_rng_state =}")
                    prev_structures.append(prev_atoms)
                    prev_rng_states.append(prev_rng_state)
                    # --
                    if hasattr(self.worker.driver.calc, "calcs"):
                        for calc in self.worker.driver.calc.calcs:
                            if hasattr(calc, "_load_checkpoint"):
                                calc._load_checkpoint(
                                    ckpt_wdir,
                                    dst_wdir=self.directory
                                    / self.comput_dirname
                                    / f"gen{istep}"
                                    / f"cand{icand}",
                                    start_step=prev_atoms.info["step"],
                                )
                structures = prev_structures
                rng_states = prev_rng_states
            else:
                ...
            step_converged = self._irun(istep, structures, curr_temperature, rng_states)
            if not step_converged:
                self._print(f"Wait Slice {istep} to finish.")
                break
            else:
                ...
        else:
            self._print("SlicedExpedition is converged.")
            with open(self.directory / "FINISHED", "w") as fopen:
                fopen.write(
                    f"FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                )

        return

    def _irun(self, istep: int, structures, temperature: float, rng_states: list):
        """"""
        self._print(f"===== SlicedExpedition Step {istep} =====")
        self._print(f"{temperature =}")

        # - We need a copy of self.worker as we may change some
        #   of the driver's settings.
        worker = self._make_step_worker(istep=istep)

        # - update driver...
        worker.driver.setting.update(temp=temperature)

        # - We need copy some files to continue exploration...
        if istep > 0:
            if hasattr(worker.driver.calc, "calcs"):
                self._print(f"{worker.driver.calc =}")
                for calc in worker.driver.calc.calcs:
                    if hasattr(calc, "_load_checkpoint"):
                        ...
            else:
                ...
        else:
            ...

        # -
        # TODO: NonLocal Scheduler will not save rng_states...
        worker.run(structures, rng_states=rng_states)
        worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            step_converged = True
        else:
            step_converged = False

        return step_converged

    def _make_step_worker(self, istep: int):
        """"""
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()
        worker = copy.deepcopy(self.worker)

        worker.directory = self.directory / self.comput_dirname / f"gen{istep}"

        return worker

    def read_convergence(self) -> bool:
        """"""
        converged = False
        if (self.directory / "FINISHED").exists():
            converged = True

        return converged

    def get_workers(self):
        """"""
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()

        workers = []
        for istep in range(self.num_slices):
            curr_worker = copy.deepcopy(self.worker)
            curr_worker.directory = self.directory / self.comput_dirname / f"gen{istep}"
            workers.append(curr_worker)

        return workers


if __name__ == "__main__":
    ...
