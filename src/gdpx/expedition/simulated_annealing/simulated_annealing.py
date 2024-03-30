#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses
import pathlib

from typing import Union

from .. import registers

from ..expedition import AbstractExpedition


class SimulatedAnnealing(AbstractExpedition):

    name: str = "simulated_annealing"

    #: Name of the folder that stores all worker computations.
    comput_dirname: str = "tmp_folder"

    def __init__(
        self,
        builder,
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

        return

    def run(self, *args, **kwargs):
        """"""
        super().run()

        # ---
        structures = self.builder.run()
        nstructures = len(structures)
        self._print(f"Number of input structures: {nstructures}.")

        # ---
        #temperatures = [300, 300]
        temperatures = [300, 300]

        rng_states = []
        for istep in range(2):
            curr_temperature = temperatures[istep]
            if istep > 0:
                # - get each candidates' final rng_states
                prev_structures, prev_rng_states = [], []
                for icand in range(nstructures):
                    prev_atoms, prev_rng_state = self.worker.driver._load_checkpoint(
                        self.directory
                        / self.comput_dirname
                        / f"gen{istep-1}"
                        / f"cand{icand}"
                    )
                    self._print(f"{prev_atoms =}")
                    self._print(f"{prev_rng_state =}")
                    prev_structures.append(prev_atoms)
                    prev_rng_states.append(prev_rng_state)
                structures = prev_structures
                rng_states = prev_rng_states
            else:
                ...
            self._irun(istep, structures, curr_temperature, rng_states)

        return

    def _irun(self, istep: int, structures, temperature: float, rng_states: list):
        """"""
        self._print(f"{temperature =}")

        # - We need a copy of self.worker as we may change some 
        #   of the driver's settings.
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()
        worker = copy.deepcopy(self.worker)
        
        worker.directory = self.directory / self.comput_dirname / f"gen{istep}"

        # - update driver...
        worker.driver.setting.update(temp=temperature)

        # -
        # TODO: NonLocal Scheduler will not save rng_states...
        worker.run(structures, rng_states=rng_states)

        return

    def read_convergence(self) -> bool:
        """"""

        return True

    def get_workers(self):
        """"""

        return


if __name__ == "__main__":
    ...
