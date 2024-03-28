#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import itertools

import numpy as np

from ase.io import read, write

from .describer import AbstractDescriber


class SpcDescriber(AbstractDescriber):

    name: str = "spc"

    """Compute some properties.
    """

    def __init__(self, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        return

    def run(self, structures, worker, *args, **kwargs):
        """"""
        self._print(f"{structures = }")
        self._print(f"{worker = }")

        status = "unfinished"

        # -
        cache_structures = self.directory / "cache.xyz"
        if not cache_structures.exists():
            worker.run(
                structures,
            )
            worker.inspect(resubmit=True)
            if worker.get_number_of_running_jobs() == 0:
                results = worker.retrieve(include_retrieved=True)
                computed_structures = list(itertools.chain(*results))
                write(cache_structures, computed_structures)
            else:
                self._print("wait worker to finish...")
                computed_structures = None
        else:
            computed_structures = read(cache_structures, ":")

        # -
        if computed_structures is not None:
            prev_energies = np.array(
                [a.get_potential_energy(apply_constraint=False) for a in structures]
            )
            curr_energies = np.array(
                [
                    a.get_potential_energy(apply_constraint=False)
                    for a in computed_structures
                ]
            )
            abs_ene_err = np.fabs(curr_energies - prev_energies)
            #self._print(f"{abs_ene_err = }")

            prev_forces = np.array(
                [a.get_forces(apply_constraint=False).flatten() for a in structures]
            )
            curr_forces = np.array(
                [a.get_forces(apply_constraint=False).flatten() for a in computed_structures]
            )
            max_frc_err = np.max(np.fabs(curr_forces - prev_forces), axis=1)
            #self._print(f"{max_frc_err = }")

            for i, a in enumerate(structures):
                a.info["abs_ene_err"] = abs_ene_err[i]
                a.info["max_frc_err"] = max_frc_err[i]
            
            status = "finished"
        else:
            ...

        return status


if __name__ == "__main__":
    ...
