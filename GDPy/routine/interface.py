#!/usr/bin/env python3
# -*- coding: utf-8 -*

import pathlib

from GDPy.utils.command import parse_input_file

from ..core.operation import Operation


"""
"""

class routine(Operation):

    def __init__(self, routine, scheduler, directory="./") -> None:
        """"""
        input_nodes = [routine, scheduler]
        super().__init__(input_nodes, directory)

        return

    def forward(self, routine, scheduler):
        """Perform a routine and forward results for further analysis.

        Returns:
            Workers that store structures.
        
        """
        super().forward()
        routine.directory = self.directory
        print(routine)
        routine.run()

        return


def run_routine(params, pot_worker=None, directory="./", run=1, report=False):
    """ task = worker + workflow
        GA - population
        MC - TODO: a single driver?
    """
    directory = pathlib.Path(directory)

    params = parse_input_file(params)

    task = params.pop("task", None)
    if task == "ga":
        from GDPy.routine.ga.engine import GeneticAlgorithemEngine
        ga = GeneticAlgorithemEngine(params)
        ga.directory = directory
        if report:
            ga.report()
        else:
            ga.run(pot_worker, run)
            if ga.read_convergence():
                ga.report()
    # TODO: merge all MC methods togather
    elif task == "mc":
        from GDPy.routine.mc import MonteCarlo
        mc = MonteCarlo(**params)
        mc.run(pot_worker, run)
    elif task == "gcmc":
        from GDPy.routine.mc.gcmc import GCMC
        gcmc = GCMC(**params)
        gcmc.run(pot_worker, run)
        # TODO: add report functions to MC simulations
    elif task == "rxn":
        from GDPy.reaction.afir import AFIRSearch
        rxn = AFIRSearch(**params)
        rxn.run(pot_worker)
        ...
    else:
        raise NotImplementedError(f"Cant find task {task}")

    return


if __name__ == "__main__":
    ...