#!/usr/bin/env python3
# -*- coding: utf-8 -*

from GDPy.utils.command import parse_input_file

"""
"""

def run_task(params, pot_worker=None, ref_worker=None, run=1, report=False):
    """ task = worker + workflow
        GA - population
        MC - TODO: a single driver?
    """
    params = parse_input_file(params)

    task = params.pop("task", None)
    if task == "ga":
        from GDPy.ga.engine import GeneticAlgorithemEngine
        ga = GeneticAlgorithemEngine(params)
        if report:
            ga.report()
        else:
            ga.run(pot_worker, run)
            if ga.read_convergence():
                ga.report()
                if ref_worker:
                    ga.refine(ref_worker)
    # TODO: merge all MC methods togather
    elif task == "mc":
        from GDPy.mc.mc import MonteCarlo
        mc = MonteCarlo(**params)
        mc.run(pot_worker, run)
    elif task == "gcmc":
        from GDPy.mc.gcmc import GCMC
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
    pass