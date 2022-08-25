#!/usr/bin/env python3
# -*- coding: utf-8 -*

from GDPy.utils.command import parse_input_file

def run_task(params, worker=None, run=1):
    """ task = worker + workflow
        GA - population
        MC - TODO: a single driver?
    """
    params = parse_input_file(params)

    task = params.pop("task", None)
    if task == "ga":
        from GDPy.ga.engine import GeneticAlgorithemEngine
        ga = GeneticAlgorithemEngine(params)
        ga.run()
    elif task == "mc":
        from GDPy.mc.gcmc import GCMC
        gcmc = GCMC(**params)
        gcmc.run(worker, run)
    else:
        raise RuntimeError(f"Cant find task {task}")

    return


if __name__ == "__main__":
    pass