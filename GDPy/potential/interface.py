#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import NoReturn

from GDPy.core.variable import Variable
from GDPy.core.operation import Operation
from GDPy.core.register import registers

from GDPy.potential.register import PotentialRegister
from GDPy.potential.register import create_potter
from GDPy.computation.worker.train import TrainerBasedWorker


@registers.variable.register
class PotterVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        #manager = PotentialRegister()
        name = kwargs.get("name", None)
        #potter = manager.create_potential(pot_name=name)
        #potter.register_calculator(kwargs.get("params", {}))
        #potter.version = kwargs.get("version", "unknown")

        potter = registers.create(
            "manager", name, convert_name=True, 
            #**kwargs.get("params", {})
        )
        potter.register_calculator(kwargs.get("params", {}))

        super().__init__(potter)

        return

@registers.variable.register
class TrainerVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        print("trainer keys: ", kwargs.keys())
        name = kwargs.get("name", None)
        trainer = registers.create(
            "trainer", name, convert_name=True, **kwargs
        )

        super().__init__(trainer)

        return

@registers.variable.register
class WorkerVariable(Variable):

    def __init__(self, **kwargs):
        """"""
        worker = create_potter(**kwargs) # DriveWorker or TrainWorker
        super().__init__(worker)

        return
    
@registers.operation.register
class train(Operation):

    def __init__(self, dataset, trainer, scheduler, size: int=1, *args, **kwargs) -> NoReturn:
        """"""
        input_nodes = [dataset, trainer, scheduler]
        super().__init__(input_nodes)

        self.size = size # number of models

        return
    
    def forward(self, dataset, trainer, scheduler):
        """"""
        super().forward()

        # - update dir
        worker = TrainerBasedWorker(trainer, scheduler, directory=self.directory)

        # - run
        manager = None

        _ = worker.run(dataset, size=self.size)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            models = worker.retrieve(ignore_retrieved=True)
            print("frozen models: ", models)
            manager = registers.create(
                "manager", trainer.name, convert_name=True
            )
            potter_params = dict(
                backend = "lammps",
                command = "lmp -in in.lammps 2>&1 > lmp.out",
                type_list = trainer.type_list,
                model = models
            )
            manager.register_calculator(potter_params)
            print("manager: ", manager.calc)
        else:
            ...
        
        if manager is not None:
            self.status = "finished"

        return manager

if __name__ == "__main__":
    ...