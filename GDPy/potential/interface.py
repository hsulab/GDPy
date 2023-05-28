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

    def __init__(self, directory="./", **kwargs):
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

        super().__init__(initial_value=potter, directory=directory)

        return

@registers.variable.register
class TrainerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        #print("trainer keys: ", kwargs.keys())
        name = kwargs.get("name", None)
        trainer = registers.create(
            "trainer", name, convert_name=True, **kwargs
        )

        super().__init__(initial_value=trainer, directory=directory)

        return

@registers.variable.register
class WorkerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        worker = create_potter(**kwargs) # DriveWorker or TrainWorker
        super().__init__(initial_value=worker, directory=directory)

        return
    
@registers.operation.register
class train(Operation):

    def __init__(
        self, dataset, trainer, scheduler, potter, size: int=1, init_models=None, directory="./", *args, **kwargs
    ) -> NoReturn:
        """"""
        input_nodes = [dataset, trainer, scheduler, potter]
        super().__init__(input_nodes=input_nodes, directory=directory)

        assert trainer.value.name == potter.value.name, "Trainer and potter have inconsistent name."
        assert trainer.value.type_list == potter.value.as_dict()["params"]["type_list"], "Trainer and potter have inconsistent type_list."

        self.size = size # number of models
        if init_models is not None:
            self.init_models = init_models
        else:
            self.init_models = [None]*self.size
        assert len(self.init_models) == self.size, f"The number of init models {self.init_models} is inconsistent with size {self.size}."

        return
    
    def forward(self, dataset, trainer, scheduler, potter):
        """"""
        super().forward()

        # - update dir
        worker = TrainerBasedWorker(trainer, scheduler, directory=self.directory)

        # - run
        manager = None

        _ = worker.run(dataset, size=self.size, init_models=self.init_models)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            models = worker.retrieve(ignore_retrieved=True)
            print("frozen models: ", models)
            #manager = registers.create(
            #    "manager", trainer.name, convert_name=True
            #)
            #potter_params = dict(
            #    backend = "ase",
            #    #backend = "lammps",
            #    #command = "lmp -in in.lammps 2>&1 > lmp.out",
            #    type_list = trainer.type_list,
            #    model = models
            #)
            #manager.register_calculator(potter_params)
            potter_params = potter.as_dict()
            potter_params["params"]["model"] = models
            print("potter: ", potter_params)
            potter.register_calculator(potter_params["params"])
            manager = potter
            print("manager: ", manager.calc)
        else:
            ...
        
        if manager is not None:
            self.status = "finished"

        return manager

if __name__ == "__main__":
    ...