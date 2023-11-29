#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re
from typing import NoReturn

from ..core.variable import Variable, DummyVariable
from ..core.operation import Operation
from ..core.register import registers

from ..potential.manager import AbstractPotentialManager
from ..potential.managers.mixer import MixerManager
from ..potential.trainer import AbstractTrainer
from ..worker.train import TrainerBasedWorker
from ..scheduler.interface import SchedulerVariable
from ..scheduler.scheduler import AbstractScheduler


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


def create_mixer(basic_params, *args, **kwargs):
    """"""
    potters = [basic_params]
    for x in args:
        potters.append(x)
    calc_params = dict(backend="ase", potters=potters)

    mixer = MixerManager()
    mixer.register_calculator(calc_params=calc_params)

    return mixer


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

    
@registers.operation.register
class train(Operation):

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self, dataset, trainer, potter, scheduler=DummyVariable(), size: int=1, 
        init_models=None, active: bool=False, directory="./", *args, **kwargs
    ) -> None:
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

        self._active = active

        return
    
    def forward(
        self, dataset, trainer: AbstractTrainer, 
        scheduler: AbstractScheduler, potter: AbstractPotentialManager
    ):
        """"""
        super().forward()

        init_models = self.init_models
        if self._active:
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print("    >>> Update init_models...")
                prev_wdir = (
                    self.directory.parent.parent / 
                    f"iter.{str(curr_iter-1).zfill(4)}" / 
                    self.directory.name
                )
                prev_mdirs = [] # model dirs
                for p in prev_wdir.iterdir():
                    if p.is_dir() and re.match("m[0-9]+", p.name):
                        prev_mdirs.append(p)
                init_models = [(p/trainer.frozen_name).resolve() for p in prev_mdirs]
                for p in init_models:
                    self._print(" "*8+str(p))
                assert init_models, "No previous models found."
        
        # - 
        if scheduler is None:
            scheduler = SchedulerVariable().value

        # - update dir
        worker = TrainerBasedWorker(trainer, scheduler, directory=self.directory)

        # - run
        manager = None

        _ = worker.run(dataset, size=self.size, init_models=init_models)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            models = worker.retrieve(include_retrieved=True)
            self._print(f"frozen models: {models}")
            potter_params = potter.as_dict()
            potter_params["params"]["model"] = models
            potter.register_calculator(potter_params["params"])
            manager = potter
        else:
            ...
        
        if manager is not None:
            self.status = "finished"

        return manager


if __name__ == "__main__":
    ...