#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from collections.abc import Mapping

import yaml

from gdpx.execution.schedulers.scheduler import BaseScheduler
from gdpx.execution.workers.train import TrainerBasedWorker
from gdpx.providers import ComponentConfig, PotentialConfig
from gdpx.providers.specs import thaw
from gdpx.providers.training import BasePotentialTrainer
from gdpx.workflow.factory import create_trainer
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.session.variable import DummyVariable, Variable

from .scheduler import SchedulerVariable
from .runtime import PotentialVariable


@registers.variable.register
class TrainerVariable(Variable):

    def __init__(self, provider, method="default", parameters=None, directory="./"):
        """"""
        self.config = ComponentConfig(provider, method, parameters or {})
        trainer = create_trainer(self.config)

        super().__init__(initial_value=trainer, directory=directory)

        return

    def as_dict(self) -> dict:
        return self.config.to_dict()


@registers.operation.register
class train(Operation):

    def __init__(
        self,
        dataset,
        trainer,
        potential=None,
        scheduler=DummyVariable(),
        size: int = 1,
        share_dataset: bool = False,
        auto_submit: bool = True,
        directory="./",
    ) -> None:
        """Train from scratch unless an initial potential is supplied."""
        if potential is None:
            potential = PotentialVariable(
                provider=trainer.config.provider,
                parameters={
                    "type_list": list(trainer.value.type_list),
                    "estimate_uncertainty": size > 1,
                },
                directory=directory,
            )
        input_nodes = [dataset, trainer, scheduler, potential]
        super().__init__(input_nodes=input_nodes, directory=directory)

        if not isinstance(potential.value, ComponentConfig):
            raise TypeError("train requires a PotentialVariable.")
        if trainer.config.provider != potential.value.provider:
            raise ValueError("Trainer and potential providers must match.")
        type_list = potential.value.parameters.get("type_list")
        if type_list is not None and trainer.value.type_list != list(type_list):
            raise ValueError("Trainer and potential type lists must match.")

        self.size = size  # number of models
        self._share_dataset = share_dataset
        self._auto_submit = auto_submit

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        dataset, trainer, scheduler, potential = input_nodes

        if isinstance(scheduler, Variable):
            scheduler = scheduler
        elif isinstance(scheduler, Mapping):
            scheduler_params = copy.deepcopy(scheduler)
            scheduler = SchedulerVariable(directory=self.directory, **scheduler_params)
        else:
            raise RuntimeError(f"Unknown {scheduler} for the scheduler.")

        return dataset, trainer, scheduler, potential

    def forward(
        self,
        dataset,
        trainer: BasePotentialTrainer,
        scheduler: BaseScheduler,
        potential: ComponentConfig,
    ):
        """"""
        super().forward()

        init_models = list(potential.parameters.get("model", ()))
        if not init_models:
            init_models = [None] * self.size
        if len(init_models) != self.size:
            raise ValueError(
                f"Potential provides {len(init_models)} initial models but training size is {self.size}."
            )

        # -
        if scheduler is None:
            scheduler = SchedulerVariable().value

        # - update dir
        worker = TrainerBasedWorker(
            trainer,
            scheduler,
            share_dataset=self._share_dataset,
            auto_submit=self._auto_submit,
            directory=self.directory,
        )

        # - run
        trained_potential = None

        _ = worker.run(dataset, size=self.size, init_models=init_models)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            models = worker.retrieve(include_retrieved=True)
            self._print("Frozen Models: ")
            for m in models:
                self._print(f"  {str(m) =}")
            parameters = thaw(potential.parameters)
            parameters["model"] = models
            trained_potential = PotentialConfig(
                potential.provider,
                potential.method,
                parameters,
                potential.backend,
            )
        else:
            self._print("TrainWorker has not finished.")

        if trained_potential is not None:
            self.status = "finished"

        # - some imported packages change `logging.basicConfig`
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        import logging

        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                logging.root.removeHandler(h)

        return trained_potential


@registers.operation.register
class save_potential(Operation):

    def __init__(self, potential, directory="./") -> None:
        """"""
        input_nodes = [potential]
        super().__init__(input_nodes, directory)

        return

    def forward(self, potential):
        """"""
        super().forward()

        self._output_path = self.directory / "potential.yaml"
        with open(self._output_path, "w") as fopen:
            yaml.safe_dump(potential.to_dict(), fopen, indent=2)

        self.status = "finished"

        return potential


if __name__ == "__main__":
    ...
