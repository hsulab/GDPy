#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
import re

import omegaconf
import yaml

from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.workflow.factory import create_trainer
from gdpx.providers import ComponentConfig
from gdpx.providers.specs import thaw
from gdpx.providers.training import BasePotentialTrainer
from gdpx.execution.schedulers.scheduler import BaseScheduler
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.variable import DummyVariable, Variable
from gdpx.execution.workers.train import TrainerBasedWorker

from .scheduler import SchedulerVariable


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

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self,
        dataset,
        trainer,
        potential,
        scheduler=DummyVariable(),
        size: int = 1,
        init_models=None,
        active: bool = False,
        share_dataset: bool = False,
        auto_submit: bool = True,
        directory="./",
    ) -> None:
        """"""
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
        if init_models is not None:
            self.init_models = [str(pathlib.Path(p).absolute()) for p in init_models]
        else:
            self.init_models = [None] * self.size
        assert (
            len(self.init_models) == self.size
        ), f"The number of init models {self.init_models} is inconsistent with size {self.size}."

        self._active = active

        self._share_dataset = share_dataset
        self._auto_submit = auto_submit

        return

    def _preprocess_input_nodes(self, input_nodes):
        """"""
        dataset, trainer, scheduler, potential = input_nodes

        if isinstance(scheduler, Variable):
            scheduler = scheduler
        elif isinstance(scheduler, dict) or isinstance(scheduler, omegaconf.DictConfig):
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

        init_models = self.init_models
        if self._active:
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print(">>> Update init_models...")
                prev_wdir = self.directory.parent.parent / f"iter.{str(curr_iter-1).zfill(4)}" / self.directory.name
                prev_mdirs = []  # model dirs
                for p in prev_wdir.iterdir():
                    if p.is_dir() and re.match("m[0-9]+", p.name):
                        prev_mdirs.append(p)
                # TODO: replace `m` with a constant
                init_models = []
                prev_mdirs = sorted(prev_mdirs, key=lambda p: int(p.name[1:]))
                for p in prev_mdirs:
                    trainer.directory = p
                    if hasattr(trainer, "get_checkpoint"):
                        init_models.append(trainer.get_checkpoint())
                    else:
                        init_models.append((p / trainer.frozen_name).resolve())
                for p in init_models:
                    self._print(f"  {str(p)}")
                assert init_models, "No previous models found."

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
            trained_potential = ComponentConfig(
                potential.provider,
                potential.method,
                parameters,
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

    def __init__(self, potential, dst_path=None, directory="./") -> None:
        """"""
        input_nodes = [potential]
        super().__init__(input_nodes, directory)

        if dst_path is not None:
            self.dst_path = pathlib.Path(dst_path).absolute()
            suffix = self.dst_path.suffix
            assert suffix == ".yaml", "dst_path should be either a yaml or a json file."
        else:
            self.dst_path = self._output_path

        return

    def forward(self, potential):
        """"""
        super().forward()

        self._output_path = self.directory / "potential.yaml"
        with open(self._output_path, "w") as fopen:
            yaml.safe_dump(potential.to_dict(), fopen, indent=2)

        if self.dst_path.exists():
            self._print("remove previous potential...")
            self.dst_path.unlink()
        self.dst_path.symlink_to(self._output_path)

        self.status = "finished"

        return potential


if __name__ == "__main__":
    ...
