#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import re

import yaml

from ..core.variable import Variable, DummyVariable
from ..core.operation import Operation
from ..core.register import registers

from ..potential.manager import AbstractPotentialManager
from ..potential.trainer import AbstractTrainer
from ..worker.train import TrainerBasedWorker
from ..scheduler.interface import SchedulerVariable
from ..scheduler.scheduler import AbstractScheduler


@registers.variable.register
class TrainerVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        # print("trainer keys: ", kwargs.keys())
        name = kwargs.get("name", None)
        trainer = registers.create("trainer", name, convert_name=True, **kwargs)

        super().__init__(initial_value=trainer, directory=directory)

        return


@registers.operation.register
class train(Operation):

    #: Whether to actively update some attrs.
    _active: bool = False

    def __init__(
        self,
        dataset,
        trainer,
        potter,
        scheduler=DummyVariable(),
        size: int = 1,
        init_models=None,
        active: bool = False,
        share_dataset: bool = False,
        auto_submit: bool = True,
        directory="./",
        *args,
        **kwargs,
    ) -> None:
        """"""
        input_nodes = [dataset, trainer, scheduler, potter]
        super().__init__(input_nodes=input_nodes, directory=directory)

        assert (
            trainer.value.name == potter.value.name
        ), "Trainer and potter have inconsistent name."
        assert (
            trainer.value.type_list == potter.value.as_dict()["params"]["type_list"]
        ), "Trainer and potter have inconsistent type_list."

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

    def forward(
        self,
        dataset,
        trainer: AbstractTrainer,
        scheduler: AbstractScheduler,
        potter: AbstractPotentialManager,
    ):
        """"""
        super().forward()

        init_models = self.init_models
        if self._active:
            curr_iter = int(self.directory.parent.name.split(".")[-1])
            if curr_iter > 0:
                self._print(">>> Update init_models...")
                prev_wdir = (
                    self.directory.parent.parent
                    / f"iter.{str(curr_iter-1).zfill(4)}"
                    / self.directory.name
                )
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
        manager = None

        _ = worker.run(dataset, size=self.size, init_models=init_models)
        _ = worker.inspect(resubmit=True)
        if worker.get_number_of_running_jobs() == 0:
            models = worker.retrieve(include_retrieved=True)
            self._print("Frozen Models: ")
            for m in models:
                self._print(f"  {str(m) =}")
            potter_params = potter.as_dict()
            potter_params["params"]["model"] = models
            potter.register_calculator(potter_params["params"])
            manager = potter
        else:
            self._print("TrainWorker has not finished.")

        if manager is not None:
            self.status = "finished"
        
        # - some imported packages change `logging.basicConfig`
        #   and accidently add a StreamHandler to logging.root
        #   so remove it...
        import logging
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                logging.root.removeHandler(h)

        return manager


@registers.operation.register
class save_potter(Operation):

    def __init__(self, potter, dst_path=None, directory="./") -> None:
        """"""
        input_nodes = [potter]
        super().__init__(input_nodes, directory)

        if dst_path is not None:
            self.dst_path = pathlib.Path(dst_path).absolute()
            suffix = self.dst_path.suffix
            assert suffix == ".yaml", "dst_path should be either a yaml or a json file."
        else:
            self.dst_path = self._output_path

        return

    def forward(self, potter):
        """"""
        super().forward()

        self._output_path = self.directory / "potter.yaml"
        with open(self._output_path, "w") as fopen:
            yaml.safe_dump(potter.as_dict(), fopen, indent=2)

        if self.dst_path.exists():
            self._print("remove previous potter...")
            self.dst_path.unlink()
        self.dst_path.symlink_to(self._output_path)

        self.status = "finished"

        return potter


if __name__ == "__main__":
    ...
