#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import List
from gdpx.potential.trainer import AbstractTrainer


class FGPTrainer(AbstractTrainer):

    name = "fgp"

    def __init__(
        self, config: dict, type_list: List[str] = None, train_epochs: int = 200, 
        directory=".", command="train", freeze_command="freeze", random_seed: int = None, 
        *args, **kwargs
    ) -> None:
        """"""
        super().__init__(
            config, type_list, train_epochs, 
            directory, command, freeze_command, 
            random_seed, *args, **kwargs
        )

        return

    @property
    def frozen_name(self):
        """"""
        return f"{self.name}.pb"

    def _resolve_freeze_command(self, *args, **kwargs):
        return super()._resolve_freeze_command(*args, **kwargs)

    def _resolve_train_command(self, *args, **kwargs):
        """"""
        command = self.command

        return command
    
    def train(self, dataset, init_model=None, *args, **kwargs):
        """"""
        self._print("miaow")
        from .representation import train
        train()

        return
    
    def write_input(self, dataset, *args, **kwargs):
        return super().write_input(dataset, *args, **kwargs)

    def read_convergence(self) -> bool:
        return super().read_convergence()


if __name__ == "__main__":
    ...