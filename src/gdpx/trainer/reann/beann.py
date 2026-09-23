#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..trainer import BasePotentialTrainer


class BeannTrainer(BasePotentialTrainer):

    name = "beann"
    command = ""
    freeze_command = ""
    prefix = "config"

    def __init__(
        self,
        config: dict,
        type_list: list[str] = None,
        train_epochs: int = 200,
        directory=".",
        command="train",
        freeze_command="freeze",
        random_seed: int = 1112,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            config=config,
            type_list=type_list,
            train_epochs=train_epochs,
            directory=directory,
            command=command,
            freeze_command=freeze_command,
            random_seed=random_seed,
            *args,
            **kwargs,
        )

        return

    def _resolve_train_command(self, *args, **kwargs):
        """python -u /users/jxu/repository/EANN/eann --config ./config.yaml train"""

        return

    def _resolve_freeze_command(self, *args, **kwargs):
        """python -u /users/jxu/repository/EANN/eann --config ./config.yaml freeze EANN.pth -o eann_latest_"""
        return super()._resolve_freeze_command(*args, **kwargs)

    @property
    def frozen_name(self):
        """"""
        return f"{self.name}.pth"

    def write_input(self, dataset, *args, **kwargs):
        """"""

        return

    def read_convergence(self) -> bool:
        """"""

        return


if __name__ == "__main__":
    ...
