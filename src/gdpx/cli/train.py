#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from gdpx.nodes.trainer import TrainerVariable

from .. import config
from ..core.register import registers
from ..utils.command import parse_input_file


def run_trainer(configuration, directory) -> None:
    """"""
    config._print(f"{configuration = }")
    params = parse_input_file(configuration)

    # Instantiate the trainer
    trainer = TrainerVariable(directory=directory, **params["trainer"]).value
    trainer.directory = directory

    # Process the dataset
    name = params["dataset"].get("name", None)
    dataset = registers.create(
        "dataloader", name, convert_name=True, **params["dataset"]
    )

    # Other options
    init_model = params.get("init_model", None)

    # Run the trainer
    trainer.train(dataset, init_model=init_model)

    trainer.freeze()

    return


if __name__ == "__main__":
    ...
  
