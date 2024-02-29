#!/usr/bin/env python3
# -*- coding: utf-8 -*

import itertools
import pathlib
from typing import NoReturn, List, Union

import numpy as np

from ase.io import read, write

from ..core.register import registers
from ..utils.command import parse_input_file


""""""


def run_newtrainer(configuration, directory):
    """"""
    print("configuration: ", configuration)
    params = parse_input_file(configuration)
    print(params)

    # - create trainer
    name = params["trainer"].get("name", None)
    print("newtrainer: ", params["trainer"])
    trainer = registers.create(
        "trainer", name, convert_name=True, **params["trainer"]
    )
    trainer.directory = directory

    # - create dataset
    name = params["dataset"].get("name", None)
    dataset = registers.create(
        "dataloader", name, convert_name=True, **params["dataset"]
    )

    # - other options
    init_model = params.get("init_model", None)

    trainer.train(dataset, init_model=init_model)

    return


if __name__ == "__main__":
    ...