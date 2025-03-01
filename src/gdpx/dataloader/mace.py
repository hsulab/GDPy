#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union


class MaceDataloader:

    name: str = "mace"

    def __init__(
        self,
        train_file: Union[str, pathlib.Path],
        test_file: Union[str, pathlib.Path],
        batchsize: int,
        directory: Union[str, pathlib.Path] = "./",
        *args,
        **kwargs,
    ) -> None:
        """"""
        self.train_file = pathlib.Path(train_file).resolve()
        self.test_file = pathlib.Path(test_file).resolve()

        self.batchsize = batchsize
        self.directory = pathlib.Path(directory).resolve()

        return

    def as_dict(
        self,
    ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        params["train_file"] = str(self.train_file)
        params["test_file"] = str(self.test_file)
        params["batchsize"] = self.batchsize
        params["directory"] = str(self.directory.resolve())

        return params


if __name__ == "__main__":
    ...
