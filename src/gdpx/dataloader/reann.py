#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union


class ReannDataloader:

    name: str = "reann"

    def __init__(self, batchsize: int, directory: Union[str, pathlib.Path] = "./") -> None:
        """"""
        self.batchsize = batchsize
        self.directory = pathlib.Path(directory).resolve()

        return

    def as_dict(
        self,
    ) -> dict:
        """"""
        params = {}
        params["name"] = self.name
        params["batchsize"] = self.batchsize
        params["directory"] = str(self.directory.resolve())

        return params


if __name__ == "__main__":
    ...
