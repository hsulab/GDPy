#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union


def run_validation(params: dict, directory: Union[str, pathlib.Path], potter):
    """ This is a factory to deal with various validations...
    """
    # run over validations
    directory = pathlib.Path(directory)

    raise NotImplementedError("Command Line Validation is NOT Suppoted.")


if __name__ == "__main__":
    ...
  
