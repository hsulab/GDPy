#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
from typing import Union

from gdpx.nodes.validator import canonicalise_validator


def run_validation(config: dict, directory: Union[str, pathlib.Path], potter):
    """ This is a factory to deal with various validations...
    """
    # run over validations
    directory = pathlib.Path(directory)

    tasks = config.get("tasks", [])
    if not tasks:
        raise Exception("No tasks found in the configuration.")

    validators = []
    for task in tasks:
        validator = canonicalise_validator(task)
        validators.append(validator)
    num_validators = len(validators)

    # run the validations sequentially
    for i, validator in enumerate(validators):
        validator.directory = directory/f"v.{i:>02d}.{validator.name}"
        validator.run()

    return


if __name__ == "__main__":
    ...
  
