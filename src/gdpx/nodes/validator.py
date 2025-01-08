#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Union


from gdpx.core.register import registers
from gdpx.core.variable import Variable

from gdpx.validator.validator import BaseValidator


@registers.variable.register
class ValidatorVariable(Variable):

    def __init__(self, directory="./", **kwargs):
        """"""
        # Instantiate a validator
        method = kwargs.pop("method", "minima")
        validator = registers.create("validator", method, convert_name=False, **kwargs)

        # Save the instance
        super().__init__(initial_value=validator, directory=directory)

        return


def canonicalise_validator(config: Union[dict, BaseValidator]) -> BaseValidator:
    """Canonicalise the validator configuration.
    """
    validator = None
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        method = config.pop("method", "minima")
        validator = registers.create("validator", method, convert_name=False, **config)
    else:
        validator = copy.deepcopy(config)

    return validator


if __name__ == "__main__":
    ...
  
