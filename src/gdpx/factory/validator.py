import copy
from typing import Union

from gdpx.core.register import registers
from gdpx.validator.validator import BaseValidator


def canonicalise_validator(
    config: Union[dict, BaseValidator],
) -> BaseValidator:
    """Canonicalise the validator configuration."""
    validator = None
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        method = config.pop("method", "minima")
        validator = registers.create("validator", method, convert_name=False, **config)
    else:
        validator = copy.deepcopy(config)

    return validator
