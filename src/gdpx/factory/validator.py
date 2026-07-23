import copy
from typing import Union

from gdpx.validator import REGISTER as VALIDATOR_REGISTER
from gdpx.validator.validator import BaseValidator


def canonicalise_validator(
    config: Union[dict, BaseValidator],
) -> BaseValidator:
    """Canonicalise the validator configuration."""
    validator = None
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        method = config.pop("method", "minima")
        validator = VALIDATOR_REGISTER[method](**config)
    else:
        validator = copy.deepcopy(config)

    return validator
