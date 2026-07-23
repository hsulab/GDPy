import copy
from typing import Union

from gdpx.validator import REGISTER as VALIDATOR_REGISTER
from gdpx.validator.validator import BaseValidator
from gdpx.factory.builder import canonicalise_builder
from gdpx.factory.computer import canonicalise_worker


def canonicalise_validator(
    config: Union[dict, BaseValidator],
) -> BaseValidator:
    """Canonicalise the validator configuration."""
    validator = None
    if isinstance(config, dict):
        config = copy.deepcopy(config)
        method = config.pop("method", "minima")
        structures = config.get("structures")
        if structures is not None:
            if isinstance(structures, (list, tuple)):
                resolved = [canonicalise_builder(item) for item in structures]
                if len(resolved) == 1:
                    resolved.append(None)
                if len(resolved) != 2:
                    raise ValueError("Validator requires one or two structure sources.")
                config["structures"] = resolved
            else:
                config["structures"] = canonicalise_builder(structures)
        if config.get("worker") is not None:
            config["worker"] = canonicalise_worker(config["worker"])
        validator = VALIDATOR_REGISTER[method](**config)
    else:
        if not isinstance(config, BaseValidator):
            raise TypeError(f"Validator must be a mapping or BaseValidator, got {type(config).__name__}.")
        validator = config

    return validator
