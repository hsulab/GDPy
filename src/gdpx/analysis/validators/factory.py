import copy
from typing import Union

from . import REGISTER as VALIDATOR_REGISTER
from .validator import BaseValidator
from gdpx.structures.builders import canonicalise_builder
from gdpx.execution.factory import create_worker


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
        runtime = config.pop("runtime", None)
        if runtime is not None:
            config["worker"] = create_worker(runtime)
        validator = VALIDATOR_REGISTER[method](**config)
    else:
        if not isinstance(config, BaseValidator):
            raise TypeError(f"Validator must be a mapping or BaseValidator, got {type(config).__name__}.")
        validator = config

    return validator
