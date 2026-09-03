"""Validator compatibility exports."""

from gdpx.factory.validator import canonicalise_validator as create_validator
from gdpx.validator import REGISTER
from gdpx.validator.validator import BaseValidator

__all__ = ["BaseValidator", "REGISTER", "create_validator"]

