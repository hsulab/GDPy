"""Descriptor compatibility exports."""

from gdpx.describer import REGISTER
from gdpx.describer.describer import BaseDescriber
from gdpx.factory.components import create_describer

__all__ = ["BaseDescriber", "REGISTER", "create_describer"]

