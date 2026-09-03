"""Structure-builder compatibility exports."""

from gdpx.builder import REGISTER
from gdpx.builder.builder import StructureBuilder, StructureModifier
from gdpx.factory.builder import canonicalise_builder as create_builder

__all__ = ["REGISTER", "StructureBuilder", "StructureModifier", "create_builder"]

