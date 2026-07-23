"""Registries belonging specifically to the workflow graph layer."""

from gdpx.core.registry import Registry


VARIABLE_REGISTRY = Registry("variable")
OPERATION_REGISTRY = Registry("operation")
PLACEHOLDER_REGISTRY = Registry("placeholder")


class workflow_registers:
    variable = VARIABLE_REGISTRY
    operation = OPERATION_REGISTRY
    placeholder = PLACEHOLDER_REGISTRY

    @classmethod
    def get(cls, category: str, name: str, convert_name: bool = True):
        if convert_name:
            name = "".join(part.capitalize() for part in name.strip().split("_")) + category.capitalize()
        return getattr(cls, category)[name]

    @classmethod
    def create(cls, category: str, name: str, convert_name: bool = True, *args, **kwargs):
        return cls.get(category, name, convert_name)(*args, **kwargs)


# Populate the legacy catalog only when the workflow layer is imported.  This
# keeps importing gdpx.core.register independent of gdpx.session.
from gdpx.core.catalog import registers

registers.variable = VARIABLE_REGISTRY
registers.operation = OPERATION_REGISTRY
registers.placeholder = PLACEHOLDER_REGISTRY
