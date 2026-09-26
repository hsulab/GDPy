"""A resolved, executable scientific computation."""

from dataclasses import dataclass
from typing import Any, Sequence

from gdpx.providers.configuration import RuntimeConfig
from gdpx.providers.specs import ModifierSpec, PotentialSpec
from gdpx.modifiers.expression import PotentialExpression, apply_modifiers


@dataclass
class Runtime:
    potential: PotentialSpec
    materialization: Any
    executor: Any
    modifiers: Sequence[ModifierSpec]
    config: RuntimeConfig
    provider_potential: Any = None
    modifier_instances: Sequence[Any] = ()
    scheduler: Any = None

    def run(self, inputs: Any, **kwargs: Any) -> Any:
        return self.executor.run(inputs, **kwargs)

    @property
    def expression(self) -> PotentialExpression:
        return apply_modifiers(self.potential, self.modifiers)
