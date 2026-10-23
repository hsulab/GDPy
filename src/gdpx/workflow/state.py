"""Typed values and named outputs used by repeated workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.variable import Variable


class StateVariable(Variable):
    """Expose the state committed by the preceding iteration as a node."""


class NamedOutputs(dict):
    """A step result with statically declared, addressable output names."""


def select_output(value: Any, name: str) -> Any:
    if not isinstance(value, Mapping) or name not in value:
        raise RuntimeError(f"Step result does not contain named output {name!r}.")
    return value[name]


def resolve_output(node, value, name=None):
    """Resolve an explicit head or the producer's declared default."""
    name = name if name is not None else getattr(node, "default_output", None)
    return value if name is None else select_output(value, name)


def validate_outputs(node, value):
    """Catch broken output contracts before downstream operations execute."""
    names = tuple(getattr(node, "output_names", ()))
    if not names:
        if isinstance(value, NamedOutputs) or getattr(node, "default_output", None) is not None:
            raise RuntimeError(f"{type(node).__name__} must declare output_names and default_output.")
        return
    default = getattr(node, "default_output", None)
    if default not in names:
        raise RuntimeError(f"{type(node).__name__} must declare a default_output from output_names.")
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{type(node).__name__} must return a mapping of its declared outputs.")
    missing = set(names) - value.keys()
    extra = value.keys() - set(names)
    if missing or extra:
        raise RuntimeError(f"{type(node).__name__} output mismatch: missing {sorted(missing)}, unexpected {sorted(extra)}.")


class OutputSelector(Operation):
    """Internal graph node that selects one declared step output."""

    receive_raw_outputs = True

    def __init__(self, source, output: str):
        super().__init__([source])
        self.output_name = output

    def forward(self, value):
        selected = select_output(value, self.output_name)
        self.status = "finished"
        return selected


class OutputVariable(Variable):
    """A selected resource value, available during downstream construction."""

    def __init__(self, source, output=None):
        self.source = source
        self.output_name = output
        super().__init__(directory=source.directory)

    @property
    def value(self):
        value = self.source.value
        validate_outputs(self.source, value)
        return resolve_output(self.source, value, self.output_name)


class TargetBarrier(Operation):
    """Internal virtual node that joins multiple workflow targets."""

    def __init__(self, targets):
        super().__init__(targets)

    def forward(self, *outputs):
        self.status = "finished"
        return outputs
