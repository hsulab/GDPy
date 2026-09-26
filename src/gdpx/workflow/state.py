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


class OutputSelector(Operation):
    """Internal graph node that selects one declared step output."""

    def __init__(self, source, output: str, directory="."):
        super().__init__([source], directory)
        self.output = output

    def forward(self, value):
        super().forward()
        selected = select_output(value, self.output)
        self.status = "finished"
        return selected
