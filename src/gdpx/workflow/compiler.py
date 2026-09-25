"""Compile validated workflow specifications into executable node graphs."""

from __future__ import annotations

import copy
import inspect
import pathlib
from dataclasses import dataclass
from typing import Any, Mapping

from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers

from .configuration import NodeSpec, WorkflowConfigError, WorkflowSpec


@dataclass(frozen=True)
class CompiledWorkflow:
    spec: WorkflowSpec
    nodes: Mapping[str, Any]
    entry: Operation


def _node_class(spec: NodeSpec, category: str):
    try:
        return registers.get(category, spec.type, convert_name=(category == "variable"))
    except KeyError as error:
        registry = getattr(registers, category)
        available = ", ".join(sorted(registry.keys())) or "(none)"
        label = "resource" if category == "variable" else "step"
        raise WorkflowConfigError(
            f"Unknown {label} type {spec.type!r} for {spec.name!r}; available: {available}."
        ) from error


def _validate_signature(spec: NodeSpec, category: str) -> None:
    cls = _node_class(spec, category)
    kwargs = {**spec.options, **{name: None for name in spec.inputs}, "directory": pathlib.Path(".")}
    try:
        inspect.signature(cls).bind(**kwargs)
    except TypeError as error:
        raise WorkflowConfigError(f"Invalid {spec.name!r} configuration: {error}") from error


def validate_workflow(spec: WorkflowSpec) -> None:
    """Validate registry types and constructor interfaces without instantiation."""
    for node in spec.resources.values():
        _validate_signature(node, "variable")
    for node in spec.steps.values():
        _validate_signature(node, "operation")


def _resolve_input(value: Any, nodes: Mapping[str, Any]) -> Any:
    if isinstance(value, str):
        return nodes[value]
    return [_resolve_input(item, nodes) for item in value]


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [reference for item in value for reference in _references(item)]


def _dependency_order(spec: WorkflowSpec) -> list[str]:
    definitions = {**spec.resources, **spec.steps}
    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        for value in definitions[name].inputs.values():
            for dependency in _references(value):
                visit(dependency)
        visited.add(name)
        order.append(name)

    for name in definitions:
        visit(name)
    return order


def compile_workflow(spec: WorkflowSpec, directory: str | pathlib.Path = ".") -> CompiledWorkflow:
    """Instantiate a validated workflow graph."""
    validate_workflow(spec)
    root = pathlib.Path(directory)
    definitions = {**spec.resources, **spec.steps}
    nodes: dict[str, Any] = {}
    for name in _dependency_order(spec):
        definition = definitions[name]
        category = "variable" if name in spec.resources else "operation"
        cls = _node_class(definition, category)
        kwargs = copy.deepcopy(dict(definition.options))
        kwargs.update({key: _resolve_input(value, nodes) for key, value in definition.inputs.items()})
        if category == "variable":
            kwargs["directory"] = root / "variables" / name
        else:
            kwargs["directory"] = root / name
        try:
            nodes[name] = cls(**kwargs)
        except Exception as error:
            raise WorkflowConfigError(f"Failed to construct {name!r} ({definition.type}): {error}") from error
    targets = [nodes[name] for name in spec.settings.targets]
    if len(targets) == 1:
        entry = targets[0]
    else:
        barrier_cls = _node_class(NodeSpec("__targets__", "seqrun"), "operation")
        entry = barrier_cls(nodes=targets, directory=root / "__targets__")
    if not isinstance(entry, Operation):
        raise WorkflowConfigError("Workflow targets must resolve to operations.")
    return CompiledWorkflow(spec, nodes, entry)


def workflow_plan(spec: WorkflowSpec) -> str:
    """Return a stable, side-effect-free text plan."""
    validate_workflow(spec)
    definitions = {**spec.resources, **spec.steps}
    lines = [
        f"workflow: {spec.source}",
        f"mode: {spec.settings.mode}",
        f"targets: {', '.join(spec.settings.targets)}",
        "nodes:",
    ]
    for name in _dependency_order(spec):
        definition = definitions[name]
        kind = "resource" if name in spec.resources else "step"
        dependencies = [
            ref
            for value in definition.inputs.values()
            for ref in _references(value)
        ]
        suffix = f" <- {', '.join(dependencies)}" if dependencies else ""
        lines.append(f"  {name} [{kind}:{definition.type}]{suffix}")
    return "\n".join(lines)


def workflow_dot(spec: WorkflowSpec) -> str:
    """Return the workflow graph in Graphviz DOT format."""
    validate_workflow(spec)
    lines = ["digraph workflow {"]
    definitions = {**spec.resources, **spec.steps}
    for name, definition in definitions.items():
        shape = "ellipse" if name in spec.resources else "box"
        lines.append(f'  "{name}" [label="{name}\\n{definition.type}", shape={shape}];')
        for value in definition.inputs.values():
            for ref in _references(value):
                lines.append(f'  "{ref}" -> "{name}";')
    lines.append("}")
    return "\n".join(lines)
