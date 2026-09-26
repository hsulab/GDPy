"""Compile validated workflow specifications into executable node graphs."""

from __future__ import annotations

import copy
import inspect
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Mapping

from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.registry import workflow_registers as registers

from .configuration import NodeSpec, OutputReference, WorkflowConfigError, WorkflowSpec
from .state import OutputSelector, StateVariable, TargetBarrier

STEP_DIRECTORY_LAYOUT = "topological-v1"
_ORDERED_STEP_DIRECTORY = re.compile(r"^\d{4,}\..+")


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
    definitions = {**spec.resources, **spec.steps}
    for definition in definitions.values():
        for value in definition.inputs.values():
            _validate_output_references(value, spec)
    for definition in spec.state.values():
        _validate_output_references(definition.update, spec)


def _validate_output_references(value: Any, spec: WorkflowSpec) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_output_references(item, spec)
        return
    if not isinstance(value, OutputReference):
        return
    if value.node not in spec.steps:
        raise WorkflowConfigError(f"Named output source {value.node!r} must be a step.")
    cls = _node_class(spec.steps[value.node], "operation")
    output_names = tuple(getattr(cls, "output_names", ()))
    validates_output = getattr(cls, "validates_output", None)
    supported = value.output in output_names
    if validates_output is not None:
        supported = supported or bool(validates_output(spec.steps[value.node], value.output))
    if not supported:
        available = ", ".join(output_names) or "(none)"
        raise WorkflowConfigError(
            f"Step {value.node!r} has no output {value.output!r}; available: {available}."
        )


def _resolve_input(
    value: Any,
    nodes: Mapping[str, Any],
    ports: dict[tuple[str, str], OutputSelector],
) -> Any:
    if isinstance(value, str):
        return nodes[value]
    if isinstance(value, OutputReference):
        key = (value.node, value.output)
        if key not in ports:
            ports[key] = OutputSelector(
                nodes[value.node],
                value.output,
            )
        return ports[key]
    return [_resolve_input(item, nodes, ports) for item in value]


def _references(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, OutputReference):
        return [value.node]
    return [reference for item in value for reference in _references(item)]


def _dependency_order(spec: WorkflowSpec) -> list[str]:
    definitions = {**spec.resources, **spec.steps}
    order: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited:
            return
        if name in spec.state:
            initial = spec.state[name].initial
            if initial is not None:
                visit(initial)
            visited.add(name)
            order.append(name)
            return
        for value in definitions[name].inputs.values():
            for dependency in _references(value):
                visit(dependency)
        visited.add(name)
        order.append(name)

    for name in definitions:
        visit(name)
    return order


def _step_directory_names(spec: WorkflowSpec) -> dict[str, str]:
    remaining = list(spec.steps)
    dependencies = {
        name: {
            dependency
            for value in definition.inputs.values()
            for dependency in _references(value)
            if dependency in spec.steps
        }
        for name, definition in spec.steps.items()
    }
    step_order = []
    resolved = set()
    while remaining:
        name = next(
            (candidate for candidate in remaining if dependencies[candidate] <= resolved),
            None,
        )
        if name is None:
            raise WorkflowConfigError("Workflow step graph contains a cycle.")
        remaining.remove(name)
        resolved.add(name)
        step_order.append(name)
    return {name: f"{index:04d}.{name}" for index, name in enumerate(step_order)}


def _reject_legacy_step_directories(root: pathlib.Path) -> None:
    steps = root / "steps"
    if not steps.is_dir():
        return
    legacy = sorted(
        path.name
        for path in steps.iterdir()
        if path.is_dir() and not _ORDERED_STEP_DIRECTORY.fullmatch(path.name)
    )
    if legacy:
        names = ", ".join(legacy)
        raise WorkflowConfigError(
            f"Legacy step directory layout detected at {steps}: {names}; "
            "use a fresh run directory."
        )


def compile_workflow(
    spec: WorkflowSpec,
    directory: str | pathlib.Path = ".",
    *,
    state_values: Mapping[str, Any] | None = None,
) -> CompiledWorkflow:
    """Instantiate a validated workflow graph."""
    validate_workflow(spec)
    root = pathlib.Path(directory)
    _reject_legacy_step_directories(root)
    definitions = {**spec.resources, **spec.steps}
    step_directories = _step_directory_names(spec)
    nodes: dict[str, Any] = {}
    ports: dict[tuple[str, str], OutputSelector] = {}
    for name in _dependency_order(spec):
        if name in spec.state:
            if state_values is not None and name in state_values:
                value = state_values[name]
            else:
                initial = spec.state[name].initial
                value = None if initial is None else nodes[initial].value
            nodes[name] = StateVariable(value, directory=root / "state" / name)
            continue
        definition = definitions[name]
        category = "variable" if name in spec.resources else "operation"
        cls = _node_class(definition, category)
        kwargs = copy.deepcopy(dict(definition.options))
        kwargs.update(
            {
                key: _resolve_input(value, nodes, ports)
                for key, value in definition.inputs.items()
            }
        )
        if category == "variable":
            kwargs["directory"] = root / "resources" / name
        else:
            kwargs["directory"] = root / "steps" / step_directories[name]
        try:
            nodes[name] = cls(**kwargs)
        except Exception as error:
            raise WorkflowConfigError(f"Failed to construct {name!r} ({definition.type}): {error}") from error
    targets = [nodes[name] for name in spec.settings.targets]
    if len(targets) == 1:
        entry = targets[0]
    else:
        entry = TargetBarrier(targets)
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
    for name, definition in spec.state.items():
        update = _format_reference(definition.update)
        lines.append(f"  {name} [state] <- {definition.initial or 'null'}; next <- {update}")
    for name in _dependency_order(spec):
        if name in spec.state:
            continue
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


def _format_reference(value: str | OutputReference) -> str:
    if isinstance(value, OutputReference):
        return f"{value.node}.{value.output}"
    return value


def workflow_dot(spec: WorkflowSpec) -> str:
    """Return the workflow graph in Graphviz DOT format."""
    validate_workflow(spec)
    lines = ["digraph workflow {"]
    definitions = {**spec.resources, **spec.steps}
    for name, definition in spec.state.items():
        lines.append(f'  "{name}" [label="{name}\\nstate", shape=diamond];')
        if definition.initial is not None:
            lines.append(f'  "{definition.initial}" -> "{name}" [label="initial"];')
        update = definition.update.node if isinstance(definition.update, OutputReference) else definition.update
        label = "next" if isinstance(definition.update, str) else f"next:{definition.update.output}"
        lines.append(f'  "{update}" -> "{name}" [label="{label}", style=dashed];')
    for name, definition in definitions.items():
        shape = "ellipse" if name in spec.resources else "box"
        lines.append(f'  "{name}" [label="{name}\\n{definition.type}", shape={shape}];')
        for value in definition.inputs.values():
            for ref in _references(value):
                lines.append(f'  "{ref}" -> "{name}";')
    lines.append("}")
    return "\n".join(lines)
