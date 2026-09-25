"""Load and validate declarative workflow configuration."""

from __future__ import annotations

import copy
import json
import pathlib
from dataclasses import dataclass, field
from typing import Any, Mapping

import yaml


class WorkflowConfigError(ValueError):
    """A user-facing workflow configuration error."""


@dataclass(frozen=True)
class NodeSpec:
    name: str
    type: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSettings:
    targets: tuple[str, ...]
    mode: str = "once"
    max_iterations: int = 2
    reset_random_state: bool = False
    reset_random_config: tuple[str, int] = ("init", 0)


@dataclass(frozen=True)
class WorkflowSpec:
    source: pathlib.Path
    settings: WorkflowSettings
    parameters: Mapping[str, Any]
    resources: Mapping[str, NodeSpec]
    steps: Mapping[str, NodeSpec]


_TOP_LEVEL_KEYS = {
    "includes",
    "parameters",
    "profiles",
    "resources",
    "steps",
    "workflow",
}
_NODE_KEYS = {"__type__", "inputs", "options"}


def _read_data(path: pathlib.Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise WorkflowConfigError(f"Configuration file not found: {path}") from error
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix not in (".yaml", ".yml"):
        raise WorkflowConfigError(f"Unsupported configuration file type: {path}")
    return yaml.safe_load(text)


def _resolve_files(value: Any, directory: pathlib.Path) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$file"}:
            path = (directory / str(value["$file"])).resolve()
            return _resolve_files(_read_data(path), path.parent)
        return {key: _resolve_files(item, directory) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_files(item, directory) for item in value]
    return value


def _merge_fragments(base: dict, incoming: Mapping[str, Any], source: pathlib.Path) -> None:
    unknown = set(incoming) - _TOP_LEVEL_KEYS
    if "schema_version" in incoming:
        raise WorkflowConfigError(
            f"{source}: workflow files always use the current schema; remove `schema_version`."
        )
    if unknown:
        raise WorkflowConfigError(f"{source}: unknown top-level fields: {', '.join(sorted(unknown))}.")
    for section in ("parameters", "resources", "steps", "profiles"):
        values = incoming.get(section, {}) or {}
        if not isinstance(values, Mapping):
            raise WorkflowConfigError(f"{source}: `{section}` must be a mapping.")
        duplicate = set(base[section]).intersection(values)
        if duplicate:
            raise WorkflowConfigError(
                f"{source}: duplicate {section}: {', '.join(sorted(duplicate))}."
            )
        base[section].update(copy.deepcopy(dict(values)))
    if "workflow" in incoming:
        if base["workflow"] is not None:
            raise WorkflowConfigError(f"{source}: more than one `workflow` section was declared.")
        base["workflow"] = copy.deepcopy(incoming["workflow"])


def _load_fragment(path: pathlib.Path, stack: tuple[pathlib.Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise WorkflowConfigError(f"Workflow include cycle: {chain}")
    data = _read_data(path)
    if not isinstance(data, Mapping):
        raise WorkflowConfigError(f"{path}: workflow configuration must be a mapping.")
    data = _resolve_files(data, path.parent)
    merged = {
        "parameters": {},
        "resources": {},
        "steps": {},
        "profiles": {},
        "workflow": None,
    }
    includes = data.get("includes", []) or []
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise WorkflowConfigError(f"{path}: `includes` must be a list of file paths.")
    for include in includes:
        included_path = (path.parent / include).resolve()
        _merge_fragments(merged, _load_fragment(included_path, (*stack, path)), included_path)
    local = dict(data)
    local.pop("includes", None)
    _merge_fragments(merged, local, path)
    return merged


def _deep_override(current: Any, override: Any, path: str) -> Any:
    if isinstance(override, Mapping):
        if not isinstance(current, Mapping):
            raise WorkflowConfigError(f"Profile override `{path}` does not target a mapping.")
        result = copy.deepcopy(dict(current))
        for key, value in override.items():
            if key not in result:
                raise WorkflowConfigError(f"Profile override targets unknown field `{path}.{key}`.")
            result[key] = _deep_override(result[key], value, f"{path}.{key}")
        return result
    return copy.deepcopy(override)


def _apply_profile(data: dict, profile: str | None) -> None:
    if profile is None:
        return
    profiles = data["profiles"]
    if profile not in profiles:
        raise WorkflowConfigError(
            f"Unknown workflow profile {profile!r}; available: {', '.join(sorted(profiles)) or '(none)'}."
        )
    overlay = profiles[profile]
    if not isinstance(overlay, Mapping):
        raise WorkflowConfigError(f"Profile {profile!r} must be a mapping.")
    unknown = set(overlay) - {"parameters", "resources", "steps"}
    if unknown:
        raise WorkflowConfigError(
            f"Profile {profile!r} has unsupported sections: {', '.join(sorted(unknown))}."
        )
    parameter_overrides = overlay.get("parameters", {}) or {}
    if not isinstance(parameter_overrides, Mapping):
        raise WorkflowConfigError(f"Profile {profile!r} `parameters` must be a mapping.")
    for key, value in parameter_overrides.items():
        if key not in data["parameters"]:
            raise WorkflowConfigError(f"Profile {profile!r} targets unknown parameter {key!r}.")
        data["parameters"][key] = _deep_override(data["parameters"][key], value, f"parameters.{key}")
    for section in ("resources", "steps"):
        node_overrides = overlay.get(section, {}) or {}
        if not isinstance(node_overrides, Mapping):
            raise WorkflowConfigError(f"Profile {profile!r} `{section}` must be a mapping.")
        for name, node_overlay in node_overrides.items():
            if name not in data[section]:
                raise WorkflowConfigError(f"Profile {profile!r} targets unknown {section[:-1]} {name!r}.")
            if not isinstance(node_overlay, Mapping) or set(node_overlay) != {"options"}:
                raise WorkflowConfigError(
                    f"Profile {profile!r} may override only `{section}.{name}.options`."
                )
            current = data[section][name].get("options", {})
            data[section][name]["options"] = _deep_override(
                current, node_overlay["options"], f"{section}.{name}.options"
            )


def _apply_overrides(parameters: dict, overrides: Mapping[str, Any] | None) -> None:
    for dotted, value in (overrides or {}).items():
        parts = dotted.split(".")
        target = parameters
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], Mapping):
                raise WorkflowConfigError(f"Override targets unknown parameter `{dotted}`.")
            target = target[part]
        if not parts or parts[-1] not in target:
            raise WorkflowConfigError(f"Override targets unknown parameter `{dotted}`.")
        target[parts[-1]] = copy.deepcopy(value)


def _resolve_parameters(value: Any, parameters: Mapping[str, Any], path: str) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"$param"}:
            dotted = str(value["$param"])
            current: Any = parameters
            for part in dotted.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    raise WorkflowConfigError(f"{path}: unknown parameter {dotted!r}.")
                current = current[part]
            return copy.deepcopy(current)
        return {key: _resolve_parameters(item, parameters, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_parameters(item, parameters, f"{path}[{index}]") for index, item in enumerate(value)]
    return value


def _parse_node(name: str, value: Any, parameters: Mapping[str, Any], section: str) -> NodeSpec:
    if not isinstance(name, str) or not name:
        raise WorkflowConfigError(f"{section} names must be nonempty strings; got {name!r}.")
    if not isinstance(value, Mapping):
        raise WorkflowConfigError(f"{section}.{name} must be a mapping.")
    unknown = set(value) - _NODE_KEYS
    if unknown:
        raise WorkflowConfigError(f"{section}.{name} has unknown fields: {', '.join(sorted(unknown))}.")
    node_type = value.get("__type__")
    if not isinstance(node_type, str) or not node_type:
        raise WorkflowConfigError(f"{section}.{name}.__type__ must be a nonempty string.")
    inputs = value.get("inputs", {}) or {}
    options = value.get("options", {}) or {}
    if not isinstance(inputs, Mapping) or not isinstance(options, Mapping):
        raise WorkflowConfigError(f"{section}.{name} inputs and options must be mappings.")
    invalid_fields = [key for key in (*inputs, *options) if not isinstance(key, str) or not key]
    if invalid_fields:
        raise WorkflowConfigError(
            f"{section}.{name} input and option names must be nonempty strings."
        )
    overlap = set(inputs).intersection(options)
    if overlap:
        raise WorkflowConfigError(
            f"{section}.{name} fields cannot appear in both inputs and options: "
            f"{', '.join(sorted(overlap))}."
        )
    if "directory" in options or "directory" in inputs:
        raise WorkflowConfigError(f"{section}.{name}.directory is managed by the workflow runner.")
    resolved = _resolve_parameters(options, parameters, f"{section}.{name}.options")
    return NodeSpec(name, node_type, copy.deepcopy(dict(inputs)), resolved)


def _references(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(ref for index, item in enumerate(value) for ref in _references(item, f"{path}[{index}]"))
    raise WorkflowConfigError(f"{path} must be a node name or list of node names.")


def _validate_graph(resources: Mapping[str, NodeSpec], steps: Mapping[str, NodeSpec], targets: tuple[str, ...]) -> None:
    duplicate = set(resources).intersection(steps)
    if duplicate:
        raise WorkflowConfigError(f"Node names must be unique: {', '.join(sorted(duplicate))}.")
    all_nodes = {**resources, **steps}
    dependencies: dict[str, tuple[str, ...]] = {}
    for name, spec in all_nodes.items():
        refs = tuple(
            ref
            for field, value in spec.inputs.items()
            for ref in _references(value, f"{name}.inputs.{field}")
        )
        missing = set(refs) - set(all_nodes)
        if missing:
            raise WorkflowConfigError(f"{name} references unknown nodes: {', '.join(sorted(missing))}.")
        if name in resources:
            invalid = set(refs).intersection(steps)
            if invalid:
                raise WorkflowConfigError(
                    f"Resource {name!r} cannot depend on steps: {', '.join(sorted(invalid))}."
                )
        dependencies[name] = refs
    missing_targets = set(targets) - set(steps)
    if missing_targets:
        raise WorkflowConfigError(f"Unknown workflow targets: {', '.join(sorted(missing_targets))}.")
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            cycle = visiting[visiting.index(name):] + [name]
            raise WorkflowConfigError(f"Workflow dependency cycle: {' -> '.join(cycle)}")
        if name in visited:
            return
        visiting.append(name)
        for dependency in dependencies[name]:
            visit(dependency)
        visiting.pop()
        visited.add(name)

    for name in all_nodes:
        visit(name)


def load_workflow(
    path: str | pathlib.Path,
    *,
    profile: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> WorkflowSpec:
    """Load, compose, and validate one workflow file."""
    source = pathlib.Path(path).resolve()
    data = _load_fragment(source)
    _apply_profile(data, profile)
    _apply_overrides(data["parameters"], overrides)
    workflow = data["workflow"]
    if not isinstance(workflow, Mapping):
        raise WorkflowConfigError(f"{source}: a `workflow` mapping is required.")
    allowed = {"mode", "targets", "max_iterations", "reset_random_state", "reset_random_config"}
    unknown = set(workflow) - allowed
    if unknown:
        raise WorkflowConfigError(f"Unknown workflow fields: {', '.join(sorted(unknown))}.")
    targets_value = workflow.get("targets")
    if isinstance(targets_value, str):
        targets = (targets_value,)
    elif isinstance(targets_value, list) and all(isinstance(item, str) for item in targets_value):
        targets = tuple(targets_value)
    else:
        raise WorkflowConfigError("workflow.targets must be a node name or list of node names.")
    if not targets:
        raise WorkflowConfigError("workflow.targets cannot be empty.")
    if len(set(targets)) != len(targets):
        raise WorkflowConfigError("workflow.targets cannot contain duplicates.")
    mode = workflow.get("mode", "once")
    if mode not in ("once", "repeat"):
        raise WorkflowConfigError("workflow.mode must be `once` or `repeat`.")
    repeat_fields = {"max_iterations", "reset_random_state", "reset_random_config"}
    ignored = repeat_fields.intersection(workflow) if mode == "once" else set()
    if ignored:
        raise WorkflowConfigError(
            f"workflow fields are valid only in `repeat` mode: {', '.join(sorted(ignored))}."
        )
    max_iterations = workflow.get("max_iterations", 2)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise WorkflowConfigError("workflow.max_iterations must be a positive integer.")
    reset_random = workflow.get("reset_random_state", False)
    if not isinstance(reset_random, bool):
        raise WorkflowConfigError("workflow.reset_random_state must be a boolean.")
    reset_config = workflow.get("reset_random_config", ["init", 0])
    if (
        not isinstance(reset_config, (list, tuple))
        or len(reset_config) != 2
        or reset_config[0] not in ("init", "zero")
        or isinstance(reset_config[1], bool)
        or not isinstance(reset_config[1], int)
        or reset_config[1] < 0
    ):
        raise WorkflowConfigError(
            "workflow.reset_random_config must be `[init|zero, nonnegative integer]`."
        )
    resources = {
        name: _parse_node(name, value, data["parameters"], "resources")
        for name, value in data["resources"].items()
    }
    steps = {
        name: _parse_node(name, value, data["parameters"], "steps")
        for name, value in data["steps"].items()
    }
    _validate_graph(resources, steps, targets)
    settings = WorkflowSettings(
        targets=targets,
        mode=mode,
        max_iterations=max_iterations,
        reset_random_state=reset_random,
        reset_random_config=(reset_config[0], reset_config[1]),
    )
    return WorkflowSpec(source, settings, copy.deepcopy(data["parameters"]), resources, steps)
