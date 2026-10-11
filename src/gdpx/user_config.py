"""Explicit, named settings from the user's GDPy configuration directory."""

import copy
import os
from pathlib import Path
from collections.abc import Mapping

import yaml


class UserConfigError(ValueError):
    """An invalid or missing user configuration preset."""


def user_config_path():
    home = os.environ.get("XDG_CONFIG_HOME")
    return (Path(home).expanduser() if home else Path.home() / ".config") / "gdpx/config.yaml"


def _merge(base, overrides):
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_scheduler_preset(value):
    """Expand an explicit preset; mappings merge, lists and scalars replace."""
    if isinstance(value, str):
        value = {"preset": value}
    if not isinstance(value, Mapping) or "preset" not in value:
        return copy.deepcopy(value)
    name = value["preset"]
    if not isinstance(name, str) or not name.strip():
        raise UserConfigError("Scheduler preset must be a nonempty name.")
    path = user_config_path()
    try:
        settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise UserConfigError(f"Cannot load scheduler preset {name!r} from {path}: {error}") from error
    if not isinstance(settings, Mapping) or not isinstance(settings.get("schedulers"), Mapping):
        raise UserConfigError(f"{path}: expected a `schedulers` mapping.")
    presets = settings["schedulers"]
    if name not in presets:
        available = ", ".join(sorted(str(key) for key in presets)) or "none"
        raise UserConfigError(f"Unknown scheduler preset {name!r} in {path}; available: {available}.")
    preset = presets[name]
    if not isinstance(preset, Mapping) or "preset" in preset or not preset.get("provider"):
        raise UserConfigError(f"{path}: scheduler preset {name!r} must define a provider and cannot reference another preset.")
    return _merge(preset, {key: item for key, item in value.items() if key != "preset"})
