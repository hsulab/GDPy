"""Transactional persistence for loop-carried workflow state."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Mapping
from typing import Any

import yaml

from gdpx.data.loaders.factory import create_dataloader
from gdpx.providers import PotentialConfig
from gdpx.providers.specs import thaw

from .configuration import OutputReference, WorkflowSpec

FORMAT_VERSION = 1


def _reference_data(value):
    if isinstance(value, OutputReference):
        return {"node": value.node, "output": value.output}
    if isinstance(value, Mapping):
        return {key: _reference_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_reference_data(item) for item in value]
    return value


def workflow_fingerprint(spec: WorkflowSpec) -> str:
    """Hash the fully resolved workflow that owns a state store."""
    payload = {
        "settings": {
            "mode": spec.settings.mode,
            "targets": list(spec.settings.targets),
            "max_iterations": spec.settings.max_iterations,
            "reset_random_state": spec.settings.reset_random_state,
            "reset_random_config": list(spec.settings.reset_random_config),
        },
        "parameters": thaw(spec.parameters),
        "state": {
            name: {
                "initial": value.initial,
                "update": _reference_data(value.update),
            }
            for name, value in spec.state.items()
        },
        "resources": {
            name: {
                "type": value.type,
                "inputs": _reference_data(dict(value.inputs)),
                "options": thaw(value.options),
            }
            for name, value in spec.resources.items()
        },
        "steps": {
            name: {
                "type": value.type,
                "inputs": _reference_data(dict(value.inputs)),
                "options": thaw(value.options),
            }
            for name, value in spec.steps.items()
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _artifact(path: str | pathlib.Path, root: pathlib.Path) -> str:
    path = pathlib.Path(path).resolve()
    if not path.exists():
        raise RuntimeError(f"Workflow state artifact is missing: {path}")
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _restore_artifact(value: str | Mapping[str, Any], root: pathlib.Path) -> str:
    if isinstance(value, Mapping):
        path = pathlib.Path(value["path"])
        is_relative = value.get("relative", not path.is_absolute())
    else:
        path = pathlib.Path(value)
        is_relative = not path.is_absolute()
    if is_relative:
        path = root / path
    path = path.resolve()
    if not path.exists():
        raise RuntimeError(f"Workflow state artifact is missing: {path}")
    return str(path)


def _write_yaml_atomic(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = pathlib.Path(stream.name)
        yaml.safe_dump(dict(value), stream, sort_keys=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_version(iteration: int | None) -> str:
    return "initial" if iteration is None or iteration < 0 else f"{iteration:04d}"


def _publish_potential(value, root: pathlib.Path, state_name: str, iteration: int | None) -> pathlib.Path:
    data = value.to_dict()
    if "model" in data.get("parameters", {}):
        models = data["parameters"]["model"]
        models = [models] if isinstance(models, (str, pathlib.Path)) else list(models)
        data["parameters"]["model"] = [_artifact(path, root) for path in models]
    manifest = {
        "format": "gdpx.model/v1",
        "iteration": iteration,
        **data,
    }
    version = _artifact_version(iteration)
    base = root / "artifacts" / "models" / state_name
    target = base / (f"{version}.yaml" if version == "initial" else f"iterations/{version}.yaml")
    _write_yaml_atomic(target, manifest)
    return target


def _dataset_manifest(snapshot, root: pathlib.Path, iteration: int | None) -> dict:
    systems = {}
    for shard in snapshot.shards:
        path = pathlib.Path(shard["path"])
        if any(_is_within(path, source) for source in snapshot.sources):
            continue
        item = _artifact(path, root)
        systems.setdefault("/".join(shard["system"]), []).append(item)
    return {
        "format": "gdpx.structure-dataset/v1",
        "iteration": iteration,
        "codec": "extxyz",
        "loader": {
            "batchsize": snapshot.batchsize,
            "train_ratio": snapshot.train_ratio,
            "random_seed": snapshot.random_seed,
            "prop_keys": snapshot.prop_keys,
            "sources": [_artifact(path, root) for path in snapshot.sources],
        },
        "systems": systems,
    }


def _is_within(path: pathlib.Path, directory: pathlib.Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _publish_dataset(snapshot, root: pathlib.Path, state_name: str, iteration: int | None) -> pathlib.Path:
    target = (
        root
        / "artifacts"
        / "datasets"
        / state_name
        / "versions"
        / f"{_artifact_version(iteration)}.yaml"
    )
    _write_yaml_atomic(target, _dataset_manifest(snapshot, root, iteration))
    return target


def encode_state(
    value: Any,
    root: pathlib.Path,
    state_name: str | None = None,
    iteration: int | None = None,
) -> dict:
    from gdpx.data.loaders.dataset import XyzDataloader, XyzSnapshotDataloader
    from gdpx.exploration.continuation import ExplorationContinuation

    if isinstance(value, PotentialConfig):
        if state_name is not None:
            manifest = _publish_potential(value, root, state_name, iteration)
            return {"kind": "potential_artifact", "artifact": _artifact(manifest, root)}
        data = value.to_dict()
        if "model" in data.get("parameters", {}):
            models = data["parameters"]["model"]
            models = [models] if isinstance(models, (str, pathlib.Path)) else list(models)
            data["parameters"]["model"] = [_artifact(path, root) for path in models]
        return {"kind": "potential", "value": data}
    if isinstance(value, XyzDataloader):
        snapshot = XyzSnapshotDataloader.from_loader(value)
        if state_name is not None:
            manifest = snapshot.manifest
            if manifest is None or not manifest.exists():
                manifest = _publish_dataset(snapshot, root, state_name, iteration)
            return {"kind": "dataset_artifact", "artifact": _artifact(manifest, root)}
        data = snapshot.as_dict()
        data["sources"] = [_artifact(path, root) for path in data["sources"]]
        data["shards"] = [
            {**item, "path": _artifact(item["path"], root)} for item in data["shards"]
        ]
        return {"kind": "xyz_snapshot", "value": data}
    if isinstance(value, ExplorationContinuation):
        return {
            "kind": "exploration_continuation",
            "value": {
                "provider": value.provider,
                "metadata": thaw(value.metadata),
                "artifacts": [_artifact(path, root) for path in value.artifacts],
            },
        }
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, ExplorationContinuation) for item in value
    ):
        return {
            "kind": "exploration_continuations",
            "value": [encode_state(item, root)["value"] for item in value],
        }
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return {"kind": "json", "value": value}
    raise TypeError(
        f"Workflow state cannot persist {type(value).__name__}; register a typed state codec."
    )


def decode_state(record: Mapping[str, Any], root: pathlib.Path) -> Any:
    kind = record["kind"]
    value = record.get("value")
    if kind == "json":
        return value
    if kind == "potential":
        data = dict(value)
        parameters = dict(data.get("parameters", {}))
        if "model" in parameters:
            parameters["model"] = [_restore_artifact(item, root) for item in parameters["model"]]
        return PotentialConfig(
            data["provider"], data.get("method"), parameters, data.get("backend")
        )
    if kind == "potential_artifact":
        manifest_path = pathlib.Path(_restore_artifact(record["artifact"], root))
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if data.get("format") != "gdpx.model/v1":
            raise RuntimeError(f"Unsupported model artifact format in {manifest_path}.")
        parameters = dict(data.get("parameters", {}))
        if "model" in parameters:
            parameters["model"] = [
                _restore_artifact(item, root) for item in parameters["model"]
            ]
        return PotentialConfig(
            data["provider"], data.get("method"), parameters, data.get("backend")
        )
    if kind == "xyz_snapshot":
        data = dict(value)
        data["sources"] = [_restore_artifact(item, root) for item in data["sources"]]
        data["shards"] = [
            {**item, "path": _restore_artifact(item["path"], root)}
            for item in data.get("shards", [])
        ]
        return create_dataloader(data)
    if kind == "dataset_artifact":
        manifest_path = pathlib.Path(_restore_artifact(record["artifact"], root))
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "gdpx.structure-dataset/v1":
            raise RuntimeError(f"Unsupported dataset artifact format in {manifest_path}.")
        loader = dict(manifest.get("loader", {}))
        loader["sources"] = [
            _restore_artifact(item, root) for item in loader.get("sources", [])
        ]
        shards = []
        for system, items in manifest.get("systems", {}).items():
            for item in items:
                shards.append(
                    {
                        "system": tuple(system.split("/")),
                        "path": _restore_artifact(item, root),
                    }
                )
        loader.update(
            {
                "name": "xyz_snapshot",
                "shards": shards,
                "manifest": str(manifest_path),
            }
        )
        return create_dataloader(loader)
    if kind == "exploration_continuation":
        from gdpx.exploration.continuation import ExplorationContinuation

        return ExplorationContinuation(
            value["provider"],
            value.get("metadata", {}),
            tuple(_restore_artifact(item, root) for item in value.get("artifacts", [])),
        )
    if kind == "exploration_continuations":
        return tuple(
            decode_state({"kind": "exploration_continuation", "value": item}, root)
            for item in value
        )
    raise RuntimeError(f"Unsupported workflow state kind {kind!r}.")


class WorkflowStateStore:
    """Own atomic, append-only state manifests for one workflow run."""

    def __init__(self, root: str | pathlib.Path, spec: WorkflowSpec):
        self.root = pathlib.Path(root)
        self.directory = self.root / "state"
        self.iterations = self.directory / "iterations"
        self.initial = self.directory / "initial.yaml"
        self.current = self.directory / "current.yaml"
        self.fingerprint = workflow_fingerprint(spec)

    def load(self) -> tuple[int, dict[str, Any]]:
        if not self.current.exists():
            if self.initial.exists():
                manifest = yaml.safe_load(self.initial.read_text(encoding="utf-8"))
                self._validate_manifest(manifest)
                values = {
                    name: decode_state(record, self.root)
                    for name, record in manifest.get("values", {}).items()
                }
                return -1, values
            legacy = next(self.root.glob("iter.*/FINISHED"), None) if self.root.exists() else None
            if legacy is not None:
                raise RuntimeError(
                    f"Legacy repeated workflow layout detected at {self.root}; use a fresh run directory."
                )
            return -1, {}
        manifest = yaml.safe_load(self.current.read_text(encoding="utf-8"))
        self._validate_manifest(manifest)
        values = {
            name: decode_state(record, self.root)
            for name, record in manifest.get("values", {}).items()
        }
        return int(manifest["iteration"]), values

    def initialise(self, values: Mapping[str, Any]) -> pathlib.Path:
        manifest = {
            "format": FORMAT_VERSION,
            "workflow": self.fingerprint,
            "iteration": -1,
            "converged": False,
            "values": {
                name: encode_state(value, self.root, state_name=name, iteration=-1)
                for name, value in values.items()
            },
        }
        if self.initial.exists():
            saved = yaml.safe_load(self.initial.read_text(encoding="utf-8"))
            self._validate_manifest(saved)
            return self.initial
        self._write_atomic(self.initial, manifest)
        return self.initial

    def _validate_manifest(self, manifest: Mapping[str, Any]) -> None:
        if manifest.get("format") != FORMAT_VERSION:
            raise RuntimeError("Unsupported workflow state manifest format; use a fresh run directory.")
        if manifest.get("workflow") != self.fingerprint:
            raise RuntimeError("Workflow configuration changed since state was committed; use a fresh run directory.")

    def commit(self, iteration: int, values: Mapping[str, Any], converged: bool) -> pathlib.Path:
        manifest = {
            "format": FORMAT_VERSION,
            "workflow": self.fingerprint,
            "iteration": iteration,
            "converged": bool(converged),
            "values": {
                name: encode_state(value, self.root, state_name=name, iteration=iteration)
                for name, value in values.items()
            },
        }
        self.iterations.mkdir(parents=True, exist_ok=True)
        target = self.iterations / f"{iteration:04d}.yaml"
        self._write_atomic(target, manifest)
        self._write_atomic(self.current, manifest)
        return target

    @staticmethod
    def _write_atomic(path: pathlib.Path, value: Mapping[str, Any]) -> None:
        _write_yaml_atomic(path, value)
