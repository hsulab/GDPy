"""Independent harmonic-distance windows for rare-event sampling."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import tempfile
from collections.abc import Mapping

import numpy as np
from ase.io import read, write

from gdpx.execution.factory import create_worker
from gdpx.structures.groups import evaluate_group_expression

from .exploration import BaseExploration

MANIFEST_VERSION = 1
MANIFEST_NAME = "windows.json"
SEEDS_NAME = "seeds.xyz"
SUMMARY_NAME = "samples.csv"


def _positive_integer(value, path):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path} must be a positive integer.")
    return value


def create_umbrella_sampling(system, strategy, random_seed=None, directory="./"):
    """Validate the public configuration and construct an umbrella exploration."""
    if not isinstance(system, Mapping):
        raise TypeError("umbrella_sampling system must be a mapping.")
    unknown = system.keys() - {"builder", "collective_variable"}
    if unknown:
        raise ValueError(f"Unsupported umbrella_sampling system settings: {', '.join(sorted(unknown))}.")
    if "builder" not in system:
        raise ValueError("umbrella_sampling requires system.builder.")
    collective_variable = system.get("collective_variable")
    if not isinstance(collective_variable, Mapping):
        raise TypeError("system.collective_variable must be a mapping.")
    unknown = collective_variable.keys() - {"method", "group"}
    if unknown:
        raise ValueError(f"Unsupported collective-variable settings: {', '.join(sorted(unknown))}.")
    if collective_variable.get("method") != "distance":
        raise ValueError("system.collective_variable.method must be distance.")
    group = collective_variable.get("group")
    if not isinstance(group, (str, list, tuple)):
        raise TypeError("system.collective_variable.group must be a group expression or index sequence.")

    if not isinstance(strategy, Mapping):
        raise TypeError("umbrella_sampling strategy must be a mapping.")
    unknown = strategy.keys() - {"centers", "kspring", "replicas", "equilibration_steps"}
    if unknown:
        raise ValueError(f"Unsupported umbrella_sampling strategy settings: {', '.join(sorted(unknown))}.")
    centers = strategy.get("centers")
    if not isinstance(centers, list) or not centers:
        raise ValueError("strategy.centers must be a nonempty list.")
    if any(
        isinstance(center, bool)
        or not isinstance(center, (int, float))
        or not np.isfinite(center)
        or center <= 0
        for center in centers
    ):
        raise ValueError("strategy.centers must contain finite positive numbers.")
    centers = [float(center) for center in centers]
    if len(set(centers)) != len(centers):
        raise ValueError("strategy.centers must be unique.")
    kspring = strategy.get("kspring")
    if (
        isinstance(kspring, bool)
        or not isinstance(kspring, (int, float))
        or not np.isfinite(kspring)
        or kspring <= 0
    ):
        raise ValueError("strategy.kspring must be finite and positive.")
    replicas = _positive_integer(strategy.get("replicas", 1), "strategy.replicas")
    equilibration_steps = _positive_integer(
        strategy.get("equilibration_steps"), "strategy.equilibration_steps"
    )
    return UmbrellaSampling(
        builder=system["builder"],
        group=copy.deepcopy(group),
        centers=centers,
        kspring=float(kspring),
        replicas=replicas,
        equilibration_steps=equilibration_steps,
        random_seed=random_seed,
        directory=directory,
    )


class UmbrellaSampling(BaseExploration):
    """Run equilibrated, independent harmonic-distance windows."""

    name = "umbrella_sampling"

    def __init__(self, builder, group, centers, kspring, replicas, equilibration_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.builder = builder
        self.group = copy.deepcopy(group)
        self.centers = tuple(float(center) for center in centers)
        self.kspring = float(kspring)
        self.replicas = replicas
        self.equilibration_steps = equilibration_steps

    @property
    def manifest_path(self):
        return self.directory / MANIFEST_NAME

    @property
    def seeds_path(self):
        return self.directory / SEEDS_NAME

    def register_worker(self, worker, *args, **kwargs):
        super().register_worker(worker, *args, **kwargs)
        config = self.worker.runtime.config
        if config.executor.provider != "ase" or config.executor.method != "md":
            raise ValueError("umbrella_sampling requires an ASE MD runtime.")
        steps = config.executor.parameters.get("steps")
        _positive_integer(steps, "runtime.executor.parameters.steps")
        if any(modifier.method == "distance_harmonic" for modifier in config.modifiers):
            raise ValueError(
                "The umbrella runtime must be unbiased; remove its distance_harmonic modifier."
            )

    def _public_config(self):
        return {
            "method": self.name,
            "random_seed": self.random_seed,
            "system": {
                "builder": self.builder.as_dict(),
                "collective_variable": {"method": "distance", "group": copy.deepcopy(self.group)},
            },
            "strategy": {
                "centers": list(self.centers),
                "kspring": self.kspring,
                "replicas": self.replicas,
                "equilibration_steps": self.equilibration_steps,
            },
            "runtime": self.worker.as_dict(),
        }

    def _config_digest(self):
        payload = json.dumps(self._public_config(), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _publish_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = pathlib.Path(stream.name)
            json.dump(data, stream, indent=2)
            stream.write("\n")
        try:
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _resolve_group(self, atoms):
        if isinstance(self.group, str):
            indices = evaluate_group_expression(atoms, self.group)
        else:
            indices = list(self.group)
        if len(indices) != 2:
            raise ValueError(
                f"The umbrella distance group must select exactly two atoms; selected {len(indices)}."
            )
        return indices

    def _distance(self, atoms):
        indices = self._resolve_group(atoms)
        return float(atoms.get_distance(indices[0], indices[1], mic=True))

    def _initialise_manifest(self):
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("version") != MANIFEST_VERSION:
                raise ValueError("Unsupported umbrella sampling manifest version; use a fresh directory.")
            if manifest.get("config_digest") != self._config_digest():
                raise ValueError("Umbrella sampling configuration changed; use a fresh directory.")
            if not self.seeds_path.exists():
                raise ValueError("Umbrella sampling seed snapshot is missing.")
            return manifest

        seeds = list(self.builder.run())
        if not seeds:
            raise ValueError("umbrella_sampling requires at least one seed structure.")
        reference = seeds[0]
        for index, atoms in enumerate(seeds):
            if len(atoms) != len(reference) or atoms.get_chemical_symbols() != reference.get_chemical_symbols():
                raise ValueError(f"Umbrella seed {index} does not match the first seed's atoms.")
            if not np.array_equal(atoms.pbc, reference.pbc):
                raise ValueError(f"Umbrella seed {index} does not match the first seed's periodicity.")
        distances = [self._distance(atoms) for atoms in seeds]
        self.directory.mkdir(parents=True, exist_ok=True)
        write(self.seeds_path, seeds)

        records = []
        for window, center in enumerate(self.centers):
            seed_index = min(range(len(seeds)), key=lambda index: abs(distances[index] - center))
            for replica in range(self.replicas):
                random_seeds = self.rng.integers(0, 2**31 - 1, size=4).tolist()
                records.append(
                    {
                        "window": window,
                        "replica": replica,
                        "center": center,
                        "kspring": self.kspring,
                        "seed_index": seed_index,
                        "initial_distance": distances[seed_index],
                        "equilibration_velocity_seed": random_seeds[0],
                        "equilibration_random_seed": random_seeds[1],
                        "production_velocity_seed": random_seeds[2],
                        "production_random_seed": random_seeds[3],
                        "directory": f"windows/w{window:03d}/r{replica:03d}",
                    }
                )
        manifest = {
            "version": MANIFEST_VERSION,
            "config_digest": self._config_digest(),
            "centers": list(self.centers),
            "replicas": self.replicas,
            "records": records,
        }
        self._publish_json(self.manifest_path, manifest)
        return manifest

    @staticmethod
    def _metadata(record):
        return {
            "umbrella_center": record["center"],
            "umbrella_kspring": record["kspring"],
            "umbrella_window": record["window"],
            "umbrella_replica": record["replica"],
            "umbrella_seed_index": record["seed_index"],
        }

    def _runtime_for(self, record, phase):
        config = copy.deepcopy(self.worker.as_dict())
        parameters = config["executor"]["parameters"]
        if phase == "equilibration":
            parameters["steps"] = self.equilibration_steps
        parameters["velocity_seed"] = int(record[f"{phase}_velocity_seed"])
        parameters["random_seed"] = int(record[f"{phase}_random_seed"])
        config.setdefault("modifiers", []).append(
            {
                "provider": "builtin",
                "method": "distance_harmonic",
                "parameters": {
                    "group": copy.deepcopy(self.group),
                    "center": float(record["center"]),
                    "kspring": float(record["kspring"]),
                },
            }
        )
        config.setdefault("dispatch", {})["worker"] = "single"
        return config

    def _worker_for(self, record, phase):
        directory = self.directory / record["directory"] / phase
        worker = create_worker(self._runtime_for(record, phase), directory=directory)
        worker._retrieve_mode = "all"
        return worker

    @staticmethod
    def _last_frame(worker):
        trajectories = worker.retrieve(include_retrieved=True)
        frames = [frame for trajectory in trajectories for frame in trajectory]
        if not frames:
            raise RuntimeError(f"Umbrella phase at {worker.directory} produced no trajectory.")
        return frames[-1]

    def _annotate_trajectory(self, worker, record):
        trajectory_path = worker.directory / "cand0" / worker._drivers[0].xyz_fname
        frames = read(trajectory_path, ":")
        metadata = self._metadata(record)
        if all(all(frame.info.get(key) == value for key, value in metadata.items()) for frame in frames):
            return
        for frame in frames:
            frame.info.update(metadata)
        temporary = trajectory_path.with_suffix(".tmp.xyz")
        write(temporary, frames)
        os.replace(temporary, trajectory_path)

    def _run_phase(self, records, phase, inputs):
        completed = {}
        for record, atoms in zip(records, inputs):
            worker = self._worker_for(record, phase)
            worker.run([atoms])
            worker.inspect(resubmit=True)
            if worker.get_number_of_running_jobs() == 0:
                if phase == "production":
                    self._annotate_trajectory(worker, record)
                completed[(record["window"], record["replica"])] = self._last_frame(worker)
        return completed

    def _write_summary(self, records):
        lines = ["window,replica,frame,step,center,distance,bias_energy,max_devi_f"]
        for record in records:
            worker = self._worker_for(record, "production")
            trajectories = worker.retrieve(include_retrieved=True)
            frames = [frame for trajectory in trajectories for frame in trajectory]
            for frame_index, atoms in enumerate(frames):
                info = atoms.info
                lines.append(
                    ",".join(
                        [
                            str(record["window"]),
                            str(record["replica"]),
                            str(frame_index),
                            str(info.get("step", "")),
                            str(record["center"]),
                            f"{self._distance(atoms):.12g}",
                            str(info.get("bias_energy", "")),
                            str(info.get("max_devi_f", "")),
                        ]
                    )
                )
        (self.directory / SUMMARY_NAME).write_text("\n".join(lines) + "\n")

    def run(self, *args, **kwargs):
        super().run(*args, **kwargs)
        manifest = self._initialise_manifest()
        if self.read_convergence():
            return
        records = manifest["records"]
        seeds = read(self.seeds_path, ":")
        equilibration_inputs = []
        for record in records:
            atoms = seeds[record["seed_index"]].copy()
            if "momenta" in atoms.arrays:
                del atoms.arrays["momenta"]
            atoms.info.update(self._metadata(record))
            equilibration_inputs.append(atoms)
        equilibrated = self._run_phase(records, "equilibration", equilibration_inputs)
        if len(equilibrated) != len(records):
            return

        production_inputs = []
        for record in records:
            atoms = equilibrated[(record["window"], record["replica"])].copy()
            for key in ("step", "confid", "wdir"):
                atoms.info.pop(key, None)
            atoms.info.update(self._metadata(record))
            production_inputs.append(atoms)
        produced = self._run_phase(records, "production", production_inputs)
        if len(produced) != len(records):
            return
        self._write_summary(records)
        (self.directory / "FINISHED").write_text("")

    def read_convergence(self, *args, **kwargs):
        return (self.directory / "FINISHED").exists()

    def get_workers(self, *args, **kwargs):
        manifest = json.loads(self.manifest_path.read_text())
        return [self._worker_for(record, "production") for record in manifest["records"]]

    def as_dict(self):
        return copy.deepcopy(self._public_config())
