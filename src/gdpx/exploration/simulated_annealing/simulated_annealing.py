"""MD annealing through a linear ramp or thermostatted temperature holds."""
import copy
import json

import numpy as np
from ase.io import read, write

from gdpx.execution.factory import create_worker
from gdpx.execution.fingerprint import payload_digest
from gdpx.providers.configuration import resolve_executor_parameters
from gdpx.structures.builders.factory import canonicalise_builder
from ..checkpoint import write_json
from ..exploration import BaseExploration


class SimulatedAnnealing(BaseExploration):
    """Cool independent candidates using the existing MD runtime."""

    name = "simulated_annealing"

    def __init__(self, builder, temperatures=None, directory="./", random_seed=None,
                 annealing=None):
        super().__init__(directory=directory, random_seed=random_seed)
        self.annealing = copy.deepcopy(annealing)
        if annealing is not None:
            if temperatures is not None:
                raise ValueError("Choose annealing or temperatures, not both.")
            if (not isinstance(annealing, dict)
                    or set(annealing) - {"start_temperature", "final_temperature", "schedule"}
                    or annealing.get("schedule", "linear") != "linear"):
                raise ValueError("annealing requires a linear schedule and start/final temperatures.")
            temperatures = [annealing.get("start_temperature"), annealing.get("final_temperature")]
        if not isinstance(temperatures, (list, tuple)) or not temperatures:
            raise ValueError("temperatures must be a nonempty list.")
        if any(isinstance(t, bool) or not isinstance(t, (int, float))
               or not np.isfinite(t) or t <= 0 for t in temperatures):
            raise ValueError("temperatures must be finite and positive.")
        self.temperatures = list(map(float, temperatures))
        if annealing is not None:
            if self.temperatures[0] < self.temperatures[1]:
                raise ValueError("Annealing final temperature must not exceed start temperature.")
            self.temperatures = self.temperatures[:1]
        self.builder = canonicalise_builder(builder) if isinstance(builder, dict) else builder

    def register_worker(self, worker, *args, **kwargs):
        if isinstance(worker, list) and len(worker) != 1:
            raise ValueError("SA requires exactly one runtime.")
        super().register_worker(worker, *args, **kwargs)
        config = self.worker.runtime.config
        if config.executor.method != "md":
            raise ValueError(
                f"SA requires executor.method='md'; "
                f"received provider={config.executor.provider!r}, method={config.executor.method!r}. "
                "Select a thermostatted MD runtime with --runtime "
                "(for Cu8: examples/global_optimisation/explorations/simulated_annealing/runtime.yaml)."
            )
        params = resolve_executor_parameters(config.executor.parameters, "md")
        if params.get("ensemble") not in ("nvt", "npt"):
            raise ValueError("SA requires thermostatted NVT or NPT dynamics.")
        steps = params.get("steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("MD steps must be a positive integer.")
        if params.get("tend") is not None:
            raise ValueError("Remove tend; the exploration defines the annealing schedule.")
        if config.modifiers:
            raise ValueError("SA currently requires an unmodified potential.")
        if len(self.worker._drivers) != 1 or self.worker._share_wdir:
            raise ValueError("SA requires one runtime and separate candidate workdirs.")

    def as_dict(self):
        return copy.deepcopy({
            "method": self.name,
            "recipe": {"random_seed": self.random_seed, "builder": self.builder.as_dict(),
                       **({"annealing": self.annealing} if self.annealing is not None
                          else {"temperatures": self.temperatures})},
            "runtime": self.worker.as_dict(),
        })

    def _signature(self):
        config = self.as_dict()
        # Direct builders serialize their frames; ASE's JSON encoder handles them.
        from ase.io.jsonio import encode
        return payload_digest(json.loads(encode(config)))

    def _manifest(self):
        path = self.directory / "annealing.json"
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest.get("version") != 1 or manifest["signature"] != self._signature():
                raise ValueError("Annealing configuration changed; use a new directory.")
            if not (self.directory / "seeds.traj").exists():
                raise ValueError("Annealing seed snapshot is missing.")
            return manifest
        if (self.directory / "FINISHED").exists() or (self.directory / "slices").exists():
            raise ValueError("Missing annealing manifest; use a new directory.")
        frames = list(self.builder.run())
        if not frames:
            raise ValueError("SA requires at least one starting structure.")
        self.directory.mkdir(parents=True, exist_ok=True)
        write(self.directory / "seeds.traj", frames)
        manifest = dict(version=1, signature=self._signature(), count=len(frames),
                        seeds=self.rng.integers(0, 2**31 - 1,
                                               size=(len(self.temperatures), len(frames))).tolist())
        write_json(path, manifest)
        return manifest

    def _worker(self, stage):
        config = self.worker.as_dict()
        # Resolve to the supported flat provider schema before overriding targets.
        params = resolve_executor_parameters(config["executor"]["parameters"], "md")
        params.update(temp=self.temperatures[stage], ignore_atoms_velocities=False)
        if self.annealing is not None:
            params["tend"] = float(self.annealing["final_temperature"])
        config["executor"]["parameters"] = params
        config["dispatch"]["worker"] = "batch"
        return create_worker(config, directory=self.directory / "slices" / f"s{stage:03d}")

    def run(self):
        super().run()
        manifest = self._manifest()
        if self.read_convergence():
            return
        frames = read(self.directory / "seeds.traj", ":")
        for stage, temperature in enumerate(self.temperatures):
            worker = self._worker(stage)
            worker.run(frames, rng_states=manifest["seeds"][stage])
            worker.inspect(resubmit=True)
            if worker.get_number_of_running_jobs():
                return
            results = worker.retrieve(include_retrieved=True)
            by_name = {traj[-1].info["wdir"]: traj[-1] for traj in results if traj}
            expected = [f"cand{i}" for i in range(manifest["count"])]
            if len(results) != len(expected) or set(by_name) != set(expected):
                raise RuntimeError("SA requires one complete trajectory per candidate.")
            frames = [by_name[name] for name in expected]
            for index, frame in enumerate(frames):
                for key in ("step", "wdir", "confid", "group"):
                    frame.info.pop(key, None)
                frame.info.update(annealing_stage=stage, annealing_temperature=(
                    self.annealing["final_temperature"] if self.annealing is not None else temperature),
                                  annealing_candidate=index)
            self._print(f"Annealing stage {stage + 1}/{len(self.temperatures)}: {temperature:g} K complete")
        write(self.directory / "final.xyz", frames)
        (self.directory / "FINISHED").write_text("")

    def read_convergence(self):
        return (self.directory / "FINISHED").exists()

    def get_workers(self):
        return [self._worker(i) for i in range(len(self.temperatures))]
