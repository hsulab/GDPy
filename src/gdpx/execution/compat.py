"""Compatibility construction for schema-v1 potential managers."""

import copy
import importlib
from typing import Any, Mapping

DRIVER_TASKS = frozenset({"spc", "min", "ts", "cmin", "md", "freq"})
PATH_TASKS = frozenset({"neb"})


def create_compat_executor(potential: Any, parameters: Mapping[str, Any]) -> Any:
    """Create an old driver/reactor without making the potential own resolution."""
    dyn_params = copy.deepcopy(dict(parameters))
    potential.dyn_params = copy.deepcopy(dyn_params)
    dynamics = dyn_params.get("backend", potential.calc_backend)
    if dynamics == "external":
        dynamics = potential.calc_backend
    if (potential.calc_backend, dynamics) not in potential.valid_combinations:
        raise RuntimeError(f"Invalid dynamics backend {dynamics} based on {potential.calc_backend} calculator")

    merged = {}
    if "task" in dyn_params:
        merged["task"] = dyn_params.get("task", "min")
    if "init" in dyn_params or "run" in dyn_params:
        merged.update(dyn_params.get("init", {}))
        merged.update(dyn_params.get("run", {}))
    else:
        merged.update(dyn_params)
    merged.update(
        ignore_convergence=dyn_params.get("ignore_convergence", False),
        random_seed=dyn_params.get("random_seed"),
    )
    ignore_convergence = merged.pop("ignore_convergence", False)
    random_seed = merged.pop("random_seed", None)

    task = merged.get("task", "min")
    if task in DRIVER_TASKS:
        implementations = {
            "ase": ("gdpx.providers.ase.driver", "AseDriver"),
            "jax": ("gdpx.providers.jax.driver", "JarexDriver"),
            "deepmd_jax": ("gdpx.providers.deepmd.jax_driver", "DeepmdJaxDriver"),
            "lammps": ("gdpx.providers.lammps.execution", "LmpDriver"),
            "lasp": ("gdpx.providers.lasp.driver", "LaspDriver"),
            "abacus": ("gdpx.providers.abacus.driver", "AbacusDriver"),
            "vasp": ("gdpx.providers.vasp.driver", "VaspDriver"),
            "cp2k": ("gdpx.providers.cp2k.driver", "Cp2kDriver"),
            "replica": ("gdpx.providers.replica.driver", "ReplicaDriver"),
        }
    elif task in PATH_TASKS:
        implementations = {
            "ase": ("gdpx.providers.ase.path", "AseStringReactor"),
            "cp2k": ("gdpx.providers.cp2k.path", "Cp2kStringReactor"),
            "vasp": ("gdpx.providers.vasp.path", "VaspStringReactor"),
            "grid": ("gdpx.providers.grid.path", "ZeroStringReactor"),
        }
    else:
        raise ValueError(f"Unknown execution task {task!r} for backend {dynamics!r}.")
    if dynamics not in implementations:
        raise ValueError(f"Unknown execution backend {dynamics!r} for task {task!r}.")
    module, attribute = implementations[dynamics]
    executor_class = getattr(importlib.import_module(module), attribute)
    executor = executor_class(
        potential.calc,
        merged,
        directory=potential.calc.directory,
        ignore_convergence=ignore_convergence,
        random_seed=random_seed,
    )
    executor.pot_params = potential.as_dict()
    return executor
