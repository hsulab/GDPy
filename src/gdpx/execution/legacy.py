"""Compatibility construction for pre-provider potential managers."""

import copy
from typing import Any, Mapping

DRIVER_TASKS = frozenset({"spc", "min", "ts", "cmin", "md", "freq"})
PATH_TASKS = frozenset({"neb"})


def create_legacy_executor(potential: Any, parameters: Mapping[str, Any]) -> Any:
    """Create an old driver/reactor without making the potential own resolution."""
    from gdpx.computation import register_drivers
    from gdpx.reactor import register_reactors

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
        executor_class = register_drivers[dynamics]
    elif task in PATH_TASKS:
        executor_class = register_reactors[dynamics]
    else:
        raise ValueError(f"Unknown execution task {task!r} for backend {dynamics!r}.")
    executor = executor_class(
        potential.calc,
        merged,
        directory=potential.calc.directory,
        ignore_convergence=ignore_convergence,
        random_seed=random_seed,
    )
    executor.pot_params = potential.as_dict()
    return executor
