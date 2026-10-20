"""Resolve sampling configuration without materializing an outer calculator."""

import copy
from collections.abc import Mapping
from dataclasses import replace

from gdpx.execution.factory import create_worker
from gdpx.execution.runtime import Runtime
from gdpx.providers.configuration import RuntimeConfig, resolve_executor_parameters
from gdpx.providers.specs import thaw

from .monte_carlo import MonteCarlo, resolve_monte_carlo_system, resolve_monte_carlo_convergence
from .hybrid_monte_carlo import HybridMonteCarlo, validate_cycle


def resolve_sampling_runtime(config, resolver):
    method = config.executor.method
    if config.dispatch.share_workdir:
        raise ValueError("Sampling executors do not support dispatch.share_workdir.")
    raw = thaw(config.executor.parameters)
    unknown = raw.keys() - {"setup", "output", "stop", "random_seed"}
    if unknown:
        raise ValueError("Sampling parameters require setup/output/stop; unsupported: "
                         + ", ".join(sorted(unknown)))
    for section in ("setup", "output", "stop"):
        if not isinstance(raw.get(section, {}), Mapping):
            raise TypeError(f"executor.parameters.{section} must be a mapping.")
    setup = raw.get("setup", {})
    allowed = {"ensemble", "operators", "ignore_atoms_tags", "evaluator"}
    if method == "hybrid_monte_carlo":
        allowed.add("cycle")
    if setup.keys() - allowed:
        raise ValueError("Unsupported sampling setup: " + ", ".join(sorted(setup.keys() - allowed)))
    operators, _ = resolve_monte_carlo_system(
        {key: value for key, value in setup.items() if key in {"ensemble", "ignore_atoms_tags"}},
        setup.get("operators"), method,
    )
    convergence = resolve_monte_carlo_convergence(raw.get("stop", {}), method)
    parameters = resolve_executor_parameters(raw, method)
    dump_period = parameters.get("dump_period", 1)
    ckpt_period = parameters.get("ckpt_period", 100)
    for name, value in (("output.trajectory.period", dump_period),
                        ("output.checkpoint.period", ckpt_period)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    ckpt_keep = parameters.get("ckpt_number", 2)
    if not isinstance(ckpt_keep, int) or isinstance(ckpt_keep, bool) or ckpt_keep <= 0:
        raise ValueError("output.checkpoint.keep must be a positive integer.")
    evaluator = setup.get("evaluator")
    if not isinstance(evaluator, Mapping) or evaluator.keys() - {"executor", "scheduler"}:
        raise ValueError("setup.evaluator requires executor and optional scheduler.")
    if not isinstance(evaluator.get("executor"), Mapping) or evaluator["executor"].get("method") != "spc":
        raise ValueError(f"{method} requires setup.evaluator.executor.method: spc.")
    child = {
        "potential": config.potential.to_dict(),
        "modifiers": [modifier.to_dict() for modifier in config.modifiers],
        "executor": copy.deepcopy(evaluator["executor"]),
        "dispatch": {"worker": "single"},
    }
    if "scheduler" in evaluator:
        child["scheduler"] = copy.deepcopy(evaluator["scheduler"])
    child_runtime = resolver.resolve(child)
    kwargs = {}
    stage_runtimes = []
    cls = MonteCarlo
    if method == "hybrid_monte_carlo":
        cls = HybridMonteCarlo
        cycle = validate_cycle(setup.get("cycle"))
        for stage in cycle:
            stage_runtime = RuntimeConfig.from_mapping(stage["runtime"])
            if stage_runtime.dispatch.share_workdir:
                raise ValueError("Hybrid stages require separate working directories.")
            # Resolve stage capabilities now so preparation fails before launching work.
            resolved_stage = resolver.resolve(stage_runtime)
            stage_runtimes.append(resolved_stage)
            stage["runtime"] = resolved_stage.config.to_dict()
        kwargs["cycle"] = cycle
        raw["setup"]["cycle"] = cycle
    executor = cls(
        operators=operators, convergence=convergence,
        random_seed=raw.get("random_seed"), dump_period=dump_period,
        ckpt_period=ckpt_period, ignore_atoms_tags=setup.get("ignore_atoms_tags", True),
        should_retry=False, restart=False, **kwargs,
    )
    executor.register_worker(create_worker(child_runtime))
    executor.ckpt_keep = ckpt_keep
    executor._cycle_runtimes = stage_runtimes
    raw["setup"]["evaluator"]["executor"] = child_runtime.config.executor.to_dict()
    # Persist backend choices while retaining the public sampling configuration.
    config = replace(config, potential=child_runtime.config.potential,
                     modifiers=child_runtime.config.modifiers,
                     executor=replace(config.executor, parameters=raw))
    executor.runtime_config = config.to_dict()
    return Runtime(
        potential=config.potential_spec(), materialization=None, executor=executor,
        modifiers=config.modifier_specs(), config=config,
        scheduler=resolver.resolve_scheduler(config),
    )
