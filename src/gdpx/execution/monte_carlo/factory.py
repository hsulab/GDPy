"""Bind native tasks and their children to one fixed potential surface."""
import copy
from collections.abc import Mapping
from dataclasses import replace

from gdpx.execution.factory import create_worker
from gdpx.execution.runtime import Runtime
from gdpx.providers.configuration import child_executor, resolve_executor_parameters
from gdpx.providers.specs import thaw
from gdpx.providers.ensemble import inherit_ensemble
from .monte_carlo import MonteCarlo, resolve_monte_carlo_system, resolve_monte_carlo_convergence
from .hybrid_monte_carlo import HybridMonteCarlo


def resolve_child(config, definition, resolver):
    definition = copy.deepcopy(definition)
    scheduler = definition.pop("scheduler", None)
    child = dict(potential=config.potential.to_dict(),
                 modifiers=[item.to_dict() for item in config.modifiers],
                 executor=definition, dispatch={"worker": "batch"})
    if scheduler is not None:
        child["scheduler"] = scheduler
    return resolver.resolve(child)


def resolve_sampling_runtime(config, resolver):
    method = config.executor.method
    if config.dispatch.share_workdir:
        raise ValueError("Native executors do not support dispatch.share_workdir.")
    raw = thaw(config.executor.parameters)
    unknown = raw.keys() - {"setup", "output", "stop", "random_seed", "ensemble"}
    if unknown:
        raise ValueError("Native parameters accept ensemble/setup/output/stop/random_seed; unsupported: " + ", ".join(sorted(unknown)))
    convergence = resolve_monte_carlo_convergence(raw.get("stop", {}), method)
    parameters = resolve_executor_parameters(raw, method)
    cadence = {"dump_period": parameters.get("dump_period", 1),
               "ckpt_period": parameters.get("ckpt_period", 100),
               "ckpt_keep": parameters.get("ckpt_number", 2)}
    for key, label in (("dump_period", "trajectory.period"), ("ckpt_period", "checkpoint.period"),
                       ("ckpt_keep", "checkpoint.keep")):
        value = cadence[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"output.{label} must be a positive integer.")
    if method == "monte_carlo":
        setup = raw.get("setup", {})
        allowed = {"operators", "ignore_atoms_tags", "executor"}
        if not isinstance(setup, Mapping) or setup.keys() - allowed:
            raise ValueError("Unsupported MC setup; use operators, ignore_atoms_tags, and optional executor.")
        operators, _ = resolve_monte_carlo_system(
            {"ensemble": parameters["ensemble"], "ignore_atoms_tags": setup.get("ignore_atoms_tags", True)},
            setup.get("operators"), method)
        evaluator = child_executor(setup.get("executor", {}), default_spc=True)
        if evaluator["method"] != "spc" or evaluator["provider"] == "native":
            raise ValueError("MC setup.executor requires method: spc.")
        child = resolve_child(config, evaluator, resolver)
        config = replace(config, potential=child.config.potential, modifiers=child.config.modifiers)
        executor = MonteCarlo(operators=operators, convergence=convergence,
                              random_seed=raw.get("random_seed"), dump_period=cadence["dump_period"],
                              ckpt_period=cadence["ckpt_period"],
                              ignore_atoms_tags=setup.get("ignore_atoms_tags", True), should_retry=False)
        executor.ckpt_keep = cadence["ckpt_keep"]
        executor.register_worker(create_worker(child))
        executor.runtime_config = config.to_dict()
    else:
        definitions = [inherit_ensemble(item, parameters["ensemble"]) for item in raw["setup"]]
        methods = [item["method"] for item in definitions]
        if set(methods) != {"md", "monte_carlo"}:
            raise ValueError("HMC setup requires MD and conventional Monte Carlo children only.")
        for item in definitions:
            if item["method"] == "monte_carlo" and item["provider"] != "native":
                raise ValueError("HMC Monte Carlo children require provider: native.")
        first = next(i for i, item in enumerate(definitions) if item["method"] == "monte_carlo")
        children = [None] * len(definitions)
        children[first] = resolve_child(config, definitions[first], resolver)
        config = replace(config, potential=children[first].config.potential,
                         modifiers=children[first].config.modifiers)
        for i, definition in enumerate(definitions):
            if i != first:
                children[i] = resolve_child(config, definition, resolver)
        initializer = children[first].executor.worker.runtime
        executor = HybridMonteCarlo(config, children, initializer, convergence, **cadence)
    return Runtime(potential=config.potential_spec(), materialization=None, executor=executor,
                   modifiers=config.modifier_specs(), config=config,
                   scheduler=resolver.resolve_scheduler(config))
