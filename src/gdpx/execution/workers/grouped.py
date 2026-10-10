"""Batch compatible executor variants together on one scheduler allocation."""

import copy

import numpy as np

from gdpx.execution.driver import BaseDriver
from gdpx.providers.configuration import resolve_executor_parameters

from .drive import DriverBasedWorker


class GroupedDriverWorker(DriverBasedWorker):
    @classmethod
    def compatible(cls, first, runtime):
        """Whether two runtimes can share a batch worker and candidate layout."""
        for item in (first, runtime):
            if (not isinstance(item.executor, BaseDriver)
                    or item.config.dispatch.worker != "batch"
                    or item.config.dispatch.share_workdir):
                return False
        for field in ("potential", "modifiers", "scheduler"):
            if getattr(runtime.config, field) != getattr(first.config, field):
                return False
        dispatches = [item.config.dispatch.to_dict() for item in (first, runtime)]
        for dispatch in dispatches:
            dispatch.pop("group_variants", None)  # Historical manifests only.
        return (
            dispatches[0] == dispatches[1]
            and (runtime.config.executor.provider, runtime.config.executor.method)
            == (first.config.executor.provider, first.config.executor.method)
            and cls._implicit(first) == cls._implicit(runtime)
        )

    def __init__(self, runtimes, **kwargs):
        if not runtimes:
            raise ValueError("At least one grouped runtime is required.")
        first = runtimes[0]
        for runtime in runtimes:
            dispatch = runtime.config.dispatch
            if dispatch.worker != "batch" or dispatch.share_workdir:
                raise ValueError("Batching variants requires worker=batch and separate workdirs.")
            if not isinstance(runtime.executor, BaseDriver):
                raise TypeError("Grouped variants support driver executors only.")
            for field in ("potential", "modifiers", "scheduler"):
                if getattr(runtime.config, field) != getattr(first.config, field):
                    raise ValueError(f"Grouped variants must have identical {field} configurations.")
            policies = [item.config.dispatch.to_dict() for item in (first, runtime)]
            for policy in policies:
                policy.pop("group_variants", None)
            if policies[0] != policies[1]:
                raise ValueError("Grouped variants must have identical dispatch configurations.")
            if (runtime.config.executor.provider, runtime.config.executor.method) != (
                first.config.executor.provider, first.config.executor.method
            ):
                raise ValueError("Grouped variants must use the same executor provider and method.")
        policies = [self._implicit(runtime) for runtime in runtimes]
        if any(policies) and not all(policies):
            raise ValueError("Grouped variants must either all specify random_seed or all omit it.")
        super().__init__(first, batchsize=first.config.dispatch.batch_size, **kwargs)
        self.runtimes = tuple(runtimes)
        self._drivers = [runtime.executor for runtime in runtimes]
        self._retain_info = first.config.dispatch.retain_info

    @staticmethod
    def _implicit(runtime):
        return resolve_executor_parameters(
            runtime.config.executor.parameters, runtime.config.executor.method
        ).get("random_seed") is None

    def _uses_implicit_seed(self):
        return all(self._implicit(runtime) for runtime in self.runtimes)

    def _make_task_plan(self, num_structures):
        # Keep all variants of one structure together and preserve driver lookup
        # from cand{index} in the shared result reader.
        return [(di, si) for si in range(num_structures) for di in range(len(self._drivers))]

    def _preprocess(self, builder, *args, **kwargs):
        identifier, frames, start, _ = super()._preprocess(builder, *args, **kwargs)
        seeds = []
        for driver in self._drivers:
            rng = np.random.Generator(np.random.PCG64(driver.random_seed))
            seeds.append([driver.random_seed] * len(frames) if self._share_random_seed else
                         [int(seed) for seed in rng.integers(0, 1e8, size=len(frames))])
        return identifier, frames, start, [seeds[di][si] for di, si in self._make_task_plan(len(frames))]

    def as_dict(self):
        dispatch = super().as_dict()["dispatch"]
        configs = [runtime.config.to_dict() for runtime in self.runtimes]
        for config in configs:
            legacy_flag = config["dispatch"].get("group_variants")
            config["dispatch"] = copy.deepcopy(dispatch)
            config["dispatch"].pop("group_variants", None)
            if legacy_flag is not None:
                config["dispatch"]["group_variants"] = legacy_flag
        return configs
