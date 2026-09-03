"""Create execution workers from resolved schema-v2 runtimes."""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence
from typing import Callable

from gdpx.execution.driver import BaseDriver
from gdpx.execution.reactor import BaseReactor
from gdpx.execution.runtime import Runtime
from gdpx.execution.workers.drive import DriverBasedWorker
from gdpx.execution.workers.react import ReactorBasedWorker
from gdpx.execution.workers.single import SingleWorker
from gdpx.execution.workers.worker import BaseWorker
from gdpx.providers import RuntimeConfig


RuntimeInput = Runtime | RuntimeConfig | Mapping


def _resolve(value: RuntimeInput) -> Runtime:
    if isinstance(value, Runtime):
        return value
    from gdpx.execution import resolve_runtime

    config = value if isinstance(value, RuntimeConfig) else RuntimeConfig.from_mapping(value)
    return resolve_runtime(config)


def create_worker(
    value: RuntimeInput,
    *,
    directory="./",
    print_func: Callable = print,
) -> BaseWorker:
    """Create exactly one worker from one complete runtime."""
    runtime = _resolve(value)
    options = dict(runtime.config.options)
    batch_size = options.pop("batch_size", 1)
    worker_kind = options.pop("worker", "batch")
    share_workdir = options.pop("share_workdir", False)
    retain_info = options.pop("retain_info", False)
    if options:
        raise TypeError(f"Unknown runtime options: {', '.join(sorted(options))}.")

    if isinstance(runtime.executor, BaseDriver):
        if worker_kind == "single":
            worker = SingleWorker(runtime=runtime)
        elif worker_kind == "batch":
            worker = DriverBasedWorker(runtime=runtime)
        else:
            raise ValueError(f"Unknown driver worker kind {worker_kind!r}; expected 'batch' or 'single'.")
        worker._share_wdir = bool(share_workdir)
        worker._retain_info = bool(retain_info)
    elif isinstance(runtime.executor, BaseReactor):
        if worker_kind != "batch":
            raise ValueError("Reactor runtimes support only the 'batch' worker kind.")
        worker = ReactorBasedWorker(runtime=runtime)
    else:
        raise TypeError(f"Unsupported runtime executor {type(runtime.executor).__name__}.")

    worker.batchsize = int(batch_size)
    worker.directory = pathlib.Path(directory)
    print_func(
        f"runtime {runtime.config.potential.provider}/"
        f"{runtime.config.executor.provider}:{runtime.config.executor.method}"
    )
    return worker


def create_workers(
    values: Sequence[RuntimeInput],
    *,
    directory="./",
    print_func: Callable = print,
) -> list[BaseWorker]:
    """Create independent workers from an explicit non-empty runtime list."""
    if isinstance(values, (str, bytes, Mapping, Runtime, RuntimeConfig)):
        raise TypeError("create_workers requires an explicit sequence; use create_worker for one runtime.")
    runtimes = list(values)
    if not runtimes:
        raise ValueError("At least one runtime is required.")
    root = pathlib.Path(directory)
    return [
        create_worker(
            value,
            directory=root if len(runtimes) == 1 else root / f"w{index}",
            print_func=print_func,
        )
        for index, value in enumerate(runtimes)
    ]


def create_worker_chains(
    chains: Sequence[Sequence[RuntimeInput]],
    *,
    directory="./",
    print_func: Callable = print,
) -> list[list[BaseWorker]]:
    """Create explicit worker chains; each inner sequence is one ordered chain."""
    if not chains:
        raise ValueError("At least one worker chain is required.")
    root = pathlib.Path(directory)
    result = []
    for chain_index, values in enumerate(chains):
        if not values:
            raise ValueError(f"Worker chain {chain_index} is empty.")
        chain_root = root if len(chains) == 1 else root / f"chain{chain_index}"
        result.append(
            [
                create_worker(
                    value,
                    directory=chain_root / f"step{step_index}",
                    print_func=print_func,
                )
                for step_index, value in enumerate(values)
            ]
        )
    return result
