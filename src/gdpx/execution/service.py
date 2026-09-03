"""Operational lifecycle contracts kept separate from scientific executors."""

import abc
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecutionHandle:
    identifier: str


@dataclass(frozen=True)
class ExecutionStatus:
    state: str
    message: Optional[str] = None


class ExecutionService(abc.ABC):
    @abc.abstractmethod
    def submit(self, runtime: Any, inputs: Any) -> ExecutionHandle:
        ...

    @abc.abstractmethod
    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        ...

    @abc.abstractmethod
    def retrieve(self, handle: ExecutionHandle) -> Any:
        ...


@runtime_checkable
class ExecutionWorker(Protocol):
    """Structural contract implemented by execution workers."""

    def run(self, inputs: Any, **kwargs: Any) -> Any:
        ...

    def inspect(self, resubmit: bool = False, **kwargs: Any) -> Any:
        ...

    def retrieve(self, **kwargs: Any) -> Any:
        ...

    def get_number_of_running_jobs(self) -> int:
        ...


class WorkerExecutionService(ExecutionService):
    """Expose a legacy worker through the new operational lifecycle."""

    def __init__(self, worker: ExecutionWorker) -> None:
        if not isinstance(worker, ExecutionWorker):
            raise TypeError(f"Expected a worker-like object, got {type(worker).__name__}.")
        self.worker = worker
        self._handles: Dict[str, bool] = {}

    def submit(self, runtime: Any, inputs: Any) -> ExecutionHandle:
        if runtime is not None and hasattr(self.worker, "runtime"):
            self.worker.runtime = runtime
        identifier = str(uuid.uuid4())
        self.worker.run(inputs)
        self._handles[identifier] = True
        return ExecutionHandle(identifier)

    def status(self, handle: ExecutionHandle) -> ExecutionStatus:
        self._require_handle(handle)
        self.worker.inspect(resubmit=False)
        state = "running" if self.worker.get_number_of_running_jobs() else "finished"
        return ExecutionStatus(state)

    def retrieve(self, handle: ExecutionHandle) -> Any:
        self._require_handle(handle)
        return self.worker.retrieve()

    def _require_handle(self, handle: ExecutionHandle) -> None:
        if handle.identifier not in self._handles:
            raise KeyError(f"Unknown execution handle {handle.identifier!r}.")
