"""Common workflow-node interface used by session executors."""

from __future__ import annotations

import enum
from typing import Protocol, runtime_checkable


class NodeKind(enum.Enum):
    VARIABLE = "VX"
    OPERATION = "OP"
    PLACEHOLDER = "PH"


@runtime_checkable
class WorkflowNode(Protocol):
    node_kind: NodeKind
    status: str
    consumers: list

    @property
    def directory(self): ...

    @directory.setter
    def directory(self, value): ...

    def reset(self) -> None: ...
