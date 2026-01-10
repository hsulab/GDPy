import abc
import pathlib
from typing import Any, Callable, Union

from gdpx import config


class Operation(abc.ABC):
    #: Node ID.
    identifier: str = "op"

    #: Whether re-compute this operation
    status: str = "unfinished"  # ["unfinished", "ready", "wait", "finished"]

    #: Standard print function.
    _print: Callable = config._print

    #: Standard debug function.
    _debug: Callable = config._debug

    def __init__(self, input_nodes=[], directory: Union[str, pathlib.Path] = "./") -> None:
        """"""
        self._directory = pathlib.Path(directory)

        self.input_nodes = self._preprocess_input_nodes(input_nodes)

        # Initialise list of consumers
        # (i.e. nodes that receive this operation's output as input)
        self.consumers = []

        # Append this operation to the list of consumers of all input nodes
        for input_node in self.input_nodes:
            input_node.consumers.append(self)

        return

    @property
    def directory(self) -> pathlib.Path:
        """"""

        return self._directory

    @directory.setter
    def directory(self, directory) -> None:
        """"""
        self._directory = pathlib.Path(directory)

        return

    def _preprocess_input_nodes(self, input_nodes) -> tuple[Any, ...]:
        """Preprocess the input nodes.

        Apply default nodes if the input node is None.

        """

        return input_nodes

    def reset(self) -> None:
        """Reset node's output and status."""
        if hasattr(self, "output"):
            delattr(self, "output")
            self.status = "unfinished"

        return

    def is_about_to_exit(self) -> bool:
        """Check whether this operation has an input is about to exit."""
        status = [node.status == "exit" for node in self.input_nodes]
        if any(status):
            return True
        else:
            return False

    def is_ready_to_forward(self) -> bool:
        """Check whether this operation is ready to forward."""
        status = [node.status == "finished" for node in self.input_nodes]
        if all(status):
            return True
        else:
            return False

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """"""
        if not self.directory.exists():
            self.directory.mkdir(parents=True)

        return
