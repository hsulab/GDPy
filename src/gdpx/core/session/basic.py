#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
import time
from typing import Union

from ..operation import Operation
from ..placeholder import Placeholder
from ..variable import Variable
from .session import AbstractSession
from .utils import traverse_postorder


class Session(AbstractSession):

    def __init__(self, directory: Union[str, pathlib.Path] = "./") -> None:
        """"""
        self.directory = pathlib.Path(directory)

        return

    def run(self, operation: Operation, feed_dict: dict = {}) -> None:
        """"""
        self.state = "StepToStart"

        # Find forward order
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if hasattr(node, "_active") and node._active:
                node._active = False
                self._print(
                    f"Set {node} active to false as it is not supported in a basic session"
                )

        self._irun(self.directory, nodes_postorder, feed_dict)
        if not (self.state == "StepFinished"):
            self._print("wait current iteration to finish...")
        else:
            if not (self.directory / "FINISHED").exists():
                # Save state to a file
                with open(self.directory / "FINISHED", "w") as fopen:
                    fopen.write(
                        f"STATE {self.state} FINISHED AT {time.asctime( time.localtime(time.time()) )}."
                    )
            else:
                ...

        return

    def _irun(
        self,
        wdir: pathlib.Path,
        nodes_postorder: list[Operation],
        feed_dict: dict = {},
    ) -> None:
        """"""
        if (wdir / "FINISHED").exists():
            self.state = "StepFinished"
            return

        # Whether clear nodes?

        # Show session information
        num_nodes = len(nodes_postorder)
        self._print(
            "\x1b[1;34;40m"
            + f"[{'START':^24s}] NUM_NODES: {num_nodes} AT MAIN: "
            + "\x1b[0m"
        )
        self._print("\x1b[1;34;40m" + f"    {str(wdir)}" + "\x1b[0m")

        # Run nodes
        self.state = "StepFinished"
        for i, node in enumerate(nodes_postorder):
            # Change node version?
            # Reset directory since it maybe changed
            prev_name = node.directory.name
            if not prev_name:
                prev_name = node.__class__.__name__
            node.directory = wdir / f"{i:>04d}.{prev_name}"
            if node.__class__.__name__.endswith("Variable"):
                node_type = "VX"
            else:
                node_type = "OP"
            self._print(
                "[{:^24s}] NAME: {} AT {}".format(
                    node_type,
                    node.__class__.__name__.upper(),
                    node.directory.name,
                )
            )

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:  # Operation
                assert isinstance(
                    node, Operation
                ), f"Unknown node type: {type(node)}"
                self._debug(f"node: {node}")
                self._process_operation(node)

        return


if __name__ == "__main__":
    ...
