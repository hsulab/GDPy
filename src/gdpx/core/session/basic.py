#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib

from ..placeholder import Placeholder
from ..variable import Variable
from .session import AbstractSession
from .utils import traverse_postorder


class Session(AbstractSession):

    def __init__(self, directory="./") -> None:
        """"""
        self.directory = pathlib.Path(directory)

        return

    def run(self, operation, feed_dict: dict = {}) -> None:
        """"""
        # - find forward order
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if hasattr(node, "_active") and node._active:
                node._active = False
                self._print(
                    f"Set {node} active to false as it is not supported in a basic session"
                )

        self._print(
            "[{:^24s}] NUM_NODES: {} AT MAIN: {}".format(
                "START", len(nodes_postorder), str(self.directory)
            )
        )

        # - run nodes
        for i, node in enumerate(nodes_postorder):
            # NOTE: reset directory since it maybe changed
            prev_name = node.directory.name
            if not prev_name:
                prev_name = node.__class__.__name__
            node.directory = self.directory / f"{str(i).zfill(4)}.{prev_name}"
            if node.__class__.__name__.endswith("Variable"):
                node_type = "VX"
            else:
                node_type = "OP"
            self._print(
                "[{:^24s}] NAME: {} AT {}".format(
                    node_type, node.__class__.__name__.upper(), node.directory.name
                )
            )

            if isinstance(node, Placeholder):
                node.output = feed_dict[node]
            elif isinstance(node, Variable):
                node.output = node.value
            else:  # Operation
                self._debug(f"node: {node}")
                self._process_operation(node)

        return


if __name__ == "__main__":
    ...
