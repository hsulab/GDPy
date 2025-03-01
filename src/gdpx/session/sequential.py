#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pathlib
import time
from typing import Union

from gdpx.session.operation import Operation

from .session import BaseSession, SessionState
from .utils import traverse_postorder


class SequentialSession(BaseSession):

    def __init__(self, directory: Union[str, pathlib.Path] = "./") -> None:
        """"""
        self.directory = pathlib.Path(directory)

        return

    def run(self, operation: Operation, feed_dict: dict = {}) -> None:
        """"""
        self.state = SessionState.StepToStart

        def set_node_directory(
            node: Operation, node_index, working_directory: pathlib.Path
        ) -> None:
            """"""
            prev_name = node.directory.name
            if not prev_name:
                prev_name = node.__class__.__name__
            node.directory = (
                working_directory / f"{node_index:>04d}.{prev_name}"
            )

            return

        # Find forward order
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if hasattr(node, "_active") and node._active:
                node._active = False
                self._print(
                    f"Set {node} active to false as it is not supported in a basic session"
                )

        self._run_nodes(
            self.directory,
            nodes_postorder=nodes_postorder,
            feed_dict=feed_dict,
            reset_states=False,
            set_node_dir_func=set_node_directory,
        )
        if not (self.state == SessionState.StepFinished):
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

            if self.state == SessionState.StepFinished:
                self.state = SessionState.LoopFinished

        return


if __name__ == "__main__":
    ...
