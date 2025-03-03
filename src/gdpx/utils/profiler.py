#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import time
from typing import Any, Callable


class CustomTimer:
    """A context manager to measure the time of a code block."""

    def __init__(self, name: str = "code", func: Callable = print):
        """"""
        self.name = name
        self._print = func

        return

    def __call__(self, func) -> Any:
        """A decorator to measure the time of a function."""

        def func_timer(*args, **kwargs):
            st = time.time()
            ret = func(*args, **kwargs)
            et = time.time()
            content = f"*** {self.name} time: {et-st:>8.4f} ***"
            self._print(content)

            return ret

        return func_timer

    def __enter__(self):
        """Store the start time."""
        self.st = time.time()

        return self

    def __exit__(self):
        """Store the end time and print the elapsed."""
        self.et = time.time()  # end time

        content = f"*** {self.name} time: {self.et-self.st:>8.4f} ***"
        self._print(content)

        return


if __name__ == "__main__":
    ...
