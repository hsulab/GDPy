#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ase import Atoms

from .selector import AbstractSelector


class InvariantSelector(AbstractSelector):
    """Perform an invariant selection."""

    name = "invariant"

    default_parameters = dict()

    def __init__(self, *args, **kwargs):
        """"""
        super().__init__(*args, **kwargs)

        return

    def _mark_structures(self, frames: list[Atoms], *args, **kwargs) -> None:
        """Return selected indices."""

        return


if __name__ == "__main__":
    ...
