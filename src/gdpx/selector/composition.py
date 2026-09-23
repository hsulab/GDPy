#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from ..data.array import AtomsNDArray
from .selector import BaseSelector


class ComposedSelector(BaseSelector):
    """Perform several selections consecutively."""

    name = "composed"

    default_parameters = dict(selectors=[])

    def __init__(
        self, selectors: list[BaseSelector], *args, **kwargs
    ):
        """"""
        super().__init__(*args, **kwargs)

        self.selectors = selectors

        return

    def _mark_structures(self, frames: AtomsNDArray, *args, **kwargs) -> None:
        """Return selected indices."""
        # - update selectors' directories
        for s in self.selectors:
            s.directory = self._directory

        # - initial index stuff
        curr_frames = frames

        # - run selection
        for i, node in enumerate(self.selectors):
            # - adjust name
            prev_fname = node._fname
            node.fname = str(i) + "-" + prev_fname
            # - map indices
            #   TODO: use _select_indices instead?
            node.select(curr_frames)

            node.fname = prev_fname

        return


if __name__ == "__main__":
    ...
