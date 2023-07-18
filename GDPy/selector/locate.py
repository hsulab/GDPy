#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools

from ..core.register import registers
from .selector import AbstractSelector


@registers.selector.register
class LocateSelector(AbstractSelector):

    name = "locate"

    default_parameters = dict()

    def __init__(self, directory="./", axis=None, *args, **kwargs) -> None:
        super().__init__(directory, axis, *args, **kwargs)

        return
    
    def _mark_structures(self, data, *args, **kwargs) -> None:
        """"""
        super()._mark_structures(data, *args, **kwargs)

        print(f"data: {data}")

        curr_markers = data.markers

        # NOTE: CANNOT USE NEGATIVE INDEX
        # TODO: if take axis=1 with index -1
        keys = [slice(None) for _ in range(data.ndim)]
        keys[1] = slice(47, 48)
        #print(f"keys: {keys}")

        indices = []
        for i, key in enumerate(keys):
            size = data.shape[i]
            curr_indices = range(size)[key]
            indices.append(curr_indices)
        products = list(itertools.product(*indices))
        #print(f"products: {products}")

        selected_markers = []
        for m in curr_markers:
            if tuple(m) in products:
                selected_markers.append(list(m))
        data.markers = selected_markers
        #print(f"markers: {data.markers}")

        return


if __name__ == "__main__":
    ...