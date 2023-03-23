#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class Variable:

    """Intrinsic, changeable parameter of a graph.
    """

    def __init__(self, initial_value=None):
        """"""
        self.value = initial_value
        self.consumers = []

        #_default_graph.variables.append(self)

        return


if __name__ == "__main__":
    ...