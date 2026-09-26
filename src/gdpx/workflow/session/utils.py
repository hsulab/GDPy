#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from .operation import Operation


def traverse_postorder(operation: Operation):

    nodes_postorder = []
    identifiers = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        if id(node) not in identifiers:
            nodes_postorder.append(node)
            identifiers.append(id(node))

    recurse(operation)

    return nodes_postorder


if __name__ == "__main__":
    ...
