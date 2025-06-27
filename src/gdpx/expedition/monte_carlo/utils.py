#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pickle

import numpy as np

from .operators import BounceOperator, ExchangeOperator, MoveOperator, ReactOperator, SwapOperator, SwapTypeOperator


def save_operator(op, p):
    """"""
    with open(p, "wb") as fopen:
        pickle.dump(op, fopen)

    return


def load_operator(p):
    """"""
    with open(p, "rb") as fopen:
        op = pickle.load(fopen)

    return op


def select_operator(operators: list, probs: list[float], rng: np.random.Generator = np.random.default_rng()):
    """Select an operator based on the relative probabilities."""
    num_operators = len(operators)
    op_idx = rng.choice(num_operators, 1, p=probs)[0]
    op = operators[op_idx]

    return op


def parse_operators(op_params: list[dict]):
    """Parse parameters for various operators.

    Currently, we have move, swap, and exchange (insert/remove).

    """
    operators, probs = [], []
    for param in op_params:
        name = param.pop("method", "move")
        prob = param.get("prob", 1.0)
        if name == "move":
            op = MoveOperator(**param)
        elif name == "bounce":
            op = BounceOperator(**param)
        elif name == "swap":
            op = SwapOperator(**param)
        elif name == "swap_type":
            op = SwapTypeOperator(**param)
        elif name == "exchange":
            op = ExchangeOperator(**param)
        elif name == "react":
            op = ReactOperator(**param)
        else:
            raise NotImplementedError(f"{name} is not supported.")
        operators.append(op)
        probs.append(prob)

    # - reweight probabilities
    probs = (np.array(probs) / np.sum(probs)).tolist()

    return operators, probs


if __name__ == "__main__":
    ...
