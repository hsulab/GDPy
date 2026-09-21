import copy

import numpy as np

from .moves import (
    AdsorbateExchangeOperator,
    BiasedVolumeExchangeOperator,
    BounceOperator,
    CavityExchangeOperator,
    ExchangeOperator,
    MoveOperator,
    RattleOperator,
    ReactOperator,
    SwapOperator,
    SwapTypeOperator,
)


def select_operator(operators: list, probs: list[float], rng: np.random.Generator):
    """Select an operator based on the relative probabilities."""
    num_operators = len(operators)
    op_idx = rng.choice(num_operators, 1, p=probs)[0]
    op = operators[op_idx]

    return op


def parse_operators(op_params: list[dict]):
    """Parse parameters for various operators.

    Includes single-particle moves, collective rattles, swaps, and exchanges.

    """
    operators, probs = [], []
    for raw_param in op_params:
        param = copy.deepcopy(raw_param)
        if "prob" in param:
            raise ValueError("Legacy operator key 'prob' is not supported; use 'probability'.")
        name = param.pop("method", "move")
        prob = param.get("probability", 1.0)
        if name == "move":
            op = MoveOperator(**param)
        elif name == "rattle":
            op = RattleOperator(**param)
        elif name == "bounce":
            op = BounceOperator(**param)
        elif name == "swap":
            op = SwapOperator(**param)
        elif name == "swap_type":
            op = SwapTypeOperator(**param)
        elif name == "exchange":
            op = ExchangeOperator(**param)
        elif name == "biased_volume_exchange":
            op = BiasedVolumeExchangeOperator(**param)
        elif name == "cavity_exchange":
            op = CavityExchangeOperator(**param)
        elif name == "adsorbate_exchange":
            op = AdsorbateExchangeOperator(**param)
        elif name == "react":
            op = ReactOperator(**param)
        else:
            raise NotImplementedError(f"{name} is not supported.")
        operators.append(op)
        probs.append(prob)

    # - reweight probabilities
    if probs:
        if not np.all(np.isfinite(probs)) or np.any(np.array(probs) < 0) or sum(probs) <= 0:
            raise ValueError("Operator probabilities must be finite, nonnegative, and have a positive sum.")
        probs = (np.array(probs) / np.sum(probs)).tolist()

    return operators, probs


if __name__ == "__main__":
    ...
