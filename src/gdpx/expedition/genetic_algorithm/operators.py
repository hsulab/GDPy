#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import inspect
from typing import Any

from ase.ga.cutandsplicepairing import CutAndSplicePairing
from ase.ga.ofp_comparator import OFPComparator
from ase.ga.particle_comparator import NNMatComparator
from ase.ga.particle_crossovers import CutSpliceCrossover
from ase.ga.soft_mutation import SoftMutation
from ase.ga.standardmutations import RattleMutation, StrainMutation

from .comparator.interatomic_distance import InteratomicDistanceComparator
from .mutation.bounce import BounceMutation
from .mutation.exchange import ExchangeMutation
from .mutation.mirror import MirrorMutation
from .mutation.rattle import RattleBufferMutation
from .mutation.swap import SwapMutation

COMPARATORS: dict[str, Any] = dict(
    # ASE built-in comparators
    ofp=OFPComparator,
    nnmat=NNMatComparator,
    # Custom comparators
    interatomic_distance=InteratomicDistanceComparator,
)

CROSSOVERS: dict[str, Any] = dict(
    # ASE built-in crossovers
    cut_and_splice=CutAndSplicePairing,
    cut_and_splice_cluster=CutSpliceCrossover,
)

MUTATIONS: dict[str, Any] = dict(
    # ASE built-in mutations
    mirror=MirrorMutation,
    rattle=RattleMutation,
    soft=SoftMutation,
    strain=StrainMutation,
    # Custom mutations
    bounce=BounceMutation,
    exchange=ExchangeMutation,
    rattle_buffer=RattleBufferMutation,
    swap=SwapMutation,
)

GENETIC_OPERATORS: dict[str, dict[str, Any]] = dict(
    comparator=COMPARATORS,
    crossover=CROSSOVERS,
    mutation=MUTATIONS,
)


def instantiate_a_genetic_operator(
    category: str,
    op_params: dict,
    specific_params: dict,
):
    """Instantiate operators such as comparator, crossover, and mutation.

    Args:
        op_params: Operator parameters loaded from input file.
        specific_params: Operator parameters obtained based on system.

    Returns:
        An instance of an operator that can be used in the genetic.

    """
    assert category in GENETIC_OPERATORS, f"Genetic operator category {category} is not found."

    op_params = copy.deepcopy(op_params)
    method = op_params.pop("method", None)
    if method is None:
        raise Exception(f"There is no operator {method}.")

    op_cls = GENETIC_OPERATORS[category].get(method, None)
    if op_cls is None:
        raise Exception(f"Operator {method} is not found in {category}.")

    init_args = inspect.getargspec(op_cls.__init__).args[1:]  # skip self
    for k, v in specific_params.items():
        if k in init_args:
            op_params.update(**{k: v})
    op = op_cls(**op_params)

    return op


if __name__ == "__main__":
    ...
