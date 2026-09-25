import copy
import inspect
from typing import Any

from ..population.comparators import COMPARATORS
from .crossover import PeriodicCutAndSpliceCrossover
from .mutation.bounce import BounceMutation
from .mutation.cluster import ClusterRattleMutation
from .mutation.cluster_rotation import ClusterRotationMutation
from .mutation.exchange import ExchangeMutation
from .mutation.group_rattle import GroupRattleMutation
from .mutation.mirror import MirrorMutation
from .mutation.rattle import RattleMutation
from .mutation.soft import SoftMutation
from .mutation.strain import StrainMutation
from .mutation.swap import SwapMutation

CROSSOVERS: dict[str, Any] = dict(
    cut_and_splice=PeriodicCutAndSpliceCrossover,
)

MUTATIONS: dict[str, Any] = dict(
    # GDPy implementations of ASE-GA-compatible mutations
    mirror=MirrorMutation,
    rattle=RattleMutation,
    soft=SoftMutation,
    strain=StrainMutation,
    # Custom mutations
    bounce=BounceMutation,
    cluster_rattle=ClusterRattleMutation,
    cluster_rotation=ClusterRotationMutation,
    exchange=ExchangeMutation,
    group_rattle=GroupRattleMutation,
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
    if "rng" in op_params:
        raise ValueError("Operator RNGs are managed by the GA engine; remove the 'rng' option.")
    method = op_params.pop("method", None)
    if category == "mutation" and method == "rattle_buffer":
        raise ValueError("Mutation 'rattle_buffer' was renamed to 'group_rattle'.")
    if category == "crossover" and method in {
        "periodic_cut_and_splice", "cluster_cut_and_splice", "cut_and_splice_cluster"
    }:
        raise ValueError(
            f"Crossover {method!r} is no longer supported; use 'cut_and_splice' "
            "with system.periodic=true and system.preserve_fragments=true "
            "for isolated clusters."
        )
    if method is None:
        raise Exception(f"There is no operator {method}.")

    op_cls = GENETIC_OPERATORS[category].get(method, None)
    if op_cls is None:
        raise Exception(f"Operator {method} is not found in {category} if {GENETIC_OPERATORS[category].keys()}.")

    sig = inspect.signature(op_cls.__init__)
    init_args = [
        name
        for name, param in sig.parameters.items()
        # skip `self` and *args/**kwargs
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]
    for k, v in specific_params.items():
        if k in init_args:
            op_params.update(**{k: v})
    op = op_cls(**op_params)

    return op
