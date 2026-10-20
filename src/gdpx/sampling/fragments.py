"""Sampling integration for geometry-owned fragment identities."""

from gdpx.structures.geometry.fragments import (
    FragmentPolicy, component, element_pair, positive_number,
)


BOND_OPERATORS = frozenset({"bond_break", "bond_form"})


def fragment_policy_for(operators):
    policies = [op.fragment_policy for op in operators if op.name in BOND_OPERATORS]
    if not policies:
        return None
    if any(op.name not in BOND_OPERATORS | {"rotate", "move", "rattle", "bounce"} for op in operators):
        raise ValueError("Bond searches currently require atom-order-preserving geometry operators.")
    if any(policy.as_dict() != policies[0].as_dict() for policy in policies[1:]):
        raise ValueError("All bond operators in a search must use the same fragment policy.")
    return policies[0]
