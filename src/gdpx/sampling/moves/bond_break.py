"""MC wrapper for bond break geometry."""

from gdpx.structures.geometry.bond_change import (
    bond_break_settings, break_bond, reject_obsolete_arguments,
)

from gdpx.sampling.moves._bond import BondOperator


class BondBreakOperator(BondOperator):
    name = "bond_break"
    geometry_function = staticmethod(break_bond)

    def __init__(self, particles=None, bond=None, cutoffs=(0.0, 3.5),
                 covalent_ratio=(0.8, 1.2), **kwargs):
        reject_obsolete_arguments(kwargs)
        settings = bond_break_settings(particles=particles, bond=bond, cutoffs=cutoffs,
            covalent_ratio=covalent_ratio)
        super().__init__(settings, **kwargs)
