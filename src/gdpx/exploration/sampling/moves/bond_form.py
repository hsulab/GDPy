"""MC wrapper for bond form geometry."""

from gdpx.structures.geometry.bond_change import (
    bond_form_settings, form_bond, reject_obsolete_arguments,
)

from ._bond import BondOperator


class BondFormOperator(BondOperator):
    name = "bond_form"
    geometry_function = staticmethod(form_bond)

    def __init__(self, particles=None, bond=None, cutoffs=(0.0, 3.5),
                 covalent_ratio=(0.8, 1.2), acceptor_group=None, **kwargs):
        reject_obsolete_arguments(kwargs)
        settings = bond_form_settings(particles=particles, bond=bond, cutoffs=cutoffs,
            covalent_ratio=covalent_ratio, acceptor_group=acceptor_group)
        super().__init__(settings, **kwargs)
