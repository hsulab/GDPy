"""MC region, transaction and history wrapper around bond-change geometry."""

import copy

from gdpx.structures.geometry.bond_change import (
    BOND_CHANGE_VERSION, retag_fragments,
)
from gdpx.structures.geometry.fragments import FragmentPolicy

from .operator import BaseMCOperator


class BondOperator(BaseMCOperator):
    """Shared MC transaction and region integration for the two bond moves."""

    geometry_version = BOND_CHANGE_VERSION

    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.particles = list(self.settings["particles"])
        self.fragment_policy = FragmentPolicy(bond_ratio=self.settings["covalent_ratio"][1])
        kwargs.setdefault("allow_isolated", True)
        super().__init__(covalent_ratio=self.settings["covalent_ratio"], **kwargs)

    def _propose(self, atoms, rng):
        retag_fragments(atoms, self.fragment_policy, before_move=self._transaction.watch)
        super()._propose(atoms, rng)
        distances = dict(getattr(self, "bond_distance_dict", {}))
        distances.update(getattr(self, "custom_pair_distance_dict", None) or {})
        event, self._extra_info = self.geometry_function(atoms, rng,
            eligible_tags={tag for values in self._curr_tags_dict.values() for tag in values},
            bond_distances=distances, allow_isolated=self.allow_isolated,
            skip_distance_check=self.skip_distance_check, max_random_attempts=self.MAX_RANDOM_ATTEMPTS,
            before_move=self._transaction.watch, **self.settings)
        if event is None:
            return None
        self._transaction.set_info("bond_change", event)
        self._state["bond_change"] = event
        return atoms

    def as_dict(self):
        return dict(super().as_dict(), **copy.deepcopy(self.settings))
