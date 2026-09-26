"""Collective, reversible rattle proposals for MC and basin hopping."""

import numpy as np
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.structures.geometry.spatial import check_atomic_distances_by_neighbour_list

from .operator import BaseMCOperator


class RattleOperator(BaseMCOperator):
    """Translate each eligible tagged particle with probability rattle_prop.

    Displacement components are uniform in [-rattle_strength, rattle_strength]
    Angstrom. Atoms sharing a tag move together without rotation.
    """

    name = "rattle"

    def __init__(self, particles, rattle_strength=0.8, rattle_prop=0.4, **kwargs):
        if not np.isfinite(rattle_strength) or rattle_strength <= 0:
            raise ValueError("rattle_strength must be finite and positive.")
        if not np.isfinite(rattle_prop) or not 0 < rattle_prop <= 1:
            raise ValueError("rattle_prop must be in (0, 1].")
        super().__init__(**kwargs)
        self.particles = particles
        self.rattle_strength = rattle_strength
        self.rattle_prop = rattle_prop

    def _propose(self, atoms, rng):
        super()._propose(atoms, rng)
        eligible = sorted(
            {tag for species, tags in self._curr_tags_dict.items() if species in self.particles for tag in tags}
        )
        if not eligible:
            self._extra_info = "Rattle_Skipped"
            return None
        tags = atoms.get_tags()
        grouped_indices = {tag: [] for tag in eligible}
        for index, tag in enumerate(tags):
            if tag in grouped_indices:
                grouped_indices[tag].append(index)
        groups = [np.asarray(indices, dtype=int) for indices in grouped_indices.values()]
        nl = None
        distances = None
        if not self.skip_distance_check:
            nl = NeighborList(
                self.covalent_max * np.array(natural_cutoffs(atoms)), skin=0.0, self_interaction=False, bothways=True
            )
            distances = dict(self.bond_distance_dict)
            distances.update(self.custom_pair_distance_dict or {})
        watched = set()
        for _ in range(self.MAX_RANDOM_ATTEMPTS):
            selected = [group for group in groups if rng.random() < self.rattle_prop]
            if not selected:
                continue
            indices = np.concatenate(selected)
            unwatched = [int(index) for index in indices if int(index) not in watched]
            if unwatched:
                self._transaction.watch(unwatched)
                watched.update(unwatched)
            original = atoms.positions[indices].copy()
            for group in selected:
                atoms.positions[group] += rng.uniform(-self.rattle_strength, self.rattle_strength, 3)
            # Check each rigid particle separately, including its distances to
            # other particles displaced in this same proposal.
            if self.skip_distance_check or all(
                check_atomic_distances_by_neighbour_list(
                    atoms,
                    neighlist=nl,
                    atomic_indices=group,
                    bond_distance_dict=distances,
                    covalent_ratio=(self.covalent_min, self.covalent_max),
                    allow_isolated=self.allow_isolated,
                )
                for group in selected
            ):
                self._extra_info = f"Rattle_{len(selected)}_particles"
                return atoms
            atoms.positions[indices] = original
        self._extra_info = "Rattle_Failed"
        return None

    def as_dict(self):
        return dict(
            super().as_dict(),
            particles=list(self.particles),
            rattle_strength=self.rattle_strength,
            rattle_prop=self.rattle_prop,
        )
