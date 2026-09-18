"""ASE-GA-compatible soft mutation implemented by GDPy."""

from __future__ import annotations

import copy

import numpy as np
from ase import Atoms

from gdpx.structures.geometry.ga import atoms_too_close

from ..core import OffspringCreator
from .utils import movable_groups


class SoftMutation(OffspringCreator):
    """Displace movable atoms along a smooth, low-frequency random mode."""

    descriptor = "SoftMutation"
    min_inputs = 1

    def __init__(
        self,
        blmin,
        bounds=(0.5, 2.0),
        calculator=None,
        rcut=10.0,
        used_modes_file="used_modes.json",
        use_tags=False,
        verbose=False,
        rng=None,
    ):
        super().__init__(verbose=verbose, rng=rng)
        self.blmin = blmin
        self.bounds = bounds
        self.calculator = calculator
        self.rcut = rcut
        self.used_modes_file = used_modes_file
        self.use_tags = use_tags

    def mutate(self, atoms: Atoms):
        groups = movable_groups(atoms, len(atoms), self.use_tags)
        if not groups:
            return None
        centers = np.array([atoms.positions[group].mean(axis=0) for group in groups])
        for _ in range(1000):
            phase = self.rng.normal(size=(len(groups), 3))
            if len(groups) > 1:
                distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
                kernel = np.exp(-distances / max(float(self.rcut), 1e-12))
                phase = kernel @ phase
            phase -= phase.mean(axis=0)
            scale = np.max(np.linalg.norm(phase, axis=1))
            if scale == 0:
                continue
            amplitude = self.rng.uniform(float(self.bounds[0]), float(self.bounds[1]))
            mutant = copy.deepcopy(atoms)
            for group, displacement in zip(groups, phase * amplitude / scale):
                mutant.positions[group] += displacement
            if not atoms_too_close(mutant, self.blmin, use_tags=self.use_tags):
                return mutant
        return None

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: soft"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: soft"
