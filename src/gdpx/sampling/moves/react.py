#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import dataclasses
from typing import List

import numpy as np
from ase import Atoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList, natural_cutoffs

from gdpx.structures.geometry.composition import convert_string_to_atoms

from .operator import BaseMCOperator


@dataclasses.dataclass
class ElementaryReaction:

    #: Particles
    particles: List[str]

    #: The standard chemical potentials.
    chempot_0: List[float]

    #: Chemical coefficients.
    coefficients: List[int]

    def __post_init__(
        self,
    ):
        """"""
        self.number_net_change = np.sum(self.coefficients)
        self.chempot_change = np.sum([c * mu for c, mu in zip(self.coefficients, self.chempot_0)])

        equation_parts = [[], []]
        self.reactants, self.products = [], []
        for c, p in zip(self.coefficients, self.particles):
            if c < 0:
                equation_parts[0].append(f"{-1*c}{p}")
                self.reactants.append(p)
            else:
                equation_parts[1].append(f"{c}{p}")
                self.products.append(p)
        self.chemical_equation = "+".join(equation_parts[0]) + "->" + "+".join(equation_parts[1])

        return



class ReactOperator(BaseMCOperator):

    name: str = "react"
    MIN_RANDOM_TAG = 10000
    MAX_RANDOM_TAG = 100000

    def __init__(
        self,
        reaction,
        region,
        temperature,
        pressure: float = 1.0,
        use_bias: bool = True,
        *args,
        **kwargs,
    ):
        """"""
        super().__init__(region=region, temperature=temperature, pressure=pressure, *args, **kwargs)

        # - parse reaction
        self.reaction = ElementaryReaction(**reaction)

        # -
        self.use_bias = use_bias

        return

    def _propose(self, atoms: Atoms, rng=np.random.Generator(np.random.PCG64())) -> Atoms:
        """"""
        super()._propose(atoms, rng)
        self._extra_info = "-"

        # -- volume?
        if self.use_bias:
            # --- determine on-the-fly
            acc_volume = self.region.get_empty_volume(atoms)
        else:
            # --- get normal region
            acc_volume = self.region.get_volume()
        self._curr_volume = acc_volume

        # -- TODO: find possible reaction sites...
        self._print(f"reaction: {self.reaction.chemical_equation}")

        # --
        reactant_numbers = [len(self._curr_tags_dict.get(r, [])) for r in self.reaction.reactants]
        has_reactants = all([r_n > 0 for r_n in reactant_numbers])

        product_numbers = [len(self._curr_tags_dict.get(p, [])) for p in self.reaction.products]
        has_products = all([p_n > 0 for p_n in product_numbers])

        self._curr_particle_numbers = [len(self._curr_tags_dict.get(p, [])) for p in self.reaction.particles]

        if not has_reactants and not has_products:
            self._print("No reactants and products.")
            self._curr_operation = "skipped"
            curr_atoms = None
        elif has_reactants and not has_products:
            self._print("Only reactants.")
            self._print("...forward...")
            self._curr_operation = "forward"
            curr_atoms = self._forward_reaction(atoms, rng)
        elif not has_reactants and has_products:
            self._print("Only products.")
            self._print("...reverse...")
            self._curr_operation = "reverse"
            curr_atoms = self._reverse_reaction(atoms, rng)
        else:
            rn_rxn = rng.uniform()
            if rn_rxn < 0.5:
                self._print("...forward...")
                self._curr_operation = "forward"
                curr_atoms = self._forward_reaction(atoms, rng)
            else:
                self._print("...reverse...")
                self._curr_operation = "reverse"
                curr_atoms = self._reverse_reaction(atoms, rng)

        self._extra_info = f"{self._curr_operation.capitalize()}_{self.reaction.chemical_equation}"

        return curr_atoms


    def _create_a_particle(self, atoms: Atoms, species: str, rng) -> Atoms:
        """"""
        # -
        particle = convert_string_to_atoms(species)

        # - add velocity in case the mixed MC/MD is performed
        MaxwellBoltzmannDistribution(particle, temperature_K=self.temperature, rng=rng)

        # - add tag
        used_tags = set(atoms.get_tags().tolist())
        new_tag = 0
        while new_tag in used_tags:
            # NOTE: np.random only has randint
            new_tag = rng.integers(self.MIN_RANDOM_TAG, self.MAX_RANDOM_TAG)
        new_tag = int(new_tag)
        # NOTE: ase accepts int or list as tags
        particle.set_tags(new_tag)

        return particle

    def _forward_reaction(self, atoms, rng):
        return self._react(atoms, rng, 1)

    def _reverse_reaction(self, atoms, rng):
        return self._react(atoms, rng, -1)

    def _react(self, atoms, rng, direction):
        changes = [direction * coefficient for coefficient in self.reaction.coefficients]
        if any(count + change < 0 for count, change in zip(self._curr_particle_numbers, changes)):
            return None
        self._direction = direction
        sites = []
        for species, change in zip(self.reaction.particles, changes):
            for _ in range(max(0, -change)):
                self._check_region(atoms)
                indices = self._select_species(atoms, [species], rng)
                sites.append(atoms.positions[indices].mean(axis=0))
                self._transaction.delete(indices)
        site = sites[0] if sites else self.region.get_random_positions(size=1, rng=rng)[0]
        for species, change in zip(self.reaction.particles, changes):
            for _ in range(max(0, change)):
                particle = self._create_a_particle(atoms, species, rng)
                particle.translate(site - particle.positions.mean(axis=0))
                indices = list(range(len(atoms), len(atoms) + len(particle)))
                self._transaction.append(particle)
                nl = NeighborList(self.covalent_max * np.array(natural_cutoffs(atoms)),
                                  skin=0.0, self_interaction=False, bothways=True)
                for _ in range(self.MAX_RANDOM_ATTEMPTS):
                    trial = self._rotate_species(particle, rng)
                    atoms.positions[indices] = trial.positions
                    if self.skip_distance_check or not self.check_overlap_neighbour(
                        nl, atoms, atoms.get_cell(complete=True), indices
                    ):
                        break
                else:
                    return None
        return atoms

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"reaction: "
        content += f"  {self.reaction.chemical_equation}\n"
        content += f"  within the region {self.region}\n"

        return content

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()
        params["reaction"] = dataclasses.asdict(self.reaction)
        params["use_bias"] = self.use_bias
        return params


if __name__ == "__main__":
    ...
