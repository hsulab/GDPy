#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import dataclasses

from typing import List

import numpy as np

from ase import Atoms
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.neighborlist import NeighborList, natural_cutoffs

from .. import convert_string_to_atoms
from .exchange import BasicExchangeOperator


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
        self.chempot_change = np.sum(
            [c * mu for c, mu in zip(self.coefficients, self.chempot_0)]
        )

        equation_parts = [[], []]
        self.reactants, self.products = [], []
        for c, p in zip(self.coefficients, self.particles):
            if c < 0:
                equation_parts[0].append(f"{-1*c}{p}")
                self.reactants.append(p)
            else:
                equation_parts[1].append(f"{c}{p}")
                self.products.append(p)
        self.chemical_equation = (
            "+".join(equation_parts[0]) + "->" + "+".join(equation_parts[1])
        )

        return

    def compute_equilibrium_constant(self, temperature: float):
        """Compute the ideal-gas equilibrium constant."""
        kBT_eV = units.kB * temperature
        beta = 1.0 / kBT_eV

        # -
        self.k = np.exp(-beta * self.chempot_change)

        # - add PV term
        p_0 = 1e5  # pascal
        ang3tom3 = 1e-30
        pv_factor = (
            beta * p_0 * ang3tom3 * units.kJ / 1000.0
        ) ** self.number_net_change

        self.k2 = pv_factor * self.k

        return


class ReactOperator(BasicExchangeOperator):

    name: str = "react"

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
        super().__init__(
            region=region, temperature=temperature, pressure=pressure, *args, **kwargs
        )

        # - parse reaction
        self.reaction = ElementaryReaction(**reaction)
        self.reaction.compute_equilibrium_constant(temperature=temperature)

        # -
        self.use_bias = use_bias

        return

    def run(self, atoms: Atoms, rng=np.random.Generator(np.random.PCG64())) -> Atoms:
        """"""
        super().run(atoms)
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
        reactant_numbers = [
            len(self._curr_tags_dict.get(r, [])) for r in self.reaction.reactants
        ]
        has_reactants = all([r_n > 0 for r_n in reactant_numbers])

        product_numbers = [
            len(self._curr_tags_dict.get(p, [])) for p in self.reaction.products
        ]
        has_products = all([p_n > 0 for p_n in product_numbers])

        self._curr_particle_numbers = reactant_numbers + product_numbers

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

        self._extra_info = (
            f"{self._curr_operation.capitalize()}_{self.reaction.chemical_equation}"
        )

        return curr_atoms

    def _compute_prefactor(self, particle_numbers: List[int], xi: int):
        """Compute the prefactor of the acceptance probability.

        Args:
            particle_numbers: Particle numbers.
            xi: Reaction direction 1 or -1.

        """
        # -
        particle_numbers = np.array(particle_numbers)
        number_changes = np.array(self.reaction.coefficients) * xi

        fac = 1.0
        for n, c in zip(particle_numbers, number_changes):
            fac *= np.math.factorial(n) / np.math.factorial(n + c)

        # -
        prefactor = (
            (self.reaction.k2**xi)
            * (self._curr_volume ** (self.reaction.number_net_change * xi))
            * fac
        )

        return prefactor

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

    def _forward_reaction(
        self, atoms_: Atoms, rng=np.random.Generator(np.random.PCG64())
    ) -> Atoms:
        """Perform a forward reaction."""
        self._curr_prefactor = self._compute_prefactor(
            self._curr_particle_numbers, xi=1
        )

        atoms = atoms_.copy()

        # - remove reactant and select reactive site
        site_positions = []
        for r in self.reaction.reactants:
            reax_indices = self._select_species(atoms, [r], rng)
            selected_species = atoms[reax_indices]
            site_positions.append(np.mean(selected_species.positions, axis=0))
            del atoms[reax_indices]
            self._print(f"remove {r} {reax_indices}")

        # -- default uses the first reactant's position
        site_position = site_positions[0]

        # - insert the product to the site position
        # TODO: we only support one product here...
        product = self.reaction.products[0]
        particle = self._create_a_particle(atoms, product, rng)
        new_tag = int(particle.get_tags()[0])
        self._print(
            f"product {particle.get_chemical_formula()} tag: {particle.get_tags()}"
        )
        atoms.extend(particle)

        particle_indices = [i for i, t in enumerate(atoms.get_tags()) if t == new_tag]
        species = atoms[particle_indices]

        new_vec = site_position - np.mean(species.positions, axis=0)
        species.translate(new_vec)

        original_positions = copy.deepcopy(species.get_positions())

        # - adjust product configuration
        nl = NeighborList(
            self.covalent_max * np.array(natural_cutoffs(atoms)),
            skin=0.0,
            self_interaction=False,
            bothways=True,
        )
        cell = atoms.get_cell(complete=True)

        for i in range(self.MAX_RANDOM_ATTEMPTS):
            species_ = self._rotate_species(species, rng=rng)
            atoms.positions[particle_indices] = copy.deepcopy(species_.positions)
            if not self.check_overlap_neighbour(nl, atoms, cell, particle_indices):
                self._print(f"succeed to insert after {i+1} attempts...")
                break
            atoms.positions[particle_indices] = original_positions
        else:
            del atoms[particle_indices]
            atoms = None

        return atoms

    def _reverse_reaction(
        self, atoms_: Atoms, rng=np.random.Generator(np.random.PCG64())
    ) -> Atoms:
        """Perform a reverse reaction."""
        self._curr_prefactor = self._compute_prefactor(
            self._curr_particle_numbers, xi=-1
        )

        # - remove products
        atoms = atoms_
        for p in self.reaction.products:
            atoms = self._remove(atoms, species=p, rng=rng)

        # - insert reactants
        for r in self.reaction.reactants:
            atoms = self._insert(atoms, species=r, rng=rng)
            if atoms is None:
                break

        return atoms

    def metropolis(self, prev_ene: float, curr_ene: float, rng=np.random) -> bool:
        """Acceptance probability.

        We have equations as
        P = (beta*P0*V)^(nu*xi)*K^xi*PROD_i^s(N0!/(N0+nu_i*xi)!)*exp(-beta*dU)
        K = \exp(-\sum_i^s{nu_i*\mu_i^0}*beta)

        nu - net number change in particles
        xi - reaction direction

        """
        # - acceptance ratio
        kBT_eV = units.kB * self.temperature
        beta = 1.0 / kBT_eV  # 1/(kb*T), eV

        ene_diff = curr_ene - prev_ene
        ene_term = np.exp(-beta * ene_diff)

        acc_ratio = np.min([1.0, self._curr_prefactor * ene_term])

        content = f"\nVolume {self._curr_volume:>.4f}\n"
        content += f"Prefactor {self._curr_prefactor}\n"
        content += f"Energy Difference {ene_diff:>.4f} [eV]  Exp {ene_term}\n"
        content += f"Accept Ratio {acc_ratio:>.4f}\n"
        for l in content.split("\n"):
            self._print(l)

        # -
        rn_rxn = rng.uniform()
        self._print(f"{self.__class__.__name__} Probability %.4f" % rn_rxn)

        # - reset stored temp data
        self._curr_operation = None
        self._curr_tags_dict = None
        self._curr_volume = None
        self._curr_particle_numbers = None
        self._curr_prefactor = None

        return rn_rxn < acc_ratio

    def __repr__(self) -> str:
        """"""
        content = f"@Modifier {self.__class__.__name__}\n"
        content += (
            f"temperature {self.temperature} [K] pressure {self.pressure} [bar]\n"
        )
        content += "covalent ratio: \n"
        content += f"  min: {self.covalent_min} max: {self.covalent_max}\n"
        content += f"reaction: "
        content += f"  {self.reaction.chemical_equation}\n"
        content += f"  within the region {self.region}\n"

        return content

    def as_dict(self) -> dict:
        """"""
        params = super().as_dict()

        return params


if __name__ == "__main__":
    ...
