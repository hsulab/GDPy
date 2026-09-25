"""Acceptance decisions independent of geometry mutation and evaluation.

These preserve the historical formulas, including heuristics used by biased
search moves. Their presence does not certify detailed balance of a proposal.
"""

import math
from dataclasses import dataclass

import numpy as np
from ase import units

from .statmech import compute_thermo_wavelength


@dataclass(frozen=True)
class AcceptanceRule:
    temperature: float

    def factors(self, metadata):
        return 1.0, 0.0

    def probability(self, metadata, previous_energy, trial_energy):
        prefactor, energy_offset = self.factors(metadata)
        with np.errstate(over="ignore", under="ignore"):
            weight = prefactor * np.exp(
                -(trial_energy - previous_energy + energy_offset) / (units.kB * self.temperature)
            )
            return float(np.minimum(1.0, weight))

    def accept(self, proposal, previous_energy, trial_energy, rng):
        metadata = proposal.metadata if hasattr(proposal, "metadata") else proposal
        return bool(rng.uniform() < self.probability(metadata, previous_energy, trial_energy))


@dataclass(frozen=True)
class ExchangeAcceptance(AcceptanceRule):
    chemical_potential: float
    cubic_wavelength: float

    def factors(self, metadata):
        count, volume = metadata["num_particles"], metadata["volume"]
        symmetry = metadata.get("symm_factor", 1.0)
        if metadata["operation"] == "insert":
            return volume / (count + 1) / self.cubic_wavelength / symmetry, -self.chemical_potential
        return symmetry * count * self.cubic_wavelength / volume, self.chemical_potential


@dataclass(frozen=True)
class SemiGrandAcceptance(AcceptanceRule):
    particles: tuple
    chemical_potentials: tuple

    def factors(self, metadata):
        before = self.particles.index(metadata["first_ptype"])
        after = self.particles.index(metadata["second_ptype"])
        return metadata.get("proposal_ratio", 1.0), (
            self.chemical_potentials[before] - self.chemical_potentials[after]
        )


@dataclass(frozen=True)
class ReactionAcceptance(AcceptanceRule):
    coefficients: tuple
    chemical_potentials: tuple

    def factors(self, metadata):
        direction = metadata["direction"]
        beta = 1.0 / (units.kB * self.temperature)
        net_change = sum(self.coefficients)
        chemical_change = sum(c * mu for c, mu in zip(self.coefficients, self.chemical_potentials))
        equilibrium = np.exp(-beta * chemical_change)
        equilibrium *= (beta * 1e5 * 1e-30 * units.kJ / 1000.0) ** net_change
        factorial_ratio = math.prod(
            math.factorial(int(count)) / math.factorial(int(count + direction * coefficient))
            for count, coefficient in zip(metadata["particle_numbers"], self.coefficients)
        )
        prefactor = equilibrium ** direction * float(metadata["volume"]) ** (net_change * direction)
        return prefactor * factorial_ratio, 0.0


def acceptance_for(move):
    if hasattr(move, "_particle_instances"):
        wavelength = compute_thermo_wavelength(move._particle_instances[0].get_masses().sum(), move.temperature)
        return ExchangeAcceptance(move.temperature, move.chempots[0], wavelength)
    if move.name == "swap_type":
        return SemiGrandAcceptance(move.temperature, tuple(move.particles), tuple(move.chempots))
    if move.name == "react":
        return ReactionAcceptance(move.temperature, tuple(move.reaction.coefficients), tuple(move.reaction.chempot_0))
    return AcceptanceRule(move.temperature)
