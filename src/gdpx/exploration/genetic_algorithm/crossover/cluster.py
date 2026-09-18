"""Cut-and-splice crossover for isolated clusters."""

from __future__ import annotations

from itertools import chain

import numpy as np
from ase import Atoms

from ..core import OffspringCreator


class ClusterCutAndSpliceCrossover(OffspringCreator):
    """Cut-and-splice crossover for unsupported clusters."""

    descriptor = "ClusterCutAndSpliceCrossover"
    min_inputs = 2

    def __init__(self, blmin, keep_composition=True, rng=None):
        super().__init__(rng=rng)
        self.blmin = blmin
        self.keep_composition = keep_composition
        self.allow_variable_composition = not keep_composition

    def get_new_individual(self, parents):
        first, second = parents
        child = self.initialize_individual(first)
        child.info["data"]["parents"] = [parent.info["confid"] for parent in parents]

        theta = self.rng.random() * 2 * np.pi
        phi = self.rng.random() * np.pi
        normal = np.array(
            (np.sin(phi) * np.cos(theta), np.sin(theta) * np.sin(phi), np.cos(phi))
        )
        epsilon = 0.0001
        first.translate(-first.get_center_of_mass())
        second.translate(-second.get_center_of_mass())
        first_map = [np.dot(position, normal) for position in first.positions]
        second_map = [-np.dot(position, normal) for position in second.positions]
        inside = sorted([value for value in chain(first_map, second_map) if value > 0], reverse=True)
        outside = sorted([value for value in chain(first_map, second_map) if value < 0], reverse=True)
        difference = len(inside) - len(first)
        if difference < 0:
            distance = (abs(outside[abs(difference) - 1]) + abs(outside[abs(difference)])) * 0.5
            first.translate(normal * distance)
            second.translate(-normal * distance)
        elif difference > 0:
            distance = (abs(inside[-difference - 1]) + abs(inside[-difference])) * 0.5
            first.translate(-normal * distance)
            second.translate(normal * distance)

        first_part, second_part = Atoms(), Atoms()
        for atom in first:
            if np.dot(atom.position, normal) > 0:
                atom.tag = 1
                first_part.append(atom)
        for atom in second:
            if np.dot(atom.position, normal) < 0:
                atom.tag = 2
                second_part.append(atom)

        if self.keep_composition:
            target_numbers = sorted(first.numbers)
            current_numbers = sorted(list(first_part.numbers) + list(second_part.numbers))
            corrections = {number: target_numbers.count(number) for number in set(target_numbers)}
            for number in current_numbers:
                corrections[number] -= 1
            correction_part = first_part if self.rng.choice([0, 1]) else second_part
            additions, removals = [], []
            for number, amount in corrections.items():
                if amount > 0:
                    additions.extend([number] * amount)
                elif amount < 0:
                    removals.extend([number] * abs(amount))
            for addition, removal in zip(additions, removals):
                candidates = [atom.index for atom in correction_part if atom.number == removal]
                if candidates:
                    correction_part[self.rng.choice(candidates)].number = addition

        maximum_shift = 0.0
        for vector, minimum_distance in self.get_vectors_below_min_dist(first_part + second_part):
            vector_length = np.linalg.norm(vector)
            solutions = [-np.dot(normal, vector)] * 2
            root = np.sqrt(np.dot(normal, vector) ** 2 - vector_length**2 + minimum_distance**2)
            solutions[0] += root
            solutions[1] -= root
            shift = sorted(abs(value) for value in solutions)[0] / 2.0 + epsilon
            maximum_shift = max(maximum_shift, shift)
        first_part.translate(normal * maximum_shift)
        second_part.translate(-normal * maximum_shift)
        for atom in chain(first_part, second_part):
            child.append(atom)
        description = f"{self.descriptor}:Parents {first.info['confid']} {second.info['confid']}"
        return self.finalize_individual(child), description

    def get_vectors_below_min_dist(self, atoms):
        positions = atoms.get_positions()
        numbers = atoms.numbers
        for i in range(len(atoms)):
            position = atoms[i].position
            for j, distance in enumerate(np.linalg.norm(other - position) for other in positions[i:]):
                if distance == 0:
                    continue
                minimum = self.blmin[tuple(sorted((numbers[i], numbers[j + i])))]
                if distance < minimum:
                    yield atoms[i].position - atoms[j + i].position, minimum

    def get_numbers(self, atoms):
        candidate = atoms.copy()
        if getattr(self, "elements", None) is not None:
            del candidate[[atom.index for atom in candidate if atom.symbol in self.elements]]
        return candidate.numbers

    def get_shortest_dist_vector(self, atoms):
        minimum = 10000.0
        positions = atoms.get_positions()
        low_pair = (0, 0)
        for i in range(len(atoms)):
            for j, distance in enumerate(np.linalg.norm(other - atoms[i].position) for other in positions[i:]):
                if distance != 0 and distance < minimum:
                    minimum = distance
                    low_pair = (i, j + i)
        return atoms[low_pair[0]].position - atoms[low_pair[1]].position
