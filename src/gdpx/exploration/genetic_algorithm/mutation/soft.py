"""ASE-GA-compatible soft mutation implemented by GDPy."""

from __future__ import annotations

import inspect
import json

import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from scipy.spatial.distance import cdist

from gdpx.structures.geometry.ga import atoms_too_close, gather_atoms_by_tag

from ..core import OffspringCreator


class TagFilter:
    """Constrain atoms sharing a tag to translate as rigid moieties."""

    def __init__(self, atoms):
        self.atoms = atoms
        gather_atoms_by_tag(self.atoms)
        self.tags = self.atoms.get_tags()
        self.unique_tags = np.unique(self.tags)
        self.n = len(self.unique_tags)

    def get_positions(self):
        positions = self.atoms.get_positions()
        centers = np.zeros((self.n, 3))
        for i, tag in enumerate(self.unique_tags):
            centers[i] = np.average(positions[np.where(self.tags == tag)], axis=0)
        return centers

    def set_positions(self, positions, **kwargs):
        centers = self.get_positions()
        all_positions = self.atoms.get_positions()
        assert np.all(np.shape(positions) == np.shape(centers))
        for i, tag in enumerate(self.unique_tags):
            selected = np.where(self.tags == tag)
            all_positions[selected] += positions[i] - centers[i]
        self.atoms.set_positions(all_positions, **kwargs)

    def get_forces(self, *args, **kwargs):
        atomic_forces = self.atoms.get_forces()
        forces = np.zeros((self.n, 3))
        for i, tag in enumerate(self.unique_tags):
            forces[i] = np.sum(atomic_forces[np.where(self.tags == tag)], axis=0)
        return forces

    def get_masses(self):
        atomic_masses = self.atoms.get_masses()
        masses = np.zeros(self.n)
        for i, tag in enumerate(self.unique_tags):
            masses[i] = np.sum(atomic_masses[np.where(self.tags == tag)])
        return masses

    def __len__(self):
        return self.n


class PairwiseHarmonicPotential:
    """Base for pairwise potentials with harmonic bond stretching."""

    def __init__(self, atoms, rcut=10.0):
        self.atoms = atoms
        self.pos0 = atoms.get_positions()
        self.rcut = rcut
        self.nl = NeighborList(
            [self.rcut / 2.0] * len(atoms),
            skin=0.0,
            bothways=True,
            self_interaction=False,
        )
        self.nl.update(atoms)
        self.calculate_force_constants()

    def calculate_force_constants(self):
        raise NotImplementedError

    def get_forces(self, atoms):
        positions = atoms.get_positions()
        cell = atoms.get_cell()
        forces = np.zeros_like(positions)
        for i in range(len(atoms)):
            indices, offsets = self.nl.get_neighbors(i)
            neighbor_positions = positions[indices] + np.dot(offsets, cell)
            distances = cdist(neighbor_positions, [positions[i]])
            directions = (neighbor_positions - positions[i]) / distances
            reference_positions = self.pos0[indices] + np.dot(offsets, cell)
            reference_distances = cdist(reference_positions, [self.pos0[i]])
            changes = distances - reference_distances
            forces[i] = np.dot(self.force_constants[i].T, changes * directions)
        return forces


def get_number_of_valence_electrons(atomic_number):
    groups = [[], [1, 3, 11, 19, 37, 55, 87], [2, 4, 12, 20, 38, 56, 88], [21, 39, 57, 89]]
    for i in range(9):
        groups.append(i + np.array([22, 40, 72, 104]))
    for i in range(6):
        groups.append(i + np.array([5, 13, 31, 49, 81, 113]))
    for i, group in enumerate(groups):
        if atomic_number in group:
            return i if i < 13 else i - 10
    raise ValueError(f"Atomic number {atomic_number} is not included in the valence dataset.")


class BondElectroNegativityModel(PairwiseHarmonicPotential):
    """Harmonic potential using the bond-electronegativity force model."""

    def calculate_force_constants(self):
        cell = self.atoms.get_cell()
        positions = self.atoms.get_positions()
        numbers = self.atoms.get_atomic_numbers()
        coordination_norms = []
        valence_states = []
        radii = []
        for i in range(len(self.atoms)):
            indices, offsets = self.nl.get_neighbors(i)
            neighbor_positions = positions[indices] + np.dot(offsets, cell)
            distances = cdist(neighbor_positions, [positions[i]])
            radius = covalent_radii[numbers[i]]
            coordination = 0.0
            for j, index in enumerate(indices):
                delta = distances[j] - radius - covalent_radii[numbers[index]]
                coordination += np.exp(-delta / 0.37)
            coordination_norms.append(coordination)
            valence_states.append(get_number_of_valence_electrons(numbers[i]))
            radii.append(radius)

        self.force_constants = []
        for i in range(len(self.atoms)):
            indices, offsets = self.nl.get_neighbors(i)
            neighbor_positions = positions[indices] + np.dot(offsets, cell)
            distances = cdist(neighbor_positions, [positions[i]])[:, 0]
            constants = []
            for j, index in enumerate(indices):
                delta = distances[j] - radii[i] - radii[index]
                chi_i = 0.481 * valence_states[i] / (radii[i] + 0.5 * delta)
                chi_j = 0.481 * valence_states[index] / (radii[index] + 0.5 * delta)
                coordination_i = coordination_norms[i] / np.exp(-delta / 0.37)
                coordination_j = coordination_norms[index] / np.exp(-delta / 0.37)
                constants.append(np.sqrt(chi_i * chi_j / (coordination_i * coordination_j)))
            self.force_constants.append(np.array(constants))


class SoftMutation(OffspringCreator):
    """Displace a structure along its lowest unused non-translational mode."""

    descriptor = "SoftMutation"
    min_inputs = 1
    supports_fragment_preservation = True
    fragment_mode_configurable = True

    def __init__(
        self,
        blmin,
        bounds=(0.5, 2.0),
        calculator=BondElectroNegativityModel,
        rcut=10.0,
        used_modes_file="used_modes.json",
        use_tags=False,
        verbose=False,
        rng=None,
    ):
        super().__init__(verbose=verbose, rng=rng)
        self.blmin = blmin
        self.bounds = bounds
        self.calc = calculator
        self.rcut = rcut
        self.used_modes_file = used_modes_file
        self.use_tags = use_tags
        self.used_modes = {}
        if self.used_modes_file is not None:
            try:
                self.read_used_modes(self.used_modes_file)
            except OSError:
                pass

    def _get_hessian(self, atoms, dx):
        positions = atoms.get_positions()
        hessian = np.zeros((3 * len(atoms), 3 * len(atoms)))
        for i in range(3 * len(atoms)):
            row = np.zeros(3 * len(atoms))
            for direction in (-1, 1):
                displacement = np.zeros(3)
                displacement[i % 3] = direction * dx
                displaced = positions.copy()
                displaced[i // 3] += displacement
                atoms.set_positions(displaced)
                row += -direction * atoms.get_forces().flatten()
            hessian[i] = row / (2.0 * dx)
        hessian = 0.5 * (hessian + hessian.T)
        atoms.set_positions(positions)
        return hessian

    def _calculate_normal_modes(self, atoms, dx=0.02, massweighing=False):
        hessian = self._get_hessian(atoms, dx)
        if massweighing:
            masses = np.array([np.repeat(atoms.get_masses() ** -0.5, 3)])
            hessian *= masses * masses.T
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        return {value: eigenvectors[:, i] for i, value in enumerate(eigenvalues)}

    def animate_mode(self, atoms, mode, nim=30, amplitude=1.0):
        positions = atoms.get_positions()
        mode = mode.reshape(np.shape(positions))
        animation = []
        for i in range(nim):
            image = atoms.copy()
            image.positions = positions + amplitude * mode * np.sin(i * 2 * np.pi / nim)
            animation.append(image)
        return animation

    def read_used_modes(self, filename):
        with open(filename) as stream:
            modes = json.load(stream)
        self.used_modes = {int(key): modes[key] for key in modes}

    def write_used_modes(self, filename):
        with open(filename, "w") as stream:
            json.dump(self.used_modes, stream)

    def mutate(self, atoms):
        filtered = atoms.copy()
        if inspect.isclass(self.calc):
            assert issubclass(self.calc, PairwiseHarmonicPotential)
            calculator = self.calc(atoms, rcut=self.rcut)
        else:
            calculator = self.calc
        filtered.calc = calculator
        if self.use_tags:
            filtered = TagFilter(filtered)

        positions = filtered.get_positions()
        modes = self._calculate_normal_modes(filtered)
        keys = np.array(sorted(modes))
        index = 3
        confid = atoms.info["confid"]
        if confid in self.used_modes:
            while index in self.used_modes[confid]:
                index += 1
            self.used_modes[confid].append(index)
        else:
            self.used_modes[confid] = [index]
        if self.used_modes_file is not None:
            self.write_used_modes(self.used_modes_file)

        mode = modes[keys[index]].reshape(np.shape(positions))
        mutant = atoms.copy()
        amplitude = 0.0
        increment = 0.1
        direction = 1
        largest_norm = np.max(np.apply_along_axis(np.linalg.norm, 1, mode))

        def expand(positions_to_expand):
            if isinstance(filtered, TagFilter):
                filtered.set_positions(positions_to_expand)
                return filtered.atoms.get_positions()
            return positions_to_expand

        while amplitude * largest_norm < self.bounds[1]:
            new_positions = expand(positions + direction * amplitude * mode)
            mutant.set_positions(new_positions)
            mutant.wrap()
            if atoms_too_close(mutant, self.blmin, use_tags=self.use_tags):
                amplitude -= increment
                mutant.set_positions(expand(positions + direction * amplitude * mode))
                mutant.wrap()
                break
            if direction == 1:
                direction = -1
            else:
                direction = 1
                amplitude += increment
        return None if amplitude * largest_norm < self.bounds[0] else mutant

    def get_new_individual(self, parents):
        child = self.mutate(parents[0])
        if child is None:
            return None, "mutation: soft"
        child = self.initialize_individual(parents[0], child)
        child.info["data"]["parents"] = [parents[0].info["confid"]]
        return self.finalize_individual(child), "mutation: soft"
