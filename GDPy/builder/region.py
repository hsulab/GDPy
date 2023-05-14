#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import time
import itertools
from typing import NoReturn, List, Mapping

import numpy as np

from ase import Atom, Atoms
from ase import data, units
from ase.io import read, write
from ase.formula import Formula
from ase.neighborlist import natural_cutoffs, NeighborList
from ase.ga.utilities import closest_distances_generator

from GDPy.core.register import registers
from GDPy.builder.species import build_species

def estimate_chemical_potential(
    temperature: float, 
    pressure: float, # pressure, 1 bar
    total_energy: float,
    zpe: float,
    dU: float,
    dS: float, # entropy
    coef: float = 1.0
) -> float:
    """Estimate Chemical Potential

    Examples:
        >>> O2 by ReaxFF
        >>>     molecular energy -5.588 atomic energy -0.109
        >>> O2 by vdW-DF spin-polarised 
        >>>     molecular energy -9.196 atomic energy -1.491
        >>>     ZPE 0.09714 
        >>>     dU 8.683 kJ/mol (exp)
        >>>     entropy@298.15K 205.147 J/mol (exp)
        >>> For two reservoirs, O and Pt
        >>> Pt + O2 -> aPtO2
        >>> mu_Pt = E_aPtO2 - G_O2
        >>> FreeEnergy = E_DFT + ZPE + U(T) + TS + pV

    References:
        Thermodynamic Data  https://janaf.nist.gov
    """
    kJm2eV = units.kJ / units.mol # from kJ/mol to eV
    # 300K, PBE-ZPE, experimental data https://janaf.nist.gov
    temp_correction = zpe + (dU*kJm2eV) - temperature*(dS/1000*kJm2eV)
    pres_correction = units.kB*temperature*np.log(pressure/1.0) # eV
    chemical_potential = coef*(
        total_energy + temp_correction + pres_correction
    )

    return chemical_potential

def get_tags_per_species(atoms: Atoms) -> Mapping[str,Mapping[int,List[int]]]:
    """Get tags per species.

    Example:

        .. code-block:: python

            >>> atoms = Atoms("PtPtPtCOCO")
            >>> tags = [0, 0, 0, 1, 1, 2, 2]
            >>> atoms.set_tags(tags)
            >>> get_tags_per_species(atoms)
            >>> {'Pt3': {0: [0,1,2]}, 'CO': {1: [3,4], 2: [5,6]}}

    """

    tags = atoms.get_tags() # default is all zero

    tags_dict = {} # species -> tag list
    for key, group in itertools.groupby(enumerate(tags), key=lambda x:x[1]):
        cur_indices = [x[0] for x in group]
        #print(key, " :", cur_indices)
        cur_atoms = atoms[cur_indices]
        formula = cur_atoms.get_chemical_formula()
        #print(formula)
        #print(key)
        if formula not in tags_dict:
            tags_dict[formula] = []
        tags_dict[formula].append([key, cur_indices])

    return tags_dict

class Reservoir():

    def __init__(self, species, temperature=300., pressure=1.):
        """"""
        self.temperature = temperature
        self.pressure = pressure

        # - parse reservoir
        self.exparts, self.chem_pot = [], {} # particle names, particle mus

        for name, data in species.items():
            self.exparts.append(name)
            if isinstance(data, dict):
                self.chem_pot[name] = estimate_chemical_potential(
                    temperature=self.temperature, pressure=self.pressure,
                    **data
                )
            else:
                # directly offer chemical potential
                self.chem_pot[name] = float(data)

        # statistical mechanics
        self.species_mass = {}
        self.beta = {}
        self.cubic_wavelength = {}
        for expart in self.exparts:
            self.species_mass[expart], self.beta[expart], self.cubic_wavelength[expart] = self.compute_thermo_wavelength(
                expart, self.temperature
            )
        
        self._parse_type_list()

        return
    
    def _parse_type_list(self):
        """"""
        type_count = {}
        for s in self.exparts:
            count = Formula(s).count()
            for k, v in count.items():
                if k in type_count:
                    type_count[k] += v
                else:
                    type_count[k] = v
        self.type_list = list(type_count.keys())

        return

    @staticmethod
    def compute_thermo_wavelength(expart: str, temperature: float):
        # - beta
        kBT_eV = units.kB * temperature
        beta = 1./kBT_eV # 1/(kb*T), eV

        # - cubic thermo de broglie 
        hplanck = units._hplanck # J/Hz = kg*m2*s-1
        #_mass = np.sum([data.atomic_masses[data.atomic_numbers[e]] for e in expart]) # g/mol
        _species = build_species(expart)
        _species_mass = np.sum(_species.get_masses())
        #print("species mass: ", _mass)
        _mass = _species_mass * units._amu
        kbT_J = kBT_eV * units._e # J = kg*m2*s-2
        cubic_wavelength = (hplanck/np.sqrt(2*np.pi*_mass*kbT_J)*1e10)**3 # thermal de broglie wavelength

        return _species_mass, beta, cubic_wavelength
    
    def __str__(self):
        """"""
        content = "----- Reservoir -----\n"
        content += "Temperature %.4f [K] Pressure %.4f [atm]\n" %(self.temperature, self.pressure)
        for expart in self.exparts:
            content += "--- %s ---\n" %expart
            content += f"Mass {self.species_mass[expart]}\n"
            content += "Beta %.4f [eV-1]\n" %(self.beta[expart])
            content += "Cubic Thermal de Broglie Wavelength %f\n" %self.cubic_wavelength[expart]
            content += "Chemical Potential of is %.4f [eV]\n" %self.chem_pot[expart]

        return content


class Region(abc.ABC):

    """The base class of region.

    Triclinic, sphere, cylinder.

    TODO: override __contains__?

    """

    def __init__(self, origin: List[float], *args, **kwargs):
        """"""
        self._origin = np.array(origin)

        return
    
    def get_random_positions(self, size=1, rng=np.random):
        """"""
        random_positions = []
        for i in range(size):
            ran_pos = self._get_a_random_position(rng)
            random_positions.append(ran_pos)
        random_positions = np.array(random_positions)

        return random_positions
    
    @abc.abstractmethod
    def _get_a_random_position(self,rng):
        """"""

        return
    
    @abc.abstractmethod
    def _is_within_region(self, position) -> bool:
        """Positions are normally atomic positions or molecular centre positions."""

        return
    
    def get_tags_dict(self, atoms: Atoms):
        """Get tags dict for atoms within the (entire) system"""

        return get_tags_per_species(atoms)
    
    def get_contained_tags_dict(self, atoms: Atoms, tags_dict: dict=None) -> Mapping[str,List[int]]:
        """"""
        # - find tags and compute cops
        if tags_dict is None:
            tags_dict_within_system = get_tags_per_species(atoms)
        else:
            tags_dict_within_system = tags_dict
        
        # - NOTE: get wrapped positions due to PBC
        positions = copy.deepcopy(atoms.get_positions(wrap=True))

        cops_dict = {}
        for key, tags_and_indices in tags_dict_within_system.items():
            for tag, curr_indices in tags_and_indices:
                cur_positions = positions[curr_indices]
                # TODO: Considering PBC, surface may have molecules across boundaries.
                cop = np.average(cur_positions, axis=0)
                if key not in cops_dict:
                    cops_dict[key] = []
                cops_dict[key].append([tag, cop])
        
        # - check 
        tags_dict_within_region = {}
        for key, tags_and_cops in cops_dict.items():
            #print(tags_and_cops)
            for tag, cop in tags_and_cops:
                if self._is_within_region(cop):
                    if key in tags_dict_within_region:
                        tags_dict_within_region[key].append(tag)
                    else:
                        tags_dict_within_region[key] = [tag]
 
        return tags_dict_within_region
    
    def get_empty_volume(self, atoms: Atoms, tags_dict: dict=None, ratio: float=1.0) -> float:
        """Empty volume = Region volume - total volume of atoms within region.

        This is not always correct since all atoms in the fragment are considered 
        within the region if their cop is in the region.
        
        """
        # - get atom indices with given tags
        if tags_dict is None:
            tags_dict = self.get_contained_tags_dict(atoms)
        tags_within_region = []
        for key, tags in tags_dict.items():
            tags_within_region.extend(tags)
        atomic_indices = [i for i, t in enumerate(atoms.get_tags()) if t in tags_within_region]

        # - get atoms' radii
        radii = np.array([data.covalent_radii[data.atomic_numbers[atoms[i].symbol]] for i in atomic_indices])
        radii *= ratio

        atoms_volume = np.sum([4./3.*np.pi*r**3 for r in radii])

        return self.get_volume() - atoms_volume
    
    @abc.abstractmethod
    def get_volume(self) -> float:
        """"""

        return
    
    #def __eq__(self, other):
    #    """"""
    #    return all(self.__dict__ == other.__dict__)

@registers.region.register
class AutoRegion(Region):

    _curr_atoms: Atoms = None

    def __init__(self, origin: List[float]=[0.,0.,0.], atoms=None, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(origin, *args, **kwargs)

        self._curr_atoms = atoms

        return
    
    def _get_a_random_position(self, rng=np.random):
        """"""
        if self._curr_atoms is None:
            raise RuntimeError(f"No atoms is attached to {self.__class__.__name__}")
        
        frac_pos = rng.uniform(0,1,3)
        ran_pos = np.dot(frac_pos, self._curr_atoms.get_cell())
        
        return ran_pos
    
    def _is_within_region(self, position) -> bool:
        """"""
        if self._curr_atoms is None:
            raise RuntimeError(f"No atoms is attached to {self.__class__.__name__}")

        is_in = False
        pos_ = position - self._origin
        frac_pos_ = np.dot(np.linalg.inv(self._curr_atoms.get_cell().T), pos_)
        if (
            0. <= np.modf(frac_pos_[0])[0] < 1. and
            0. <= np.modf(frac_pos_[1])[0] < 1. and
            0. <= np.modf(frac_pos_[2])[0] < 1.
        ):
            is_in = True

        return is_in
    
    def get_volume(self) -> float:
        """"""
        if self._curr_atoms is None:
            raise RuntimeError(f"No atoms is attached to {self.__class__.__name__}")

        return self._curr_atoms.get_volume()
    
@registers.region.register
class CubeRegion(Region):

    def __init__(self, origin: List[float], boundary: List[float], *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        boundaries_ = np.array(boundary, dtype=np.float64)
        assert len(boundaries_) == 6, "Cubic region needs 6 numbers to define."
        self.boundaries = boundaries_

        return
    
    def _get_a_random_position(self, rng=np.random):
        """"""
        boundaries_ = copy.deepcopy(self.boundaries)
        boundaries_ = np.reshape(boundaries_, (2,3))

        ran_frac_pos = rng.uniform(0,1,3)
        ran_pos = (
            self._origin + boundaries_[0,:] +
            (boundaries_[0,:] - boundaries_[1,:])*ran_frac_pos
        )

        return ran_pos

    def _is_within_region(self, position) -> bool:
        """"""
        ox, oy, oz = self._origin
        (xl, yl, zl, xh, yh, zh) = self.boundaries

        position = np.array(position)

        is_in = False
        if (
            (ox+xl <= position[0] <= ox+xh) and
            (oy+yl <= position[1] <= oy+yh) and
            (oz+zl <= position[2] <= oz+zh)
        ):
            is_in = True

        return is_in

    def get_volume(self) -> float:
        """"""
        (xl, yl, zl, xh, yh, zh) = self.boundaries

        return (xh-xl)*(yh-yl)*(zh-zl)

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"origin {self.boundaries[:3]} "
        content += f"lmin {self.boundaries[3:6]} "
        content += f"lmax {self.boundaries[6:9]} "

        return content

@registers.region.register
class SphereRegion(Region):

    def __init__(self, origin: List[float], radius: float, *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._radius = radius

        return
    
    def _get_a_random_position(self, rng):
        """"""
        ran_coord = rng.uniform(0,1,3)
        polar = np.array([self._radius, np.pi, np.pi]) * ran_coord
        r, theta, phi = polar

        ran_pos = np.array(
            [
                r*np.sin(theta)*np.cos(phi),
                r*np.sin(theta)*np.sin(phi),
                r*np.cos(theta)
            ]
        )
        ran_pos += self._origin

        return ran_pos
    
    def _is_within_region(self, position) -> bool:
        """"""
        is_in = False

        position = np.array(position)
        distance = np.linalg.norm(position-self._origin)
        if distance <= self._radius:
            is_in = True

        return is_in
    
    def get_volume(self):
        """"""

        return 4./3.*np.pi*self._radius**3

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"radius {self._radius} "
        content += f"volume {self.get_volume()} "

        return content

@registers.region.register
class CylinderRegion(Region):

    """Region by a vertical cylinder.
    """

    def __init__(self, origin: List[float], radius: float, height: float, *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)

        self._radius = radius
        self._height = height

        return

    def _get_a_random_position(self, rng):
        """"""
        r, theta, h = rng.uniform(0,1,3) # r, theta, h

        ran_pos = np.array(
            [
                r*np.cos(theta),
                r*np.sin(theta),
                h
            ]
        )
        ran_pos += self._origin

        return ran_pos
    
    def _is_within_region(self, position) -> bool:
        """"""
        ox, oy, oz = self._origin

        is_in = False
        if oz <= position[2] <= oz+self._height:
            distance = np.linalg.norm(position[:2] - self._origin[:2])
            if distance <= self._radius:
                is_in = True

        return is_in
    
    def get_volume(self) -> float:
        """"""
        return np.pi*self._radius**2*self._height

    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"radii {self._radius} "
        content += f"height   {self._height} "

        return content

@registers.region.register
class LatticeRegion(Region):

    def __init__(self, origin: List[float], cell: List[float], *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._cell = np.reshape(cell, (3,3))

        return

    def _get_a_random_position(self, rng):
        """"""
        ran_frac_coord = rng.uniform(0,1,3)
        ran_pos = np.dot(ran_frac_coord, self._cell)
        ran_pos += self._origin

        return ran_pos
    
    def _is_within_region(self, position) -> bool:
        """"""
        is_in = False
        pos_ = position - self._origin
        frac_pos_ = np.dot(np.linalg.inv(self._cell.T), pos_)
        if (
            0. <= np.modf(frac_pos_[0])[0] < 1. and
            0. <= np.modf(frac_pos_[1])[0] < 1. and
            0. <= np.modf(frac_pos_[2])[0] < 1.
        ):
            is_in = True

        return is_in
    
    def get_volume(self) -> float:
        """"""
        a, b, c = self._cell

        return np.dot(np.cross(a,b), c)
    
    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__} "
        content += f"origin {self._origin} "
        content += f"cell   {self._cell} "

        return content


class ReducedRegion():
    """
    spherical or cubic box
    """

    MAX_RANDOM_ATTEMPS = 1000
    MAX_NUM_PER_SPECIES = 10000

    _reservoir = None # determine whether insert or delete is possible

    def __init__(
        self, 
        atoms, # init atoms
        reservoir,
        caxis: list, # min and max in z-axis
        covalent_ratio = [0.8, 2.0],
        max_movedisp = 2.0,
        use_rotation = False,
        rng = None
    ):
        """"""
        # - parse type list
        self.atoms = atoms
        self.reservoir = reservoir

        # - find all possible elements
        chemical_symbols = self.atoms.get_chemical_symbols()
        chemical_symbols.extend(self.reservoir.type_list)
        type_list = list(set(chemical_symbols))

        # cutoff radii
        assert len(covalent_ratio) == 2, "covalent ratio should have min and max"
        self.covalent_min = covalent_ratio[0]
        self.covalent_max = covalent_ratio[1]

        unique_atomic_numbers = [data.atomic_numbers[a] for a in type_list]
        self.blmin = closest_distances_generator(
            atom_numbers=unique_atomic_numbers,
            ratio_of_covalent_radii = self.covalent_min # be careful with test too far
        )

        # move operation
        self.max_movedisp = max_movedisp

        self.use_rotation = use_rotation

        # - region
        self.cell = self.atoms.get_cell(complete=True)
        
        # TODO: assert zaxis should be perpendiculer to xy plane
        assert len(caxis) == 2
        self.cmin, self.cmax = caxis
        self.cell[2,2] = self.cmax 

        self.cvect = self.cell[2] / np.linalg.norm(self.cell[2])
        self.volume = np.dot(
            self.cvect*(self.cmax-self.cmin), np.cross(self.cell[0], self.cell[1])
        )

        # - find existed exchangeable particles
        auto_tag = True
        for expart in self.reservoir.exparts:
            natoms_per_species = sum(Formula(expart).count().values())
            if natoms_per_species > 1:
                # TODO: use graph to determine?
                auto_tag = False
                break
        else:
            pass
        if auto_tag:
            if "tags" not in self.atoms.arrays:
                tag_list = {k:[] for k in self.reservoir.exparts}
                cur_tags = self.atoms.get_tags()
                for i, atom in enumerate(self.atoms):
                    if atom.symbol in self.reservoir.exparts and self._is_inside_region(atom.position):
                        i_symbol = self.reservoir.exparts.index(atom.symbol)
                        cur_tag = self.MAX_NUM_PER_SPECIES*(i_symbol+1)+len(tag_list[atom.symbol])
                        cur_tags[i] = cur_tag
                        tag_list[atom.symbol].append(cur_tag)
                self.atoms.set_tags(cur_tags)
        else:
            if "tags" not in self.atoms.arrays:
                self.atoms.set_tags(0)

        # - set tag list
        tag_list = {k:[] for k in self.reservoir.exparts}
        tag_symbol = zip(self.atoms.get_tags(), self.atoms.get_chemical_symbols())
        for k, g in itertools.groupby(tag_symbol, lambda d: d[0]):
            symbols = sorted([x[1] for x in g])
            formula = str(Formula("".join(symbols)).convert("hill"))
            if formula in tag_list:
                tag_list[formula].append(k)
        self.tag_list = tag_list

        # - random
        self.rng = rng

        return
    
    @property
    def reservoir(self):

        return self._reservoir
    
    @reservoir.setter
    def reservoir(self, reservoir_):
        self._reservoir = reservoir_
        return

    def random_position_neighbour(
        self, 
        new_atoms: Atoms, 
        species_indices,
        operation # move or insert
    ):
        """"""
        print("\n---- Generate Random Position -----\n")
        st = time.time()

        # - find tag atoms
        # record original position of idx_pick
        species = new_atoms[species_indices]

        #org_pos = new_atoms[idx_pick].position.copy() # original position
        # TODO: deal with pbc, especially for move step
        org_com = np.mean(species.positions.copy(), axis=0)
        org_positions = species.positions.copy()

        chemical_symbols = new_atoms.get_chemical_symbols()
        nl = NeighborList(
            self.covalent_max*np.array(natural_cutoffs(new_atoms)), 
            skin=0.0, self_interaction=False, bothways=True
        )
        for i in range(self.MAX_RANDOM_ATTEMPS): # maximum number of attempts
            print(f"random attempt {i}")
            if operation == "insert":
                # NOTE: z-axis should be perpendicular
                ran_frac_pos = self.rng.uniform(0,1,3)
                ran_pos = np.dot(ran_frac_pos, self.cell) 
                ran_pos[2] = self.cmin + ran_frac_pos[2] * (self.cmax-self.cmin)
            elif operation == "move":
                # get random motion vector
                rsq = 1.1
                while (rsq > 1.0):
                    rvec = 2*self.rng.uniform(size=3) - 1.0
                    rsq = np.linalg.norm(rvec)
                ran_pos = org_com + rvec*self.max_movedisp
            # update position of idx_pick
            #new_atoms.positions[species_indices] += new_vec
            species_ = species.copy()
            # --- Apply a random rotation to multi-atom blocks
            if self.use_rotation and len(species_indices) > 1:
                phi, theta, psi = 360 * self.rng.uniform(0,1,3)
                species_.euler_rotate(
                    phi=phi, theta=0.5 * theta, psi=psi,
                    center=org_com
                )
            # -- Apply translation
            new_vec = ran_pos - org_com
            species_.translate(new_vec)
            new_atoms.positions[species_indices] = species_.positions.copy()
            # use neighbour list
            if not self.check_overlap_neighbour(nl, new_atoms, chemical_symbols, species_indices):
                print(f"succeed to random after {i+1} attempts...")
                #print("original position: ", org_com)
                #print("random position: ", ran_pos)
                break
            new_atoms.positions[species_indices] = org_positions
        else:
            new_atoms = None

        et = time.time()
        print("used time: ", et - st)

        return new_atoms
    
    def check_overlap_neighbour(
        self, nl, new_atoms, chemical_symbols, 
        species_indices
    ):
        """ use neighbour list to check newly added atom is neither too close or too
            far from other atoms
        """
        overlapped = False
        nl.update(new_atoms)
        for idx_pick in species_indices:
            print("- check index ", idx_pick)
            indices, offsets = nl.get_neighbors(idx_pick)
            if len(indices) > 0:
                print("nneighs: ", len(indices))
                # should close to other atoms
                for ni, offset in zip(indices, offsets):
                    dis = np.linalg.norm(new_atoms.positions[idx_pick] - (new_atoms.positions[ni] + np.dot(offset, self.cell)))
                    pairs = [chemical_symbols[ni], chemical_symbols[idx_pick]]
                    pairs = tuple([data.atomic_numbers[p] for p in pairs])
                    #print("distance: ", ni, dis, self.blmin[pairs])
                    if dis < self.blmin[pairs]:
                        overlapped = True
                        break
            else:
                # TODO: is no neighbours valid?
                print("no neighbours, being isolated...")
                overlapped = True
                # TODO: try rotate?
                break

        return overlapped
    
    def calc_acc_volume(self, atoms) -> float:
        """calculate acceptable volume"""
        atoms_inside = [atom for atom in atoms if atom.position[2] > self.cmin]
        print("number of atoms inside the region", len(atoms_inside))
        radii = [data.covalent_radii[data.atomic_numbers[atom.symbol]] for atom in atoms_inside]
        atoms_volume = np.sum([4./3.*np.pi*r**3 for r in radii])

        acc_volume = self.volume - atoms_volume # A^3

        return acc_volume
    
    def _is_inside_region(self, position):
        """"""
        is_inside = False
        if self.cmin < position[2] < self.cmax:
            is_inside = True

        return is_inside
    
    def __str__(self):
        """"""
        content = "----- Region -----\n"
        content += "cell \n"
        for i in range(3):
            content += ("{:<8.4f}  "*3+"\n").format(*list(self.cell[i, :]))
        content += "covalent ratio: {}  {}\n".format(self.covalent_min, self.covalent_max)
        content += "maximum movedisp: {}\n".format(self.max_movedisp)
        content += "use rotation: {}\n".format(self.use_rotation)

        content += "tags\n"
        for expart, expart_tags in self.tag_list.items():
            content += f"{expart:<8s} number: {len(expart_tags):>8d}\n"

        content += "\n"
        content += str(self.reservoir)

        return content


if __name__ == "__main__":
    ...