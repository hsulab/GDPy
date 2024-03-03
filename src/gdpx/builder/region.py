#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import abc
import copy
import itertools

from typing import NoReturn, List, Mapping

import numpy as np

from ase import Atoms
from ase import data

from . import registers


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


class Region(abc.ABC):

    """The base class of region.

    Triclinic, sphere, cylinder.

    TODO: override __contains__?

    """

    def __init__(self, origin: List[float], *args, **kwargs):
        """"""
        self._origin = np.array(origin)

        return
    
    @abc.abstractmethod
    def from_str(command: str):
        """Init a region from the command"""

        return
    
    def get_contained_indices(self, atoms: Atoms):
        """"""
        indices_within_region = []
        for i, a in enumerate(atoms):
            if self._is_within_region(a.position):
                indices_within_region.append(i)

        return indices_within_region
    
    def get_random_positions(self, size=1, rng=np.random):
        """"""
        random_positions = []
        for i in range(size):
            ran_pos = self._get_a_random_position(rng)
            random_positions.append(ran_pos)
        random_positions = np.array(random_positions)

        return random_positions
    
    @abc.abstractmethod
    def _get_a_random_position(self, rng):
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

    @abc.abstractmethod
    def as_dict(self) -> dict:
        """"""

        return


class AutoRegion(Region):

    _curr_atoms: Atoms = None

    def __init__(self, origin: List[float]=[0.,0.,0.], atoms=None, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(origin, *args, **kwargs)

        self._curr_atoms = atoms

        return
    
    @staticmethod
    def from_str(command: str):
        """"""

        raise NotImplementedError("AutoRegion does support init_from_str.")
    
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

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "auto"

        return region_params
    

class CubeRegion(Region):

    def __init__(self, origin: List[float], boundary: List[float], *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        boundaries_ = np.array(boundary, dtype=np.float64)
        assert len(boundaries_) == 6, "Cubic region needs 6 numbers to define."
        self.boundaries = boundaries_

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        boundary = data[3:]

        return CubeRegion(origin, boundary)
    
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

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "cube"
        region_params["origin"] = self._origin.tolist()
        region_params["boundary"] = self.boundaries.tolist()

        return region_params


class SphereRegion(Region):

    def __init__(self, origin: List[float], radius: float, *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._radius = radius

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        radius = data[3]

        return SphereRegion(origin, radius)
    
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

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "sphere"
        region_params["origin"] = self._origin.tolist()
        region_params["radius"] = self._radius

        return region_params


class CylinderRegion(Region):

    """Region by a vertical cylinder.
    """

    def __init__(self, origin: List[float], radius: float, height: float, *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)

        self._radius = radius
        self._height = height

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        radius = data[3]
        height = data[4]

        return CylinderRegion(origin, radius, height)

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
        content += f"radius {self._radius} "
        content += f"height   {self._height} "

        return content

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "cylinder"
        region_params["origin"] = self._origin.tolist()
        region_params["radius"] = self._radius
        region_params["height"] = self._height

        return region_params


class LatticeRegion(Region):

    def __init__(self, origin: List[float], cell: List[float], *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._cell = np.reshape(cell, (3,3))

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        cell = data[3:]

        return LatticeRegion(origin, cell)

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
        content = f"{self.__class__.__name__}\n"
        content += f"origin\n"
        content += ("  "+"{:<12.8f}  "*3+"\n").format(*self._origin)
        content += f"cell\n"
        content += (("  "+"{:<12.8f}  "*3+"\n")*3).format(*self._cell.flatten())

        return content

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "lattice"
        region_params["origin"] = self._origin.tolist()
        region_params["cell"] = self._cell.tolist()

        return region_params


class SurfaceLatticeRegion(LatticeRegion):

    def __init__(self, origin: List[float], cell: List[float], *args, **kwargs):
        """"""
        super().__init__(origin, cell, *args, **kwargs)

        assert self._origin[0] == 0. and self._origin[1] == 0., "The x and y of origin should be ZERO."
        assert self._cell[2][0] == 0. and self._cell[2][1] == 0., "The x and y of cell 3rd vec should be ZERO."

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        cell = data[3:]

        return SurfaceLatticeRegion(origin, cell)

    def _is_within_region(self, position) -> bool:
        """"""
        is_in = False
        #vec1, vec2 = self._cell[0], self._cell[1]
        #normal = np.cross(vec1, vec2)
        #normal = normal/np.linalg.norm(normal)
        if self._origin[2] <= position[2] < self._origin[2] + self._cell[2][2]:
            is_in = super()._is_within_region(position)

        return is_in

    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "surface_lattice"
        region_params["origin"] = self._origin.tolist()
        region_params["cell"] = self._cell.tolist()

        return region_params
    

class SurfaceRegion(Region):

    def __init__(self, origin: List[float], normal: List[float], thickness: float, *args, **kwargs):
        """"""
        super().__init__(origin=origin, *args, **kwargs)
        self._normal = np.array(normal, dtype=float)
        self._thickness = thickness

        return

    @staticmethod
    def from_str(command: str):
        """"""
        data = [float(x) for x in command.strip().split()[1:]]
        origin = data[:3]
        normal = data[3:6]
        thickness = data[6:9]

        return SurfaceRegion(origin, normal, thickness)

    def _get_a_random_position(self, rng):
        """"""
        ran_frac_coord = rng.uniform(0,1,3)
        ran_pos = np.dot(ran_frac_coord, self._cell)
        ran_pos += self._origin

        raise NotImplementedError()
    
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

        raise NotImplementedError()
    
    def get_volume(self) -> float:
        """"""
        a, b, c = self._cell

        #return np.dot(np.cross(a,b), c)
        raise NotImplementedError()
    
    def __repr__(self) -> str:
        """"""
        content = f"{self.__class__.__name__}\n"
        content += f"origin\n"
        content += ("  "+"{:<12.8f}  "*3+"\n").format(*self._origin)
        #content += f"cell\n"
        #content += (("  "+"{:<12.8f}  "*3+"\n")*3).format(*self._cell.flatten())

        return content


class IntersectRegion(Region):

    #: Maximum number of attempts to get a random position.
    MAX_ATTEMPTS: int = 1000

    def __init__(self, regions, origin=np.zeros(0), *args, **kwargs):
        """Initialise an IntersectRegion.

        Args:
            regions: A List of regions.
            origin: Region origin should be always zero.
        
        """
        super().__init__(origin, *args, **kwargs)

        # NOTE: Can sub-regions be intersect ones?
        self.regions = regions
        assert len(self.regions), "IntersectRegion supports only twp sub-regions."

        self._regions = []
        for r in copy.deepcopy(regions):
            shape = r.pop("method", None)
            curr_region = registers.create(
                "region", shape, convert_name=True, **r
            )
            self._regions.append(curr_region)

        return

    @staticmethod
    def from_str(command: str):

        raise NotImplementedError()
    
    def _get_a_random_position(self, rng):
        """"""
        for i in range(self.MAX_ATTEMPTS):
            r1, r2 = self._regions
            ran_pos = r1._get_a_random_position(rng)
            if not r2._is_within_region(ran_pos):
                break
        else:
            raise RuntimeError("Fail to get a random position in the IntersectRegion.")

        return ran_pos
    
    def _is_within_region(self, position) -> bool:
        """"""

        raise NotImplementedError()
    
    def get_volume(self) -> float:
        """"""

        raise NotImplementedError()
    
    def as_dict(self):
        """"""
        region_params = {}
        region_params["method"] = "intersect"
        region_params["origin"] = self._origin.tolist()
        region_params["regions"] = self.regions

        return region_params


if __name__ == "__main__":
    ...