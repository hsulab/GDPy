#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import dataclasses
from typing import NoReturn, Union, List, Callable

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.datatype import isAtomsFrames, isTrajectories
from GDPy.selector.selector import AbstractSelector
from GDPy.selector.cur import boltz_selection

@dataclasses.dataclass
class PropertyItem:

    #: Property name that can be found in atoms.info or atoms.arrays.
    name: str

    #: metric config...
    metric: Union[str,List[str]] = None

    #: List of functions, min, max, average and ...
    _metric: List[Callable] = dataclasses.field(init=False, default_factory=list)

    # metric_params

    #: Sparsifiction method.
    sparsify: str = "filter"

    #: Whether perform sparsifiction on each traj separately.
    trajwise: bool = False

    #: Whether reverse the sparsifiction behaviour.
    reverse: bool = False

    range: List[float] = dataclasses.field(default_factory=lambda: [None,None])
    pmin: float = dataclasses.field(init=False, default=-np.inf)
    pmax: float = dataclasses.field(init=False, default= np.inf)

    #: Boltzmann temperature (eV).
    kBT: float = None

    # expression
    # weight

    #worker_config: dataclasses.InitVar[str] = None

    def __post_init__(self):
        """"""
        # - bound
        bounds_ = self.range
        if bounds_[0] is None:
            bounds_[0] = -np.inf
        if bounds_[1] is None:
            bounds_[1] = np.inf
        assert bounds_[0] < bounds_[1], f"{self.name} has invalid bounds..."
        self.pmin, self.pmax = bounds_

        # - metric
        if self.metric is not None:
            if isinstance(self.metric, str):
                metric_config = [self.metric]
            else:
                # a list of metric function names
                metric_config = self.metric
        
            for metric_name in metric_config:
                if metric_name == "fabs":
                    metric_func = np.fabs
                elif metric_name == "max":
                    metric_func = np.max
                elif metric_name == "min":
                    metric_func = np.min
                else:
                    raise NotImplementedError(f"Unknown metric function {metric_name}.")
                self._metric.append(metric_func)
        else:
            self._metric = []
        
        # - sparsify
        assert self.sparsify in ["filter", "sort", "boltz"], f"Unknown sparsification {self.sparsify}."

        return
    
    def _convert_raw_(self, raws_, weights_=None):
        """Convert raw values by the metric."""
        if len(self._metric) > 0:
            converts_ = []
            for raw_ in raws_:
                convert_ = raw_ # NOTE: copy?
                for metric_func in self._metric:
                    convert_ = metric_func(convert_)
                converts_.append(convert_)
        else:
            converts_ = raws_

        return converts_

    #def __repr__(self) -> str:
    #    """"""
    #    content = f"{self.name}:\n"
    #    content += f"  range: {self.pmin} - {self.pmax}\n"
    #    content += f"  sparsify: {self.sparsify}\n"

    #    return content


class PropertyBasedSelector(AbstractSelector):

    """Select structures based on structural properties.

    Each structure (trajectory) is represented by a float property.

    """

    name = "property"

    default_parameters = dict(
        properties = [],
        worker = None, # compute properties on-the-fly
        number = [4, 0.2]
    )

    def __init__(self, directory="./", *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(directory, *args, **kwargs)

        # - convert properties
        prop_items = []
        for name, params in self.properties.items():
            prop_item = PropertyItem(name=name, **params)
            prop_items.append(prop_item)
        self._prop_items = prop_items

        return

    def _select_indices(self, frames: List[Atoms], *args, **kwargs) -> List[int]:
        """Return selected indices."""
        # - get property values
        #   NOTE: properties should be pre-computed...
        nframes = len(frames)
        if nframes > 0:
            prev_indices = list(range(nframes))
            for prop_item in self._prop_items:
                self.pfunc(str(prop_item))
                # -- each structure is represented by one float value
                #    get per structure values
                prop_vals = self._extract_property(frames, prop_item)
                # --
                scores, prev_indices = self._sparsify(prop_item, prop_vals, prev_indices)
                self.pfunc(f"nselected: {len(prev_indices)}")
                selected_indices = prev_indices # frames [0,1,2,3] or trajectories [[0,0],[0,2]]
                # --
                if not (len(prev_indices) > 0):
                    break

            # - add score to atoms
            #   only save scores from last property
            for score, i in zip(scores, prev_indices):
                frames[i].info["score"] = score
        else:
            selected_indices = []

        return selected_indices
    
    def _extract_property(self, frames: List[Atoms], prop_item: PropertyItem):
        """Extract property values from frames.

        Returns:
            property values: List[float] or 1d-np.array.

        """
        prop_vals = []
        for atoms in frames:
            if prop_item.name == "atomic_energy":
                # TODO: move this part to PropertyItem?
                energy = atoms.get_potential_energy()
                natoms = len(atoms)
                atoms_property = energy/natoms
            elif prop_item.name == "energy":
                energy = atoms.get_potential_energy()
                atoms_property = energy
            elif prop_item.name == "forces":
                forces = atoms.get_forces(apply_constraint=True)
                atoms_property = forces
            else:
                # -- any property stored in atoms.info
                #    e.g. max_devi_f
                atoms_property = atoms.info.get(prop_item.name, None)
                if atoms_property is None:
                    atoms_property = atoms.arrays.get(prop_item.name, None)
                if atoms_property is None:
                    raise KeyError(f"{prop_item.name} does not exist.")
            prop_vals.append(atoms_property)
        prop_vals = prop_item._convert_raw_(prop_vals)

        return prop_vals
    
    def _sparsify(self, prop_item: PropertyItem, prop_vals, prev_indices):
        """"""
        cur_indices = []
        if prop_item.sparsify == "filter":
            # -- select current property
            # TODO: possibly use np.where to replace this code
            if not prop_item.reverse:
                for i in prev_indices:
                    if prop_item.pmin <= prop_vals[i] < prop_item.pmax:
                        cur_indices.append(i)
            else:
                for i in prev_indices:
                    if prop_item.pmin > prop_vals[i] and prop_vals[i] >= prop_item.pmax:
                        cur_indices.append(i)
            scores = [prop_vals[i] for i in prev_indices]
        elif prop_item.sparsify == "sort":
            npoints = len(prev_indices)
            numbers = copy.deepcopy(prev_indices)
            sorted_numbers = sorted(numbers, key=lambda i: prop_vals[i])

            num_fixed = self._parse_selection_number(npoints)
            if not prop_item.reverse:
                cur_indices = sorted_numbers[:num_fixed]
            else:
                cur_indices = sorted_numbers[-num_fixed:]
            scores = [prop_vals[i] for i in prev_indices]
        elif prop_item.sparsify == "boltz":
            if not prop_item.trajwise:
                npoints = len(prev_indices)
                num_fixed = self._parse_selection_number(npoints)
                scores, cur_indices = boltz_selection(
                    prop_item.kBT, [prop_vals[i] for i in prev_indices], prev_indices, num_fixed, self.rng
                )
            else:
                # TODO: deal with trajectories
                raise NotImplementedError("Can't sparsify each trajectory separately.")
        else:
            # NOTE: check sparsifiction method in PropertyItem
            ...

        return scores, cur_indices


if __name__ == "__main__":
    ...