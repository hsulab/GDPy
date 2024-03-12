#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import dataclasses
import itertools
from typing import NoReturn, Union, List, Callable

import numpy as np

from ase import Atoms
from ase.io import read, write

from ..data.array import AtomsNDArray
from .selector import AbstractSelector
from .cur import stat_str2val, boltz_selection, hist_selection


@dataclasses.dataclass
class PropertyItem:

    #: Property name that can be found in atoms.info or atoms.arrays.
    name: str

    #: metric config...
    metric: Union[str, List[str]] = None

    #: List of functions, min, max, average and ...
    _metric: List[Callable] = dataclasses.field(init=False, default_factory=list)

    # metric_params

    #: Sparsifiction method. [filter, sort, hist, boltz]
    sparsify: str = "filter"

    #: Whether reverse the sparsifiction behaviour.
    reverse: bool = False

    #: Property range to filter, which should be two strings or two numbers or mixed.
    range: List[float] = dataclasses.field(default_factory=lambda: [None, None])

    #: Property minimum.
    pmin: float = dataclasses.field(init=False, default=-np.inf)

    #: Property maximum.
    pmax: float = dataclasses.field(init=False, default=np.inf)

    #: Number of bins for histogram-based sparsification.
    nbins: int = 20

    #: Boltzmann temperature (eV).
    kBT: float = None

    # expression
    # weight

    # worker_config: dataclasses.InitVar[str] = None

    def __post_init__(self):
        """"""
        # - bound
        bounds_ = self.range
        if bounds_[0] is None:
            bounds_[0] = "min"
        if bounds_[1] is None:
            bounds_[1] = "max"

        # NOTE: assert range when property values are available
        # assert bounds_[0] < bounds_[1], f"{self.name} has invalid bounds..."
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
        assert self.sparsify in [
            "filter",
            "sort",
            "hist",
            "boltz",
        ], f"Unknown sparsification {self.sparsify}."

        return

    def _convert_raw_(self, raws_, weights_=None):
        """Convert raw values by the metric."""
        if len(self._metric) > 0:
            converts_ = []
            for raw_ in raws_:
                convert_ = raw_  # NOTE: copy?
                for metric_func in self._metric:
                    convert_ = metric_func(convert_)
                converts_.append(convert_)
        else:
            converts_ = raws_

        return converts_

    # def __repr__(self) -> str:
    #    """"""
    #    content = f"{self.name}:\n"
    #    content += f"  range: {self.pmin} - {self.pmax}\n"
    #    content += f"  sparsify: {self.sparsify}\n"

    #    return content


class PropertySelector(AbstractSelector):
    """Select structures based on structural properties.

    Each structure (trajectory) is represented by a float property.

    """

    name = "property"

    default_parameters = dict(
        mode="stru",
        properties=[],
        worker=None,  # compute properties on-the-fly
        number=[4, 0.2],
    )

    def __init__(self, directory="./", *args, **kwargs) -> None:
        """"""
        super().__init__(directory, *args, **kwargs)

        assert self.mode in ["stru", "traj"], f"Unknown selection mode {self.mode}."

        # - convert properties
        prop_items = []
        for name, params in self.properties.items():
            prop_item = PropertyItem(name=name, **params)
            prop_items.append(prop_item)
        self._prop_items = prop_items

        return

    def _mark_structures(self, data: AtomsNDArray, *args, **kwargs) -> None:
        """Return selected indices."""
        # - get property values
        #   NOTE: properties should be pre-computed...

        for prop_item in self._prop_items:
            self._print(str(prop_item))

            # - group markers
            if self.axis is None:
                marker_groups = dict(all=data.markers)
            else:
                marker_groups = {}
                for k, v in itertools.groupby(data.markers, key=lambda x: x[self.axis]):
                    if k in marker_groups:
                        marker_groups[k].extend(list(v))
                    else:
                        marker_groups[k] = list(v)
            self._debug(f"marker_groups: {marker_groups}")

            selected_markers = []
            for grp_name, curr_markers in marker_groups.items():
                curr_frames = data.get_marked_structures(curr_markers)
                curr_nframes = len(curr_frames)

                # --
                if curr_nframes > 0:
                    scores, selected_indices = self._sparsify(prop_item, curr_frames)
                    self._print(f"number of structures: {len(selected_indices)}")
                    curr_selected_markers = [curr_markers[i] for i in selected_indices]
                    selected_markers.extend(curr_selected_markers)

                    # - add score to atoms
                    for score, i in zip(scores, selected_indices):
                        curr_frames[i].info["score"] = score

                else:
                    ...

            data.markers = np.array(selected_markers)

            if len(selected_markers) == 0:
                break

        return

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
                atoms_property = energy / natoms
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

    def _statistics(self, prop_item: PropertyItem, prop_vals):
        """"""
        # - here are all data
        npoints = len(prop_vals)
        # pmax, pmin, pavg = np.max(prop_vals), np.min(prop_vals), np.mean(prop_vals)
        # pstd = np.sqrt(np.var(prop_vals-pavg))
        pmax = stat_str2val("max", prop_vals)
        pmin = stat_str2val("min", prop_vals)

        pavg = stat_str2val("avg", prop_vals)
        pstd = stat_str2val("svar", prop_vals)

        # NOTE: convert range to pmin and pmax
        #       hist data only in the range
        prop_item.pmin = stat_str2val(prop_item.pmin, prop_vals)
        prop_item.pmax = stat_str2val(prop_item.pmax, prop_vals)
        if prop_item.pmax < prop_item.pmin:
            prop_item.pmax = prop_item.pmin

        hist_max, hist_min = prop_item.pmax, prop_item.pmin
        # if hist_max == np.inf:
        #    hist_max = pmax
        # if hist_min == -np.inf:
        #    hist_min = pmin

        bins = np.linspace(hist_min, hist_max, prop_item.nbins, endpoint=False).tolist()
        bins.append(hist_max)
        hist, bin_edges = np.histogram(prop_vals, bins=bins, range=[hist_min, hist_max])

        # - output
        content = f"# Property {prop_item.name}\n"
        content += "# min {:<12.4f} max {:<12.4f}\n".format(pmin, pmax)
        content += "# avg {:<12.4f} std {:<12.4f}\n".format(pavg, pstd)
        content += "# histogram of {} points in the range (npoints: {})\n".format(
            np.sum(hist), npoints
        )
        content += f"# min {prop_item.pmin:<12.4f} max {prop_item.pmax:<12.4f}\n"
        for x, y in zip(hist, bin_edges[:-1]):
            content += "{:>12.4f}  {:>12d}\n".format(y, x)
        content += "{:>12.4f}  {:>12s}\n".format(bin_edges[-1], "-")

        with open(
            self.info_fpath.parent
            / (self.info_fpath.stem + f"-{prop_item.name}-stat.txt"),
            "w",
        ) as fopen:
            fopen.write(content)

        for l in content.split("\n"):
            self._print(l)

        return

    def _sparsify(self, prop_item: PropertyItem, frames: List[Atoms]):
        """"""
        # -- each structure is represented by one float value
        #    get per structure values
        prop_vals = self._extract_property(frames, prop_item)
        # -- give statistics of this property
        self._statistics(prop_item, prop_vals)

        nframes = len(frames)

        curr_indices, scores = [], []
        if prop_item.sparsify == "filter":
            # -- select current property
            # TODO: possibly use np.where to replace this code
            if not prop_item.reverse:
                for i in range(nframes):
                    if prop_item.pmin <= prop_vals[i] <= prop_item.pmax:
                        curr_indices.append(i)
            else:
                for i in range(nframes):
                    if prop_item.pmin > prop_vals[i] and prop_vals[i] > prop_item.pmax:
                        curr_indices.append(i)
            scores = [prop_vals[i] for i in curr_indices]
        elif prop_item.sparsify == "sort":
            numbers = list(range(nframes))
            sorted_numbers = sorted(numbers, key=lambda i: prop_vals[i])

            num_fixed = self._parse_selection_number(nframes)
            if not prop_item.reverse:
                curr_indices = sorted_numbers[:num_fixed]
            else:
                curr_indices = sorted_numbers[-num_fixed:]
            scores = [prop_vals[i] for i in curr_indices]
        elif prop_item.sparsify == "hist":
            num_fixed = self._parse_selection_number(nframes)
            prev_indices = list(range(nframes))
            scores, curr_indices = hist_selection(
                prop_item.nbins,
                prop_item.pmin,
                prop_item.pmax,
                [prop_vals[i] for i in prev_indices],
                prev_indices,
                num_fixed,
                self.rng,
            )
        elif prop_item.sparsify == "boltz":
            num_fixed = self._parse_selection_number(nframes)
            prev_indices = list(range(nframes))
            scores, curr_indices = boltz_selection(
                prop_item.kBT,
                [prop_vals[i] for i in prev_indices],
                prev_indices,
                num_fixed,
                self.rng,
            )
        else:
            # NOTE: check sparsifiction method in PropertyItem
            ...

        return scores, curr_indices


if __name__ == "__main__":
    ...
