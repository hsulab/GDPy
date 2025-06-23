#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import copy
import dataclasses
from typing import Optional, Union

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from gdpx.data.array import AtomsNDArray
from gdpx.describer import REGISTER as DESCRIBER_REGISTER

from .clustering import group_structures_by_axis
from .selector import BaseSelector
from .sparsification import IMPLEMENTED_SPARSIFY_METHODS, ScalarSparsification
from .utils import stat_str2val

IMPLEMENTED_SCALAR_PROPERTIES: list[str] = [
    "atomic_energy",
    "energy",
    "forces",
    "volume",
    "min_distance",
    "max_devi_f",
    # from describer
    "max_frc_err",
    "abs_ene_err",
]
IMPLEMENTED_STRING_PROPERTIES: list[str] = [
    "chemical_formula",
]
IMPLEMENTED_PROPERTIES: list[str] = IMPLEMENTED_SCALAR_PROPERTIES + IMPLEMENTED_STRING_PROPERTIES


def get_metric_func(metric_name: str):
    """"""
    if metric_name == "fabs":
        metric_func = np.fabs
    elif metric_name == "max":
        metric_func = np.max
    elif metric_name == "min":
        metric_func = np.min
    else:
        raise NotImplementedError(f"Unknown metric function {metric_name}.")

    return metric_func


def compute_minimum_distance(atoms: Atoms, cutoff: float):
    """"""
    i, j, d = neighbor_list("ijd", atoms, cutoff=cutoff)

    # pair specific?

    return np.min(d)


def get_aligned_chemical_formula(atoms: Atoms, symbol_list: list[str], padding_width: int = 4):
    """"""
    chemical_symbols = atoms.get_chemical_symbols()
    counter = collections.Counter(chemical_symbols)

    chemical_formula = ""
    for s in symbol_list:
        n = counter.get(s, 0)
        chemical_formula += f"{s}{n:>0{padding_width}d}"

    return chemical_formula


@dataclasses.dataclass
class PropertyItem:

    #: Property name that can be found in atoms.info or atoms.arrays.
    name: str

    #: Parameters for initialising a describer.
    params: dict = dataclasses.field(default_factory=dict)

    #: The metric functions applied to the property values.
    metric: Optional[Union[str, list[str]]] = None

    #: Group-based selection by a representative structure's property.
    represent_by: Optional[str] = None

    #: Sparsifiction method.
    sparsify: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        """"""
        # Map meteric functions
        self._metric_functions = []
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
                self._metric_functions.append(metric_func)
        else:
            ...

        # Check sparsification method
        sparsify_params = copy.deepcopy(self.sparsify)
        sparsify_method = sparsify_params.pop("method", "filter")
        if sparsify_method not in IMPLEMENTED_SPARSIFY_METHODS:
            raise NotImplementedError(f"Unknown sparsification method {sparsify_method}.")
        else:
            self._sparsify = IMPLEMENTED_SPARSIFY_METHODS[sparsify_method](**sparsify_params)

        return

    def _convert_raw_(self, raws_, weights_=None):
        """Convert raw values by the metric."""
        if len(self._metric_functions) > 0:
            converts_ = []
            for raw_ in raws_:
                convert_ = raw_
                for metric_func in self._metric_functions:
                    convert_ = metric_func(convert_)
                converts_.append(convert_)
        else:
            converts_ = raws_

        return converts_


class PropertySelector(BaseSelector):
    """Select structures based on structural properties.

    Each structure (trajectory) is represented by a float property.

    """

    name = "property"

    default_parameters = dict(
        name="property",
        params={},
        metric=None,
        represent_by=None,
        sparsify={},
        number=[4, 0.2],
    )

    def __init__(self, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        # Convert input paramters into one property
        prop_params = copy.deepcopy(self.parameters)
        prop_params.pop("number")
        self._property = PropertyItem(**prop_params)

        return

    def _mark_structures(self, data: AtomsNDArray) -> None:
        """Select structures based on pre-computed property."""

        self._print(f"property -> {self._property.name}")

        # Group markers by certain criteria (axis for now)
        marker_groups = group_structures_by_axis(data, self.group_by)
        self._debug(f"marker_groups: {marker_groups}")

        num_groups = len(marker_groups)
        self._print(f"number of groups: {num_groups}")

        if num_groups > 1:
            if self._property.represent_by is None:
                selected_markers = self._mark_group_separate(data, self._property, marker_groups)
            else:
                self._print(
                    "Group-based selection is enabled "
                    + f"using representative structure by {self._property.represent_by}."
                )
                selected_markers = self._mark_group_represent(data, self._property, marker_groups)
        else:
            selected_markers = self._mark_group_separate(data, self._property, marker_groups)

        data.markers = np.array(selected_markers)

        return

    def _mark_group_represent(self, data, prop_item: PropertyItem, marker_groups):
        """Mark a group of structures based on a representative structure's property."""

        assert prop_item.represent_by is not None, "No representative method is provided."
        metric_func = get_metric_func(prop_item.represent_by)

        rep_groups = []  # data for representative groups
        for grp_name, curr_markers in marker_groups.items():
            curr_frames = data.get_marked_structures(curr_markers)
            curr_nframes = len(curr_frames)

            assert curr_nframes > 0, f"No structures is found in group {grp_name}."

            curr_values = self._extract_property(curr_frames, prop_item)
            metric_val = metric_func(curr_values)

            # For some groups, there might be multiple frames with the same metric value,
            # for instance, the minimisation trajectories.
            for i, val in enumerate(curr_values):
                if np.isclose(val, metric_val):
                    rep_frame = curr_frames[i]
                    break
            else:
                rep_frame = None
            assert rep_frame is not None, f"Cannot find representative frame with metric value {metric_val}."
            rep_groups.append((grp_name, rep_frame))

        rep_frames = [x[1] for x in rep_groups]

        selected_markers = []
        scores, selected_indices = self._sparsify(prop_item, rep_frames)
        self._print(f"number of groups selected: {len(selected_indices)}")

        _counter = 0
        for s_i in selected_indices:
            grp_name = rep_groups[s_i][0]
            curr_selected_markers = marker_groups[grp_name]
            selected_markers.extend(curr_selected_markers)
            curr_score = scores[selected_indices.index(s_i)]
            curr_selected_frames = data.get_marked_structures(curr_selected_markers)
            for a in curr_selected_frames:
                a.info["score"] = curr_score
            num_curr_frames = len(curr_selected_frames)
            _counter += num_curr_frames

        assert _counter == len(selected_markers)

        return selected_markers

    def _mark_group_separate(self, data, prop_item: PropertyItem, marker_groups):
        """Mark a group of structures based on a structure's own property."""
        selected_markers = []
        for grp_name, curr_markers in marker_groups.items():
            curr_frames = data.get_marked_structures(curr_markers)
            curr_nframes = len(curr_frames)

            if curr_nframes > 0:
                scores, selected_indices = self._sparsify(prop_item, curr_frames)
                self._print(f"group: {grp_name} -> number of structures: {len(selected_indices)}")
                curr_selected_markers = [curr_markers[i] for i in selected_indices]
                selected_markers.extend(curr_selected_markers)

                # Add score to atoms
                for score, i in zip(scores, selected_indices):
                    curr_frames[i].info["score"] = score

            else:
                ...

        return selected_markers

    def _extract_property(self, frames: list[Atoms], prop_item: PropertyItem):
        """Extract property values from frames.

        Returns:
            property values: list[float] or 1d-np.array.

        """
        if prop_item.name in IMPLEMENTED_PROPERTIES:
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
                elif prop_item.name == "volume":
                    atoms_property = atoms.get_volume()
                elif prop_item.name == "chemical_formula":
                    atoms_property = get_aligned_chemical_formula(
                        atoms,
                        prop_item.params["type_list"],
                        prop_item.params["padding_width"],
                    )
                elif prop_item.name == "min_distance":
                    # TODO: Move to observables?
                    atoms_property = compute_minimum_distance(atoms, prop_item.params["cutoff"])
                else:
                    # -- any property stored in atoms.info
                    #    e.g. max_devi_f
                    atoms_property = atoms.info.get(prop_item.name, None)
                    if atoms_property is None:
                        atoms_property = atoms.arrays.get(prop_item.name, None)
                    if atoms_property is None:
                        raise KeyError(f"{prop_item.name} does not exist.")
                prop_vals.append(atoms_property)
        else:
            # Try use describer to get properties, and make sure the property
            # values are a list as some sparsify needs a list (hist, boltz).
            desc_name = prop_item.name
            if desc_name not in DESCRIBER_REGISTER:
                raise KeyError(f"Unknown describer {desc_name} in {DESCRIBER_REGISTER.keys()}.")
            describer = DESCRIBER_REGISTER[desc_name](**prop_item.params)
            prop_vals = describer.run(frames).tolist()

        prop_vals = prop_item._convert_raw_(prop_vals)

        return prop_vals

    def _statistics(self, prop_name, prop_vals, sparsify: ScalarSparsification):
        """Show statistics of the property and update the lower and upper limites of the sparsification."""
        # Get basic statistics for property values
        pmax = stat_str2val("max", prop_vals)
        pmin = stat_str2val("min", prop_vals)

        pavg = stat_str2val("avg", prop_vals)
        pstd = stat_str2val("std", prop_vals)

        # Update sparsification's pmin and pmax by a custom range
        s_pmin, s_pmax = sparsify._pmin, sparsify._pmax
        assert s_pmin is not None and s_pmax is not None
        s_pmin = stat_str2val(s_pmin, prop_vals)
        s_pmax = stat_str2val(s_pmax, prop_vals)
        if s_pmax < s_pmin:
            s_pmax = s_pmin
        sparsify._pmin, sparsify._pmax = s_pmin, s_pmax

        nbins = sparsify.nbins
        hist_max, hist_min = s_pmax, s_pmin

        bins = np.linspace(hist_min, hist_max, nbins, endpoint=False).tolist()
        bins.append(hist_max)
        hist, bin_edges = np.histogram(prop_vals, bins=bins, range=(hist_min, hist_max))

        # Output histogram
        content = f"# Property {prop_name}\n"
        content += f"# min {pmin:<12.4f} max {pmax:<12.4f}\n"
        content += f"# avg {pavg:<12.4f} std {pstd:<12.4f}\n"
        content += f"# histogram of {np.sum(hist)} points in the range (npoints: {len(prop_vals)})\n"
        content += f"# min {s_pmin:<12.4f} max {s_pmax:<12.4f}\n"
        for x, y in zip(hist, bin_edges[:-1]):
            content += f"{y:>12.4f}  {x:>12d}\n"
        content += f"{bin_edges[-1]:>12.4f}  {'-':>12s}\n"

        with open(
            self.info_fpath.parent / (self.info_fpath.stem + f"-{prop_name}-stat.txt"),
            "w",
        ) as fopen:
            fopen.write(content)

        for l in content.split("\n"):
            self._print(l)

        return

    def _sparsify(self, prop_item: PropertyItem, frames: list[Atoms]):
        """"""
        # Each structure is represented by one float/string value
        prop_vals = self._extract_property(frames, prop_item)

        # Show statistics of this property
        if prop_item.name in IMPLEMENTED_SCALAR_PROPERTIES and isinstance(prop_item._sparsify, ScalarSparsification):
            prop_type = "scalar"
            self._statistics(prop_item.name, prop_vals, prop_item._sparsify)
        elif prop_item.name in IMPLEMENTED_STRING_PROPERTIES:
            prop_type = "string"
            unique_types = sorted(list(set(prop_vals)))
            counter = collections.Counter(prop_vals)
            for unique_name in unique_types:
                self._print(f"  {unique_name} -> {counter[unique_name]}")
        else:
            # These properties may be from describers.
            prop_type = "scalar"  # TODO: More property types?
            if prop_type == "scalar":
                self._statistics(prop_item.name, prop_vals, prop_item._sparsify)
            self._print(f"{prop_item.name} does not support statistics.")

        if prop_type is None:
            raise Exception(f"Unknown property type {prop_item.name}.")

        # Run sparsification
        sparsify = prop_item._sparsify

        sparsify_params = sparsify.get_sparsify_params()
        extra_params = dict(
            prop_type=prop_type,
            props=prop_vals,
            num_selected=self._parse_selection_number(len(frames)),
            rng=self.rng,
        )
        for k, v in extra_params.items():
            if k in sparsify_params:
                sparsify_params[k] = v  # type: ignore
        scores, selected_indices = sparsify.run(**sparsify_params)

        return scores, selected_indices


if __name__ == "__main__":
    ...
