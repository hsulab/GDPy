#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import pathlib
from typing import Union, List

import numpy as np

import matplotlib as mpl
mpl.use("Agg") #silent mode
from matplotlib import pyplot as plt
try:
    plt.style.use("presentation")
except Exception as e:
    #print("Used default matplotlib style.")
    ...

import ase
from ase import Atoms
from ase.io import read, write

from .validator import AbstractValidator
from .utils import wrap_traj, smooth_curve
from ..data.array import AtomsNDArray

"""This module is used to analyse the particle distribution in the system.
"""


def plot_mass_distribution(wdir, prefix, bincentres, mass, title):
    """"""

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16,12))

    plt.suptitle("Mass Distribution Function")

    ax.set_xlabel("R [Å]")
    ax.set_ylabel("Mass [a.u./Å^2]")
    ax.set_title(title)

    avg_mass = np.average(mass, axis=0)
    bincentres_, avg_mass_ = smooth_curve(bincentres, avg_mass)
    ax.plot(bincentres_, avg_mass_, label="avg")

    std_mass = np.sqrt(np.var(mass, axis=0))
    bincentres_, lo_mass_ = smooth_curve(bincentres, avg_mass-std_mass)
    bincentres_, hi_mass_ = smooth_curve(bincentres, avg_mass+std_mass)
    ax.fill_between(bincentres_, lo_mass_, hi_mass_, alpha=0.2, label="${\sigma}$")

    # - stepwise
    #nframes = rdf.shape[0]
    #for i in range(0, nframes, step):
    #    cur_rdf = rdf[i]
    #    ax.plot(bincentres, cur_rdf, label=f"rdf{i}")

    ax.legend()

    plt.savefig(wdir/f"{prefix}mdf.png")

    data = np.vstack((bincentres_, avg_mass_, lo_mass_, hi_mass_)).T

    np.savetxt(
        wdir/f"{prefix}mdf.dat", data, 
        fmt=("%12.4f  "*4),
        header=("{:<8s}  "*4).format("r", "avg", "avg-sigma", "avg+sigma")
    )

    return


class MassDistributionValidator(AbstractValidator):

    """Validate mass distribution.

    TODO:
        Support molecular mass distribution by tags.

    """

    def __init__(self, symbols: List[str], drange, vector=[0,0,1], nbins=60, directory: Union[str, pathlib.Path] = "./", *args, **kwargs):
        """Init a mass distribution validator.

        Args:
            symbols:

        """
        super().__init__(directory, *args, **kwargs)

        self.symbols = symbols
        self.vector = np.array(vector) / np.linalg.norm(vector)
        self.nbins = nbins

        self.rmin, self.rmax = drange

        return
    
    def _process_data(self, data) -> List[List[Atoms]]:
        """"""
        data = AtomsNDArray(data)

        if data.ndim == 1:
            data = [data.tolist()]
        elif data.ndim == 2: # assume it is from extract_cache...
            data = data.tolist()
        elif data.ndim == 3: # assume it is from a compute node...
            data_ = []
            for d in data[:]: # TODO: add squeeze method?
                data_.extend(d)
            data = data_
        else:
            raise RuntimeError(f"Invalid shape {data.shape}.")

        return data
    
    def run(self, dataset, worker=None, *args, **kwargs):
        """Process reference and prediction data separately.

        TODO:

            Support average over several trajectories.

        """
        self._print("process reference ->")
        reference = dataset.get("reference", None)
        if reference is not None:
            self._irun(self._process_data(reference)[0], prefix="ref-")

        self._print("process prediction ->")
        prediction = dataset.get("prediction", None)
        if prediction is not None:
            self._irun(self._process_data(prediction)[0], prefix="pre-")

        return

    def _irun(self, frames: List[Atoms], prefix):
        """"""
        #pmin = np.floor(np.min(projected_distances))
        #pmax = np.ceil(np.max(projected_distances))
        bin_edges = np.linspace(self.rmin, self.rmax, self.nbins, endpoint=False).tolist()
        bin_edges.append(self.rmax)
        bin_edges = np.array(bin_edges)
        #print(len(bin_edges), bin_edges)
        bincentres = (bin_edges[:-1]+bin_edges[1:])/2.
        #print("bincentres: ", bincentres)

        # NOTE: only support structure with fixed cell
        atoms = frames[0]
        cell = atoms.get_cell(complete=True)
        surf_area = np.dot(np.cross(cell[0], cell[1]), self.vector)

        mass = []
        for atoms in frames:
            curr_mass = self._compute_mass_distribution(
                atoms, bin_edges, self.nbins
            )
            mass.append(curr_mass)
        #avg_mass = np.mean(mass, axis=0)
        #min_mass = np.min(mass, axis=0)
        #max_mass = np.max(mass, axis=0)

        mass = np.array(mass) / surf_area # normalised by surface area

        plot_mass_distribution(self.directory, prefix, bincentres, mass, ", ".join(self.symbols))

        return
    
    def _compute_mass_distribution(self, atoms: List[Atoms], bin_edges, nbins: int):
        """"""
        selected_indices = []
        for i, a in enumerate(atoms):
            if a.symbol in self.symbols:
                selected_indices.append(i)
        selected_positions = copy.deepcopy(atoms.get_positions()[selected_indices, :])
        projected_distances = [np.dot(pos, self.vector) for pos in selected_positions]

        masses = atoms.get_masses()
        selected_masses = [masses[i] for i in selected_indices]

        # - create bins
        bin_indices = np.digitize(projected_distances, bin_edges, right=False)

        groups = [[] for i in range(nbins)] # mass
        for i, i_bin in enumerate(bin_indices):
            # dump prop not in pmin and pmax
            if i_bin <= nbins:
                groups[i_bin-1].append(selected_masses[i])
        mass_by_digit = np.array([np.sum(x) for x in groups])

        #print("groups: ", groups)
        #print("mas: ", mass_by_digit)

        return mass_by_digit


if __name__ == "__main__":
    ...