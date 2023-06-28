#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import abc
import copy
import pathlib
from typing import Optional, List, Mapping

import numpy as np
from sklearn.decomposition import PCA

from ase import Atoms
from ase.io import read, write

from GDPy.core.node import AbstractNode

try:
    from dscribe.descriptors import SOAP
except Exception as e:
    print(e)

USE_CHEMISCOPE = 0
try:
    import chemiscope
    USE_CHEMISCOPE = 1
except Exception as e:
    print(e)


class AbstractDescriber(AbstractNode):

    cache_features = "features.npy"

    descriptor = dict(
        name = "soap",
        species = ["Al", "Cu", "O"],
        #r_cut = 6.0,
        #n_max = 12,
        #l_max = 8,
        rcut = 6.0,
        nmax = 12,
        lmax = 8,
        sigma = 0.2,
        average = "inner",
        periodic = True,
    )

    def __init__(self, *args, **kwargs) -> None:
        """"""
        super().__init__(*args, **kwargs)

        return

    @abc.abstractmethod
    def run(self, dataset, *args, **kwargs):
        """"""
        ...
        #wdir = pathlib.Path(
        #    #"/scratch/gpfs/jx1279/copper+alumina/dataset/s001/Cu13+s001p32-Al96Cu13O144-surf"
        #    "./Cu13+s001p32-Al96Cu13O144-surf"
        #)
        #xyzpaths = sorted(list(wdir.glob("*.xyz")))
        ##print(xyzpaths)

        ## - read structures and get groups
        #groups, frames = {}, []
        #for p in xyzpaths:
        #    print(p.name)
        #    start = len(frames)
        #    curr_frames = read(p, ":")
        #    curr_nframes = len(curr_frames)
        #    groups[p.name] = list(range(start,start+curr_nframes))
        #    frames.extend(curr_frames)
        ##print(groups, groups)
        #print(f"nframes: {len(frames)}")
        self._debug(f"n_jobs: {self.njobs}")
        print(dataset)

        # - for single system
        frames = dataset[0]._images
        if not (self.directory/self.cache_features).exists():
            features = self._compute_descripter(frames=frames)
            np.save(self.directory/self.cache_features, features)
        else:
            features = np.load(self.directory/self.cache_features)
        
        group_indices = {}
        for k, v in self.groups.items():
            group_indices[k] = dataset[0].get_matched_indices(v)
        
        self._plot_results(features=features, groups=group_indices)

        return

    def _write_chemiscope(self, frames, features):
        """"""
        # - write chemiscope inputs
        pca = PCA(n_components=2).fit_transform(features)
        properties = dict(
            PCA = dict(
                target = "structure",
                values = pca
            ),
            #energies = dict(
            #    target = "structure",
            #    values = [a.get_potential_energy() for a in frames],
            #    units = "eV"
            #)
        )

        frame_properties = chemiscope.extract_properties(
            frames,
            only=["energy"]
        )
        properties.update(**frame_properties)

        chemiscope.write_input(
            self.directory/"my-input.json.gz", frames=frames, properties=properties
        )

        return

    def _plot_results(self, features, groups: Mapping[str,List[int]], *args, **kwargs):
        """"""
        # - plot selection
        import matplotlib.pyplot as plt
        try:
            plt.style.use("presentation")
        except Exception as e:
            ...

        reducer = PCA(n_components=2)
        reducer.fit(features)

        for i, (name, indices) in enumerate(groups.items()):
            proj = reducer.transform(features[indices,:])
            plt.scatter(
                proj[:, 0], proj[:, 1], alpha=0.5, zorder=100-i,
                label=f"{name} {len(indices)}"
            )

        plt.legend()
        plt.axis("off")
        plt.savefig(self.directory/"pca.png")

        return 


if __name__ == "__main__":
    ...