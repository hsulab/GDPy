#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write

from dscribe.descriptors import SOAP

from GDPy.selector.selector import AbstractSelector
from GDPy.selector.cur import cur_selection


""" References
    [1] Bernstein, N.; Csányi, G.; Deringer, V. L. 
        De Novo Exploration and Self-Guided Learning of Potential-Energy Surfaces. 
        npj Comput. Mater. 2019, 5, 99.
    [2] Mahoney, M. W.; Drineas, P. 
        CUR Matrix Decompositions for Improved Data Analysis. 
        Proc. Natl. Acad. Sci. USA 2009, 106, 697–702.
"""


class DescriptorBasedSelector(AbstractSelector):

    name = "dscribe"
    selection_criteria = "geometry"

    """
    {
    "soap":
        {
            "species" : ["O", "Pt"],
            "rcut" : 6.0,
            "nmax" : 12,
            "lmax" : 8,
            "sigma" : 0.3,
            "average" : "inner",
            "periodic" : true
        },
    "selection":
        {
            "zeta": -1,
            "strategy": "descent"
        }
    }
    """

    default_parameters = dict(
        random_seed = None,
        descriptor = None,
        criteria = dict(
            method = "cur",
            zeta = "-1",
            strategy = "descent"
        ),
        number = [4, 0.2]
    )

    verbose = False

    def __init__(
        self, 
        directory = Path.cwd(),
        *args, **kwargs
    ):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        return

    def calc_desc(self, frames):
        """"""
        # calculate descriptor to select minimum dataset

        features_path = self.directory / "features.npy"
        # TODO: read cached features
        # if features_path.exists():
        #    print("use precalculated features...")
        #    features = np.load(features_path)
        #    assert features.shape[0] == len(frames)
        # else:
        #    print('start calculating features...')
        #    features = calc_feature(frames, desc_dict, njobs, features_path)
        #    print('finished calculating features...')

        self.pfunc("start calculating features...")
        desc_params = copy.deepcopy(self.descriptor)
        desc_name = desc_params.pop("name", None)

        features = None
        if desc_name == "soap":
            soap = SOAP(**desc_params)
            ndim = soap.get_number_of_features()
            self.pfunc(f"soap descriptor dimension: {ndim}")
            features = soap.create(frames, n_jobs=self.njobs)
        else:
            raise RuntimeError(f"Unknown descriptor {desc_name}.")
        self.pfunc("finished calculating features...")

        # save calculated features 
        features = features.reshape(-1,ndim)
        if self.verbose:
            np.save(features_path, features)
            self.pfunc(f"number of soap instances {len(features)}")

        return features

    def _select_indices(self, frames, *args, **kwargs):
        """ number can be in any forms below
            [num_fixed, num_percent]
        """
        nframes = len(frames)
        num_fixed = self._parse_selection_number(nframes)

        # NOTE: currently, only CUR is supported
        # TODO: farthest sampling, clustering ...
        if num_fixed > 0:
            features = self.calc_desc(frames)

            # cur decomposition
            if nframes == 1:
                selected_indices = [0]
            else:
                cur_scores, selected_indices = cur_selection(
                    features, num_fixed,
                    self.criteria["zeta"], self.criteria["strategy"],
                    rng = self.rng
                )
        else:
            cur_scores, selected_indices = [], []

        # - output
        data = []
        for i, s in enumerate(selected_indices):
            atoms = frames[s]
            # - gather info
            confid = atoms.info.get("confid", -1)
            natoms = len(atoms)
            try:
                ae = atoms.get_potential_energy() / natoms
            except:
                ae = np.NaN
            try:
                maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            except:
                maxforce = np.NaN
            score = cur_scores[i]
            data.append([s, confid, natoms, ae, maxforce, score])
        if data:
            np.savetxt(
                self.info_fpath, data, 
                fmt="%8d  %8d  %8d  %12.4f  %12.4f  %12.4f",
                #fmt="{:>8d}  {:>8d}  {:>8d}  {:>12.4f}  {:>12.4f}",
                header="{:>6s}  {:>8s}  {:>8s}  {:>12s}  {:>12s}  {:>12s}".format(
                    *"index confid natoms AtomicEnergy MaxForce  CurScore".split()
                ),
                footer=f"random_seed {self.random_seed}"
            )

        return selected_indices


if __name__ == "__main__":
    pass