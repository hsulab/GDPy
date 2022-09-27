#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

    name = "descriptor"
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
        selection_ratio = 0.2,
        selection_number = 16
    )

    njobs = 1

    verbose = False

    def __init__(
        self, 
        descriptor,
        criteria,
        directory = Path.cwd(),
        *args, **kwargs
    ):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        self.desc_dict = descriptor
        self.selec_dict = criteria

        self.pfunc("selector uses njobs ", self.njobs)

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
        desc_params = self.desc_dict.copy()
        desc_name = desc_params.pop("name", None)

        features = None
        if desc_name == "soap":
            soap = SOAP(**desc_params)
            ndim = soap.get_number_of_features()
            self.pfunc(f"descriptor dimension: {ndim}")
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

    def select(self, frames, index_map=None, ret_indices: bool=False, *args, **kwargs) -> List[Atoms]:
        """"""
        super().select(*args, **kwargs)
        if len(frames) == 0:
            return []

        features = self.calc_desc(frames)

        selected_indices = self._select_indices(features)
        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        # if manually_selected is not None:
        #    selected.extend(manually_selected)

        self.pfunc(f"nframes {len(frames)} -> nselected {len(selected_indices)}")

        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            if True: # TODO: check if output data
                write(self.directory/("-".join([self.prefix,self.name,"selection"])+".xyz"), selected_frames)

            if True: # TODO: check if output data
                np.save(self.directory/("-".join([self.prefix,self.name,"indices"])+".npy"), selected_indices)

            return selected_frames
        else:
            return selected_indices

    def _select_indices(self, features):
        """ number can be in any forms below
            [num_fixed, num_percent]
        """
        nframes = features.shape[0]
        number = self._parse_selection_number(nframes)

        # cur decomposition
        if nframes == 1:
            selected = [0]
        else:
            cur_scores, selected = cur_selection(
                features, number,
                self.selec_dict["zeta"], self.selec_dict["strategy"]
            )

        # TODO: if output
        if self.verbose:
            content = '# idx cur sel\n'
            for idx, cur_score in enumerate(cur_scores):
                stat = "F"
                if idx in selected:
                    stat = "T"
                content += "{:>12d}  {:>12.8f}  {:>2s}\n".format(idx, cur_score, stat)
            with open((self.directory / "cur_scores.txt"), "w") as writer:
               writer.write(content)
        #np.save((prefix+"indices.npy"), selected)

        #selected_frames = []
        # for idx, sidx in enumerate(selected):
        #    selected_frames.append(frames[int(sidx)])

        return selected


if __name__ == "__main__":
    pass