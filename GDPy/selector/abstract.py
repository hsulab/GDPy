#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# TODO: move this to the input file
DESC_JSON_PATH = "/mnt/scratch2/users/40247882/catsign/eann-main/soap_param.json"
DESC_JSON_PATH = "/mnt/scratch2/users/40247882/oxides/eann-main/soap_param.json"

from pathlib import Path

from dscribe.descriptors import SOAP

from GDPy.selector.structure_selection import cur_selection

class Selector():

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
    njobs = 4

    def __init__(
        self, 
        desc_dict,
        selec_dict,
        res_dir
    ):
        self.desc_dict = desc_dict
        self.selec_dict = selec_dict

        self.res_dir = Path(res_dir)

        return

    def calc_desc(self, frames):
        # calculate descriptor to select minimum dataset

        features_path = self.res_dir / "features.npy"
        # if features_path.exists():
        #    print("use precalculated features...")
        #    features = np.load(features_path)
        #    assert features.shape[0] == len(frames)
        # else:
        #    print('start calculating features...')
        #    features = calc_feature(frames, desc_dict, njobs, features_path)
        #    print('finished calculating features...')

        print("start calculating features...")
        features = self.calc_feature(frames, self.desc_dict, self.njobs, features_path)
        print("finished calculating features...")

        return features

    def calc_feature(self, frames, soap_parameters, njobs=1, saved_npy="features.npy"):
        """ Calculate feature vector for each configuration 
        """
        soap = SOAP(
            **soap_parameters
        )

        print(soap.get_number_of_features())

        # TODO: must use outer average? more flexible features 
        soap_instances = soap.create(frames, n_jobs=njobs)
        #for cur_soap, atoms in zip(soap_instances, frames):
        #    atoms.info['feature_vector'] = cur_soap

        # save calculated features 
        np.save(saved_npy, soap_instances)

        print('number of soap instances', len(soap_instances))

        return soap_instances


    def select_structures(
        self, features, num, index_map=None
    ):
        """ 
        """
        # cur decomposition
        cur_scores, selected = cur_selection(
            features, num, self.selec_dict["zeta"], self.selec_dict["strategy"]
        )

        # map selected indices
        if index_map is not None:
            selected = [index_map[s] for s in selected]
        # if manually_selected is not None:
        #    selected.extend(manually_selected)

        # TODO: if output
        # content = '# idx cur sel\n'
        # for idx, cur_score in enumerate(cur_scores):
        #     stat = 'F'
        #     if idx in selected:
        #         stat = 'T'
        #     if index_map is not None:
        #         idx = index_map[idx]
        #     content += '{:>12d}  {:>12.8f}  {:>2s}\n'.format(idx, cur_score, stat)
        # with open((prefix+"cur_scores.txt"), 'w') as writer:
        #    writer.write(content)
        #np.save((prefix+"indices.npy"), selected)

        #selected_frames = []
        # for idx, sidx in enumerate(selected):
        #    selected_frames.append(frames[int(sidx)])

        return selected


if __name__ == "__main__":
    pass