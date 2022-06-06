#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, List

import numpy as np

from pathlib import Path

from ase import Atoms

from dscribe.descriptors import SOAP
from GDPy.selector.cur import cur_selection

from GDPy import config

""" Various Selection Protocols
"""

class DeviationSelector():

    # TODO: should be arbitrary property deviation
    # not only energy and force

    name = "deviation"
    selection_criteria = "deviation"

    deviation_criteria = dict(
        # energy tolerance would be natoms*atomic_energy
        atomic_energy = [None, None],
        # maximum fractional force deviation in the system
        force = [None, None] 
    )

    def __init__(
        self,
        properties: dict,
        criteria: dict,
        potential = None # file
    ):
        self.deviation_criteria = criteria
        self.__parse_criteria()

        self.__register_potential(potential)

        # - parse properties
        # TODO: select on properties not only on fixed name (energy, forces)
        # if not set properly, will try to call calculator
        self.energy_tag = properties["atomic_energy"]
        self.force_tag = properties["force"]

        return
    
    def __parse_criteria(self):
        """"""
        use_ae, use_force = True, True

        emin, emax = self.deviation_criteria["atomic_energy"]
        fmin, fmax = self.deviation_criteria["force"]

        if emin is None and emax is None:
            use_ae = False
        if emin is None:
            self.deviation_criteria["atomic_energy"][0] = -np.inf
        if emax is None:
            self.deviation_criteria["atomic_energy"][1] = np.inf

        if fmin is None and fmax is None:
            use_force = False
        if fmin is None:
            self.deviation_criteria["force"][0] = -np.inf
        if fmax is None:
            self.deviation_criteria["force"][1] = np.inf

        emin, emax = self.deviation_criteria["atomic_energy"]
        fmin, fmax = self.deviation_criteria["force"]

        if (emin > emax):
            raise RuntimeError("emax should be larger than emin...")
        if (fmin > fmax):
            raise RuntimeError("fmax should be larger than fmin...")
        
        if not (use_ae or use_force):
            raise RuntimeError("neither energy nor force criteria is set...")
        
        self.use_ae = use_ae
        self.use_force = use_force

        return
    
    def calculate(self, frames):
        """"""
        # TODO: move this part to potential manager?
        if self.calc is None:
            raise RuntimeError("calculator is not set properly...")
        
        energies, maxforces = [], []
        energy_deviations, force_deviations = [], []
        
        for atoms in frames:
            self.calc.reset()
            self.calc.calc_uncertainty = True # TODO: this is not a universal interface
            atoms.calc = self.calc
            # obtain results
            energy = atoms.get_potential_energy()
            fmax = np.max(np.fabs(atoms.get_forces()))
            enstdvar = atoms.calc.results["en_stdvar"] / len(atoms)
            maxfstdvar = np.max(atoms.calc.results["force_stdvar"])
            # add results
            energies.append(energy)
            maxforces.append(fmax)
            energy_deviations.append(enstdvar)
            force_deviations.append(maxfstdvar)

        return (energies, maxforces, energy_deviations, force_deviations)
    
    def select(self, frames, index_map=None) -> List[Atoms]:
        """"""
        energy_deviations = [a.info[self.energy_tag] for a in frames]
        force_deviations = [a.info[self.force_tag] for a in frames]
        selected_indices = self._select_indices(energy_deviations, force_deviations)

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        selected_frames = [frames[i] for i in selected_indices]

        return selected_frames
    
    def _select_indices(self, energy_deviations, force_deviations = None) -> List[int]:
        """
        """
        if force_deviations is not None:
            assert len(energy_deviations) == len(force_deviations), "shapes of energy and force deviations are inconsistent"
        else:
            force_deviations = np.empty(len(energy_deviations))
            force_deviations[:] = np.NaN

        emin, emax = self.deviation_criteria["atomic_energy"]
        fmin, fmax = self.deviation_criteria["force"]
        
        # NOTE: deterministic selection
        selected = []
        for idx, (en_devi, force_devi) in enumerate(zip(energy_deviations, force_deviations)):
            if self.use_ae:
                if emin < en_devi < emax:
                    selected.append(idx)
                    continue
            if self.use_force:
                if fmin < force_devi < fmax:
                    selected.append(idx)
                    continue

        return selected
    
    def __register_potential(self, potential=None):
        """"""
        # load potential
        from GDPy.potential.manager import create_manager
        if potential is not None:
            atypes = None
            pm = create_manager(potential)
            if not pm.uncertainty:
                raise RuntimeError(
                    "Potential should be able to predict deviation if DeviationSelector is used..."
                )
            calc = pm.generate_calculator(atypes)
            print("MODELS: ", pm.models)
        else:
            calc = None

        self.calc = calc

        return

class DescriptorBasedSelector():

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

    njobs = 1

    verbose = False

    def __init__(
        self, 
        descriptor,
        criteria,
        directory = Path.cwd()
    ):
        self.desc_dict = descriptor
        self.selec_dict = criteria

        self.res_dir = Path(directory)

        self.njobs = config.NJOBS

        print("selector uses njobs ", self.njobs)

        return

    def calc_desc(self, frames):
        """"""
        # calculate descriptor to select minimum dataset

        features_path = self.res_dir / "features.npy"
        # TODO: read cached features
        # if features_path.exists():
        #    print("use precalculated features...")
        #    features = np.load(features_path)
        #    assert features.shape[0] == len(frames)
        # else:
        #    print('start calculating features...')
        #    features = calc_feature(frames, desc_dict, njobs, features_path)
        #    print('finished calculating features...')

        print("start calculating features...")
        desc_params = self.desc_dict.copy()
        desc_name = desc_params.pop("name", None)

        features = None
        if desc_name == "soap":
            soap = SOAP(**desc_params)
            print("descriptor dimension: ", soap.get_number_of_features())
            features = soap.create(frames, n_jobs=self.njobs)
        else:
            raise RuntimeError(f"Unknown descriptor {desc_name}.")
        print("finished calculating features...")

        # save calculated features 
        if self.verbose:
            np.save(features_path, features)
            print('number of soap instances', len(features))

        return features

    def select(self, frames, index_map=None) -> List[Atoms]:
        """"""
        features = self.calc_desc(frames)

        selected_indices = self._select_indices(features)
        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        # if manually_selected is not None:
        #    selected.extend(manually_selected)
        selected_frames = [frames[i] for i in selected_indices]

        return selected_frames

    def _select_indices(self, features):
        """ 
        """
        # cur decomposition
        cur_scores, selected = cur_selection(
            features, self.selec_dict["number"],
             self.selec_dict["zeta"], self.selec_dict["strategy"]
        )

        # TODO: if output
        if self.verbose:
            content = '# idx cur sel\n'
            for idx, cur_score in enumerate(cur_scores):
                stat = "F"
                if idx in selected:
                    stat = "T"
                if index_map is not None:
                    idx = index_map[idx]
                content += "{:>12d}  {:>12.8f}  {:>2s}\n".format(idx, cur_score, stat)
            with open((self.res_dir / "cur_scores.txt"), "w") as writer:
               writer.write(content)
        #np.save((prefix+"indices.npy"), selected)

        #selected_frames = []
        # for idx, sidx in enumerate(selected):
        #    selected_frames.append(frames[int(sidx)])

        return selected


def create_selector(input_list: list):
    selectors = []
    for s in input_list:
        params = s.copy()
        method = params.pop("method", None)
        if method == "deviation":
            selectors.append(DeviationSelector(**params))
        elif method == "descriptor":
            selectors.append(DescriptorBasedSelector(**params))
        else:
            raise RuntimeError(f"Cant find selector with method {method}.")

    return selectors


if __name__ == "__main__":
    pass