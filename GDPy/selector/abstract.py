#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Union, List

import numpy as np

import abc 
from pathlib import Path

from ase import Atoms
from ase.io import read, write

from dscribe.descriptors import SOAP
from GDPy.selector.cur import cur_selection

from GDPy import config

""" Various Selection Protocols
"""

# TODO: create a composed selector

class AbstractSelector(abc.ABC):

    default_parameters = dict(
        selection_ratio = 0.2,
        selection_number = 320
    )

    prefix = "structure"

    @abc.abstractmethod
    def select(self, *args, **kargs):

        return

    def _parse_selection_number(self, nframes):
        """"""
        ratio = self.default_parameters["selection_ratio"]
        number = self.default_parameters["selection_number"]

        number_info = self.selec_dict.get("number", [None,ratio])
        if isinstance(number_info, int):
            number_info = [number_info, ratio]
        elif isinstance(number_info, float):
            number_info = [number, number_info]
        else:
            assert len(number_info) == 2, "Cant parse number for selection..."
        
        num_fixed, num_percent = number_info
        if num_fixed is not None:
            if num_fixed > nframes:
                num_fixed = int(nframes*num_percent)
        else:
            num_fixed = int(nframes*num_percent)

        return num_fixed

class ComposedSelector(AbstractSelector):
    
    """ perform a list of selections on input frames
    """

    name = None
    verbose = True

    def __init__(self, selectors, directory=Path.cwd()):
        """"""
        self.selectors = selectors
        self._directory = directory

        self._check_convergence()

        # - set namd and directory
        self.name = "-".join([s.name for s in self.selectors])

        return
    
    def _check_convergence(self):
        """ check if there is a convergence selector
            if so, selections will be performed on converged ones and 
            others separately
        """
        conv_i = None
        selectors_ = self.selectors
        for i, s in enumerate(selectors_):
            if s.name == "convergence":
                conv_i = i
                break
        
        if conv_i is not None:
            self.conv_selection = selectors_.pop(conv_i)
            self.selectors = selectors_
        else:
            self.conv_selection = None

        return
    
    @property
    def directory(self):
        return self._directory
    
    @directory.setter
    def directory(self, directory_: Union[str,Path]):
        """"""
        self._directory = directory_

        for s in self.selectors:
            s.directory = self._directory

        return
    
    def select(self, frames, index_map=None, ret_indces: bool=False):
        """"""
        # - initial index stuff
        nframes = len(frames)
        cur_index_map = list(range(nframes))

        # NOTE: find converged ones and others
        # convergence selector is standalone
        frame_index_groups = {}
        if self.conv_selection is not None:
            selector = self.conv_selection
            converged_indices = selector.select(frames, ret_indices=True)
            traj_indices = [m for m in cur_index_map if m not in converged_indices] # means not converged
            frame_index_groups = dict(
                converged = converged_indices,
                traj = traj_indices
            )
            # -- converged
            converged_frames = [frames[x] for x in converged_indices]
            print("nconverged: ", len(converged_frames))
            if converged_frames:
                write(self.directory/f"{selector.name}-frames.xyz", converged_frames)
        else:
            frame_index_groups = dict(
                traj = cur_index_map
            )

        # - run selectors
        merged_final_indices = []

        for sys_name, global_indices in frame_index_groups.items():
            # - init 
            metadata = [] # selected indices
            final_indices = []

            # - prepare index map
            cur_frames = [frames[i] for i in global_indices]
            cur_index_map = global_indices.copy()

            metadata.append(cur_index_map)

            print(f"----- Start System {sys_name} -----")
            print("ncandidates: ", len(cur_frames))
            for isele, selector in enumerate(self.selectors):
                # TODO: add info to selected frames
                # TODO: use cached files
                print(f"--- Selection {isele} Method {selector.name}---")
                print("ncandidates: ", len(cur_frames))
                cur_indices = selector.select(cur_frames, ret_indices=True)
                mapped_indices = sorted([cur_index_map[x] for x in cur_indices])
                metadata.append(mapped_indices)
                final_indices = mapped_indices.copy()
                cur_frames = [frames[x] for x in mapped_indices]
                print("nselected: ", len(cur_frames))
                # - create index_map for next use
                cur_index_map = mapped_indices.copy()

                # TODO: should print-out intermediate results?
                write(self.directory/f"{sys_name}-{selector.name}-selection-{isele}.xyz", cur_frames)
            
            merged_final_indices += final_indices

            # TODO: map selected indices
            # is is ok?
            if index_map is not None:
                final_indices = [index_map[s] for s in final_indices]

            # - ouput data
            # TODO: write selector parameters to metadata file
            maxlength = np.max([len(m) for m in metadata])
            data = -np.ones((maxlength,len(metadata)))
            for i, m in enumerate(metadata):
                data[:len(m),i] = m
            header = "".join(["init "+"{:<24s}".format(s.name) for s in self.selectors])
        
            np.savetxt(self.directory/(f"{sys_name}-{self.name}_metadata.txt"), data, fmt="%24d", header=header)

        if ret_indces:
            return merged_final_indices
        else:
            selected_frames = [frames[i] for i in merged_final_indices]
            write(self.directory/f"merged-{selector.name}-selection-final.xyz", selected_frames)

            return selected_frames

class ConvergenceSelector(AbstractSelector):

    """ find geometrically converged frames
    """

    name = "convergence"

    def __init__(self, fmax=0.05, directory=Path.cwd()):
        """"""
        self.fmax = fmax # eV

        self.directory = directory

        return
    
    def select(self, frames, index_map = None, ret_indices: bool=False):
        """"""
        # NOTE: input atoms should have constraints attached
        selected_indices = []
        for i, atoms in enumerate(frames):
            maxforce = np.max(np.fabs(atoms.get_forces(apply_constraint=True)))
            if maxforce < self.fmax:
                selected_indices.append(i)

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]

        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
            return selected_indices

class DeviationSelector(AbstractSelector):

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
        directory = Path.cwd(),
        potential = None # file
    ):
        self.deviation_criteria = criteria
        self.__parse_criteria()

        self.directory = directory

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
    
    def select(self, frames, index_map=None, ret_indices: bool=False) -> List[Atoms]:
        """"""
        energy_deviations = [a.info[self.energy_tag] for a in frames]
        force_deviations = [a.info[self.force_tag] for a in frames] # TODO: may not exist
        selected_indices = self._select_indices(energy_deviations, force_deviations)

        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        
        if not ret_indices:
            selected_frames = [frames[i] for i in selected_indices]
            return selected_frames
        else:
            return selected_indices
    
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

        self.directory = Path(directory)

        self.njobs = config.NJOBS

        print("selector uses njobs ", self.njobs)

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

    def select(self, frames, index_map=None, ret_indices: bool=False) -> List[Atoms]:
        """"""
        if len(frames) == 0:
            return []

        features = self.calc_desc(frames)

        selected_indices = self._select_indices(features)
        # map selected indices
        if index_map is not None:
            selected_indices = [index_map[s] for s in selected_indices]
        # if manually_selected is not None:
        #    selected.extend(manually_selected)

        print(f"nframes {len(frames)} -> nselected {len(selected_indices)}")

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


def create_selector(input_list: list, directory=Path.cwd()):
    selectors = []
    for s in input_list:
        params = s.copy()
        method = params.pop("method", None)
        if method == "convergence":
            selectors.append(ConvergenceSelector(**params))
        elif method == "deviation":
            selectors.append(DeviationSelector(**params))
        elif method == "descriptor":
            selectors.append(DescriptorBasedSelector(**params))
        else:
            raise RuntimeError(f"Cant find selector with method {method}.")
    
    # - try a simple composed selector
    if len(selectors) > 1:
        selector = ComposedSelector(selectors, directory=directory)
    else:
        selector = selectors[0]
        selector.directory = directory

    return selector


if __name__ == "__main__":
    pass