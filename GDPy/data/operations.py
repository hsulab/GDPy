#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import pathlib
import re
from typing import NoReturn, Optional, List, Mapping
import warnings

import numpy as np

from ase import Atoms
from ase.io import read, write

from GDPy.core.operation import Operation
from GDPy.core.register import registers
from GDPy.data.dataset import XyzDataloader
from GDPy.data.array import AtomsArray2D

@registers.operation.register
class end_session(Operation):

    def __init__(self, *args) -> NoReturn:
        super().__init__(args)
    
    def forward(self, *args):
        """"""
        return super().forward()

@registers.operation.register
class chain(Operation):

    """Merge arbitrary nodes' outputs into one list.
    """

    status = "finished" # Always finished since it is not time-consuming

    def __init__(self, nodes, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(nodes)

        # - some operation parameters

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()
        self._debug(f"chain outputs: {outputs}")

        return list(itertools.chain(*outputs))

@registers.operation.register
class map(Operation):

    """Give each input node a name and construct a dict.

    This is useful when validating structures from different systems.

    """

    status = "finished"

    def __init__(self, nodes, names, directory="./") -> NoReturn:
        """"""
        super().__init__(nodes, directory)

        assert len(nodes) == len(names), "Numbers of nodes and names are inconsistent."
        self.names = names

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()

        ret = {}
        for k, v in zip(self.names, outputs):
            ret[k] = v
        
        self._debug(f"map ret: {ret}")

        return ret


@registers.operation.register
class transfer(Operation):

    """Transfer worker results to target destination.
    """

    def __init__(self, structure, target_dir, version, system="mixed", directory="./") -> NoReturn:
        """"""
        input_nodes = [structure]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.target_dir = pathlib.Path(target_dir).resolve()
        self.version = version

        self.system = system # molecule/cluster, surface, bulk

        return
    
    def forward(self, frames: List[Atoms]):
        """"""
        super().forward()

        if isinstance(frames, AtomsArray2D):
            frames = frames.get_marked_structures()

        self._print(f"target dir: {str(self.target_dir)}")

        # - check chemical symbols
        system_dict = {} # {formula: [indices]}

        formulae = [a.get_chemical_formula() for a in frames]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            system_dict[k] = [x[0] for x in v]
        
        # - transfer data
        for formula, curr_indices in system_dict.items():
            # -- TODO: check system type
            system_type = self.system # currently, use user input one
            # -- name = description+formula+system_type
            dirname = "-".join([self.directory.parent.name, formula, system_type])
            target_subdir = self.target_dir/dirname
            target_subdir.mkdir(parents=True, exist_ok=True)

            # -- save frames
            curr_frames = [frames[i] for i in curr_indices]
            curr_nframes = len(curr_frames)

            strname = self.version + ".xyz"
            target_destination = self.target_dir/dirname/strname
            if not target_destination.exists():
                write(target_destination, curr_frames)
                self._print(f"nframes {curr_nframes} -> {target_destination.name}")
            else:
                warnings.warn(f"{target_destination} exists.", UserWarning)
        
        dataset = XyzDataloader(self.target_dir)
        self.status = "finished"

        return dataset


@registers.operation.register
class scope(Operation):

    def __init__(
            self, dataset, describer, 
            groups: Optional[dict]=None, subgroups: Optional[dict]=None, 
            add_legend: bool=True, directory="./"
        ) -> None:
        """"""
        super().__init__(input_nodes=[dataset, describer], directory=directory)

        if groups is None:
            groups = {"all": r".*"}
        self.groups = {k: fr"{v}" for k, v in groups.items()}

        if subgroups is None:
            subgroups = {"all": r".*"}
        self.subgroups = {k: fr"{v}" for k, v in subgroups.items()}

        self.add_legend = add_legend

        return
    
    def forward(self, dataset, describer):
        """"""
        super().forward()

        describer.directory = self.directory
        features = describer.run(dataset=dataset)

        starts = [0] + [len(d._images) for d in dataset]
        starts = np.cumsum(starts)
        self._debug(f"starts: {starts}")

        group_indices = {}
        for i, system in enumerate(dataset):
            for k, v in self.groups.items():
                if re.match(v, system.prefix) is not None:
                    break
            else:
                continue
            for k, v in self.subgroups.items():
                if k not in group_indices:
                    group_indices[k] = [x+starts[i] for x in system.get_matched_indices(v)]
                else:
                    group_indices[k].extend([x+starts[i] for x in system.get_matched_indices(v)])
        self._debug(f"groups: {group_indices}")

        self._plot_results(features, group_indices, self.add_legend)

        return

    def _plot_results(self, features, groups: Mapping[str,List[int]], add_legend, *args, **kwargs):
        """"""
        # - plot selection
        import matplotlib.pyplot as plt
        try:
            plt.style.use("presentation")
        except Exception as e:
            ...

        from sklearn.decomposition import PCA

        reducer = PCA(n_components=2)
        reducer.fit(features)

        for i, (name, indices) in enumerate(groups.items()):
            if len(indices) > 0:
                proj = reducer.transform(features[indices,:])
                plt.scatter(
                    proj[:, 0], proj[:, 1], alpha=0.5, zorder=100-i,
                    label=f"{name} {len(indices)}"
                )

        if add_legend:
            plt.legend()

        plt.axis("off")
        plt.savefig(self.directory/"pca.png")

        return 

if __name__ == "__main__":
    ...