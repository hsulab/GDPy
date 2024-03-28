#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
import pathlib
import re
import warnings

from typing import Any, NoReturn, Optional, List, Mapping

import numpy as np

from ase import Atoms
from ase.io import read, write

from .. import config
from ..core.register import registers
from ..core.variable import Variable
from ..core.operation import Operation

from .array import AtomsNDArray
from .dataset import XyzDataloader
from .system import DataSystem


# --- variable

@registers.variable.register
class TempdataVariable(Variable):

    def __init__(self, system_dirs, *args, **kwargs):
        """"""
        initial_value = self._process_dataset(system_dirs)
        super().__init__(initial_value)

        return
    
    def _process_dataset(self, system_dirs: List[str]):
        """"""
        #system_dirs.sort()
        system_dirs = [pathlib.Path(s) for s in system_dirs]

        dataset = []
        for s in system_dirs:
            prefix, curr_frames = s.name, []
            xyzpaths = sorted(list(s.glob("*.xyz")))
            for p in xyzpaths:
                curr_frames.extend(read(p, ":"))
            dataset.append([prefix, curr_frames])

        return dataset


@registers.variable.register
class NamedTempdataVariable(Variable):

    def __init__(self, system_dirs, *args, **kwargs):
        """"""
        initial_value = self._process_dataset(system_dirs)
        super().__init__(initial_value)

        return
    
    def _process_dataset(self, system_dirs: List[str]):
        """"""
        #system_dirs.sort()
        system_dirs = [pathlib.Path(s) for s in system_dirs]

        data_systems = []
        for s in system_dirs:
            config._debug(str(s))
            d = DataSystem(directory=s)
            data_systems.append(d)

        return data_systems


@registers.variable.register
class DatasetVariable(Variable):

    def __init__(self, name, directory="./", *args, **kwargs):
        """"""
        dataset = registers.create("dataloader", name, convert_name=True, **kwargs)
        super().__init__(initial_value=dataset, directory=directory)

        return


# --- operation
@registers.operation.register
class assemble(Operation):

    def __init__(self, variable, directory="./", **kwargs) -> None:
        """"""
        vkwargs = {} # variable non-variable/operation kwargs
        node_names, input_nodes = [], []
        for k, v in kwargs.items():
            if isinstance(v, (Variable, Operation)):
                node_names.append(k)
                input_nodes.append(v)
            else:
                vkwargs[k] = v
        super().__init__(input_nodes, directory)

        self.variable = variable
        self.vkwargs = vkwargs
        self.node_names = node_names

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()

        # NOTE: check whether dependant nodes all have valid outputs
        is_finished = True
        for o in outputs:
            if o is None:
                is_finished = False
                break
        
        if is_finished:
            params = copy.deepcopy(self.vkwargs)
            params.update({k: v for k, v in zip(self.node_names, outputs)})
            variable = registers.create("variable", self.variable, **params)
            ret = variable.value
        else:
            ret = None

        self.status = "finished" if is_finished else "unfinished"

        return ret

@registers.operation.register
class seqrun(Operation):

    def __init__(self, nodes, *args, **kwargs) -> NoReturn:
        """"""
        super().__init__(nodes)

        return
    
    def forward(self, *outputs):
        """"""
        super().forward()

        return

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

        # NOTE: some dependant nodes may not finish
        self.status = "finished"

        return ret

@registers.operation.register
class zip_nodes(Operation):

    status = "finished"

    def __init__(self, nodes, directory="./") -> None:
        """"""
        super().__init__(input_nodes=nodes, directory=directory)

        return

    def forward(self, *outputs):
        """"""
        super().forward()

        ret = [list(x) for x in zip(*outputs)]

        return ret

@registers.operation.register
class list_nodes(Operation):

    status = "finished"

    def __init__(self, nodes, directory="./") -> None:
        """"""
        super().__init__(input_nodes=nodes, directory=directory)

        return

    def forward(self, *outputs):
        """"""
        super().forward()

        ret = list(outputs)

        return ret


@registers.operation.register
class transfer(Operation):

    """Transfer worker results to target destination.
    """

    def __init__(
        self, structures, dataset, version, 
        prefix: str="", system: str="mixed", clean_info: bool=False,
        directory="./"
    ) -> None:
        """"""
        input_nodes = [structures, dataset]
        super().__init__(input_nodes=input_nodes, directory=directory)

        self.version = version

        self.prefix = prefix
        self.system = system # molecule/cluster, surface, bulk

        self.clean_info = clean_info # whether clean atoms info

        return
    
    def forward(self, frames: List[Atoms], dataset):
        """"""
        super().forward()

        if isinstance(frames, AtomsNDArray):
            frames = frames.get_marked_structures()
        nframes = len(frames)
        self._print(f"{nframes = }")

        target_dir = dataset.directory.resolve()
        self._print(f"target dir: {str(target_dir)}")

        # - check chemical symbols
        system_dict = {} # {formula: [indices]}

        # NOTE: groupby only collects contiguous data
        #       we need aggregate by ourselves
        formulae = [a.get_chemical_formula() for a in frames]
        for k, v in itertools.groupby(enumerate(formulae), key=lambda x: x[1]):
            if k not in system_dict:
                system_dict[k] = [x[0] for x in v]
            else:
                system_dict[k].extend([x[0] for x in v])
        
        # - transfer data
        acc_nframes = 0
        for formula, curr_indices in system_dict.items():
            # -- TODO: check system type
            system_type = self.system # currently, use user input one
            # -- name = description+formula+system_type
            #dirname = "-".join([self.directory.parent.name, formula, system_type])
            dirname = "-".join([self.prefix, formula, system_type])

            target_subdir = target_dir/dirname
            target_subdir.mkdir(parents=True, exist_ok=True)

            # -- save frames
            curr_frames = [frames[i] for i in curr_indices]
            curr_nframes = len(curr_frames)

            if self.clean_info:
                self._clean_frames(curr_frames)

            strname = self.version + ".xyz"
            target_destination = target_dir/dirname/strname
            if not target_destination.exists():
                write(target_destination, curr_frames)
                self._print(f"nframes {curr_nframes} -> {str(target_destination.relative_to(target_dir))}")
            else:
                #warnings.warn(f"{target_destination} exists.", UserWarning)
                self._print(f"WARN: {str(target_destination.relative_to(target_dir))} exists.")

            acc_nframes += curr_nframes

        assert nframes == acc_nframes
        
        self.status = "finished"

        return dataset
    
    def _clean_frames(self, frames: List[Atoms]):
        """"""
        for atoms in frames:
            info_keys = copy.deepcopy(list(atoms.info.keys()))
            for k in info_keys:
                if k not in ["energy", "free_energy"]:
                    del atoms.info[k]

        return


@registers.operation.register
class scope(Operation):

    def __init__(
            self, dataset, describer, 
            groups: Optional[dict]=None, subgroups: Optional[dict]=None, level: int=0,
            add_legend: bool=True, write_chemiscope=False, directory="./"
        ) -> None:
        """"""
        super().__init__(input_nodes=[dataset, describer], directory=directory)
        
        self.level = level

        if groups is None:
            groups = {"all": r".*"}
        self.groups = {k: fr"{v}" for k, v in groups.items()}

        if subgroups is None:
            subgroups = {"all": r".*"}
        self.subgroups = {k: fr"{v}" for k, v in subgroups.items()}

        self.add_legend = add_legend
        self.write_chemiscope = write_chemiscope

        return
    
    def forward(self, dataset, describer):
        """"""
        super().forward()

        describer.directory = self.directory
        features = describer.run(dataset=dataset)

        starts = [0] + [len(d._images) for d in dataset]
        starts = np.cumsum(starts)
        self._debug(f"starts: {starts}")

        # - get groups
        group_indices = {k: {sk: [] for sk in self.subgroups} for k in self.groups}
        for i, system in enumerate(dataset):
            # -- match group
            for k, v in self.groups.items():
                if re.match(v, system.prefix) is not None:
                    self._print(f"{v}, {system.prefix}")
                    break
            else:
                continue
            # -- match subgroups
            for sk, sv in self.subgroups.items():
                curr_indices = [x+starts[i] for x in system.get_matched_indices(sv)]
                #if sk not in group_indices[k]:
                #    group_indices[k][sk] = curr_indices
                #else:
                #    group_indices[k][sk].extend(curr_indices)
                group_indices[k][sk].extend(curr_indices)
        #self._debug(f"groups: {group_indices}")

        # - merge groups
        merged_groups = {}
        for name, curr_group in group_indices.items():
            for subname, indices in curr_group.items():
                if self.level == 0:
                    gname = f"{name}+{subname}"
                elif self.level == 1:
                    gname = f"{name}"
                elif self.level == 2:
                    gname = f"{subname}"
                else:
                    raise RuntimeError()
                if gname in merged_groups:
                    merged_groups[gname].extend(indices)
                else:
                    merged_groups[gname] = indices

        self._plot_results(features, merged_groups, self.add_legend)

        # - save chemiscope?
        if self.write_chemiscope:
            frames = []
            for d in dataset:
                frames.extend(d._images)
            self._write_chemiscope(frames=frames, features=features)

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

        # -- separate
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))

        curves = []
        for i, (name, indices) in enumerate(groups.items()):
            print(f"{name} -> {len(indices)}")
            if len(indices) > 0:
                proj = reducer.transform(features[indices,:])
                curve = ax.scatter(
                    proj[:, 0], proj[:, 1], alpha=0.25, zorder=100-i,
                    label=f"{name} {len(indices)}"
                )
                curves.append(curve)
        labels = [c.get_label() for c in curves]

        ax.axis("off")
        fig.savefig(self.directory/"pca.png", transparent=True)

        label_params = ax.get_legend_handles_labels()
        if add_legend:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,8))
            ax.axis("off")
            ax.legend(*label_params)
            fig.savefig(self.directory/"pca-legend.png", transparent=True)

        return 

    def _write_chemiscope(self, frames, features):
        """"""
        USE_CHEMISCOPE = 0
        try:
            import chemiscope
            USE_CHEMISCOPE = 1
        except Exception as e:
            print(e)

        from sklearn.decomposition import PCA
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
            str(self.directory/"chemiscope.json.gz"), 
            frames=frames, properties=properties
        )

        return


if __name__ == "__main__":
    ...
