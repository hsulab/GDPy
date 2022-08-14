#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from typing import List

from joblib import Parallel, delayed

import ase
from ase import Atoms
from ase.io import read, write

from GDPy import config
from GDPy.builder.builder import StructureGenerator
from GDPy.builder.species import build_species

from GDPy.graph.creator import StruGraphCreator, SiteGraphCreator
from GDPy.graph.utils import unique_chem_envs, compare_chem_envs
from GDPy.graph.graph_main import create_structure_graphs, add_adsorbate, del_adsorbate, exchange_adsorbate
from GDPy.graph.utils import unpack_node_name
from GDPy.graph.para import paragroup_unique_chem_envs

from GDPy.utils.command import CustomTimer


class AdsorbateGraphGenerator(StructureGenerator):

    """ generate initial configurations with 
        different adsorbates basedo on graphs
    """


    def __init__(self, substrate: str, composition: dict, graph: dict, directory=Path.cwd()):
        """"""
        # - read substrate
        self.substrate = read(substrate, ":")

        # --- unpack some params
        # TODO: only support one species now
        for data in composition:
            self.species = data["species"] # atom or molecule
            self.action = data["action"]
            self.distance_to_site = data.get("distance_to_site", 1.5)
            break
        else:
            pass

        self.check_site_unique = graph.pop("check_site_unique", True)
        self.graph_params = graph

        self.directory = Path(directory)

        self.njobs = config.NJOBS

        return
    
    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        frames = self.substrate

        if self.action == "add":
            created_frames = self.add_adsorbate(frames, self.species, self.distance_to_site)
        elif self.action == "delete":
            # TODO: fix bug
            created_frames = self.del_adsorbate(frames)
        elif self.action == "exchange":
            # TODO: fix bug
            print("---run exchange---")
            selected_indices = self.mut_params["selected_indices"]
            print("for atom indices ", selected_indices)
            ads_species, target_species = self.mut_content.split("->")
            ads_species = ads_species.strip()
            assert ads_species == self.ads_chem_sym, "adsorbate species is not consistent"
            target_species = target_species.strip()
            created_frames = self.exchange_adsorbate(
                frames, target_species, selected_indices=selected_indices
            )

        print(f"number of adsorbate structures: {len(created_frames)}")
        # add confid
        for i, a in enumerate(created_frames):
            a.info["confid"] = i
        ncandidates = len(created_frames)
        
        # - compare frames based on graphs
        selected_indices = [None]*len(created_frames)
        if self.action == "exchange":
            # find target species
            for fidx, a in enumerate(created_frames):
                s = []
                for i, x in enumerate(a.get_chemical_symbols()):
                    if x == target_species:
                        s.append(i)
                selected_indices[fidx] = s

        unique_groups = self.compare_graphs(created_frames, selected_indices=selected_indices)
        print(f"number of unique groups: {len(unique_groups)}")

        # -- unique info
        unique_data = []
        for i, x in enumerate(unique_groups):
            data = ["ug"+str(i)]
            data.extend([a[0] for a in x])
            unique_data.append(data)
        content = "# unique, indices\n"
        content += f"# ncandidates {ncandidates}\n"
        for d in unique_data:
            content += ("{:<8s}  "+"{:<8d}  "*(len(d)-1)+"\n").format(*d)

        unique_info_path = self.directory / "unique-g.txt"
        with open(unique_info_path, "w") as fopen:
            fopen.write(content)

        # --- distance

        # only calc unique ones
        unique_candidates = [] # graphly unique
        for x in unique_groups:
            unique_candidates.append(x[0][1])

        #return created_frames
        return unique_candidates

    def add_adsorbate(self, frames, species: str, distance_to_site: float = 1.5):
        """"""
        print("start adsorbate addition")
        # - build adsorbate
        adsorbate = build_species(species)

        # joblib version
        with CustomTimer(name="add-adsorbate"):
            ads_frames = Parallel(n_jobs=self.njobs)(
                delayed(add_adsorbate)(
                    self.graph_params, idx, a, adsorbate, distance_to_site, check_unique=self.check_site_unique
                ) for idx, a in enumerate(frames)
            )
            #print(ads_frames)

            created_frames = []
            for af in ads_frames:
                created_frames.extend(af)

        return created_frames
    
    def del_adsorbate(self, frames):
        """ delete valid adsorbates and
            check graph differences
        """
        # joblib version
        st = time.time()

        ads_frames = Parallel(n_jobs=self.njobs)(delayed(del_adsorbate)(self.graph_params, a, self.ads_chem_sym) for idx, a in enumerate(frames))
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)

        et = time.time()
        print("del_adsorbate time: ", et - st)

        return created_frames
    
    def exchange_adsorbate(self, frames, target_species, selected_indices=None):
        """ change an adsorbate to another species
        """
        # joblib version
        st = time.time()

        ads_frames = Parallel(n_jobs=self.njobs)(
            delayed(exchange_adsorbate)(self.graph_params, a, self.ads_chem_sym, target_species, selected_indices) for a in frames
        )
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)

        et = time.time()
        print("exg_adsorbate time: ", et - st)

        return created_frames

    def compare_graphs(self, frames, graph_params=None, selected_indices=None):
        """"""
        # TODO: change this into a selector
        # calculate chem envs

        with CustomTimer(name="comparasion"):
            if graph_params is None:
                graph_params = self.graph_params

            chem_groups = Parallel(n_jobs=self.njobs)(
                delayed(create_structure_graphs)(graph_params, idx, a, s) for idx, (a, s) in enumerate(zip(frames,selected_indices))
            )

            with CustomTimer(name="unique"):
                # compare chem envs
                #unique_envs, unique_groups = unique_chem_envs(
                #    chem_groups, list(enumerate(frames))
                #)
                unique_envs, unique_groups = paragroup_unique_chem_envs(
                    chem_groups, list(enumerate(frames)), directory=self.directory, n_jobs=self.njobs
                )

                print("number of unique groups: ", len(unique_groups))

        return unique_groups


if __name__ == "__main__":
    pass