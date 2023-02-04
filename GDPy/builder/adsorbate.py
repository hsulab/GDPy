#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import copy
import pathlib
from typing import Union, List

import numpy as np
import networkx as nx

from joblib import Parallel, delayed

import ase
from ase import Atoms
from ase.io import read, write
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy import config
from GDPy.builder.builder import StructureGenerator
from GDPy.builder.species import build_species

from GDPy.graph.creator import StruGraphCreator
from GDPy.graph.sites import SiteFinder
from GDPy.graph.utils import unpack_node_name
from GDPy.graph.comparison import get_unique_environments_based_on_bonds, paragroup_unique_chem_envs

from GDPy.utils.command import CustomTimer


def make_clean_atoms(atoms_, results=None):
    """Create a clean atoms from the input."""
    atoms = Atoms(
        symbols=atoms_.get_chemical_symbols(),
        positions=atoms_.get_positions().copy(),
        cell=atoms_.get_cell().copy(),
        pbc=copy.deepcopy(atoms_.get_pbc())
    )
    if results is not None:
        spc = SinglePointCalculator(atoms, **results)
        atoms.calc = spc

    return atoms

def single_create_structure_graph(graph_params: dict, spec_params: dict, atoms: Atoms) -> List[nx.Graph]:
    """Create structure graph and get selected chemical environments.

    Find atoms with selected chemical symbols or in the defined region.

    Args:
        graph_params: Parameters for the graph representation.
        spec_params: Selected criteria.
        atoms: Input structure.

    Returns:
        A list of graphs that represent the chemical environments of selected atoms.

    """
    stru_creator = StruGraphCreator(**graph_params)

    # - check if spec_indices are all species
    selected_species = spec_params["selected_species"]
    #print(selected_species)

    chemical_symbols = atoms.get_chemical_symbols()
    #print(chemical_symbols)

    spec_indices = spec_params.get("spec_indices", None)
    if spec_indices is None:
        region = spec_params.get("region", None)
        if region is None:
            ads_indices = [] # selected indices
            for i, sym in enumerate(chemical_symbols):
                if sym in selected_species:
                    ads_indices.append(i)
        else:
            ads_indices = []
            #print(region)
            (ox, oy, oz, xl, yl, zl, xh, yh, zh) = region
            for i, a in enumerate(atoms):
                if a.symbol in selected_species:
                    pos = a.position
                    if (
                        (ox+xl <= pos[0] <= ox+xh) and
                        (oy+yl <= pos[1] <= oy+yh) and
                        (oz+zl <= pos[2] <= oz+zh)
                    ):
                        ads_indices.append(i)
    else:
        ads_indices = copy.deepcopy(spec_indices)
    #print(ads_indices)

    _ = stru_creator.generate_graph(atoms, ads_indices_=ads_indices)

    #print(stru_creator.graph)
    #return stru_creator.graph

    chem_envs = stru_creator.extract_chem_envs(atoms)
    return chem_envs

def single_insert_adsorbate(graph_params: dict, idx, atoms, ads, site_params: list, pfunc=print):
    """Insert adsorbate into the graph.
    """
    #print(f"====== create sites {i} =====")
    #pfunc("check_unique in sites: ", check_unique)
    #print(graph_params)

    distance_to_site = 1.5 # Ang

    site_creator = SiteFinder(**graph_params)
    site_creator.pfunc = pfunc
    site_groups = site_creator.find(atoms, site_params)

    created_frames = []
    for i, (sites, params) in enumerate(zip(site_groups,site_params)):
        ads_params = params.get("ads", [{}])
        cur_frames = []
        for s in sites: 
            ads_frames = s.adsorb(
                ads, site_creator.ads_indices, ads_params
            )
            cur_frames.extend(ads_frames)
        created_frames.extend(cur_frames)
        pfunc(f"group {i} unique sites {len(sites)} with {len(cur_frames)} frames for substrate {idx}.")
    
    return created_frames

def single_remove_adsorbate(graph_params: dict, spec_params: dict, atoms: Atoms):
    """Remove selected particles from the structure.

    Currently, only single atom can be removed. TODO: molecule.

    Args:
        graph_params: Parameters for creating graphs.
        spec_params: Parameters for finding species to remove.

    """
    # - create graph from structure
    stru_creator = StruGraphCreator(
        **graph_params
    )

    # - check if spec_indices are all species
    species = spec_params["species"]

    chemical_symbols = atoms.get_chemical_symbols()

    spec_indices = spec_params.get("spec_indices", None)
    if spec_indices is None:
        region = spec_params.get("region", None)
        if region is None:
            ads_indices = [] # selected indices
            for i, sym in enumerate(chemical_symbols):
                if sym == species:
                    ads_indices.append(i)
        else:
            ads_indices = []
            #print(region)
            (ox, oy, oz, xl, yl, zl, xh, yh, zh) = region
            for i, a in enumerate(atoms):
                if a.symbol == species:
                    pos = a.position
                    if (
                        (ox+xl <= pos[0] <= ox+xh) and
                        (oy+yl <= pos[1] <= oy+yh) and
                        (oz+zl <= pos[2] <= oz+zh)
                    ):
                        ads_indices.append(i)
    else:
        ads_indices = copy.deepcopy(spec_indices)
    #print(ads_indices)

    # TODO: tags for molecule?
    for i in ads_indices:
        if chemical_symbols[i] != species:
            raise RuntimeError("Species to remove is inconsistent for those by indices.")

    # - get chem envs
    _ = stru_creator.generate_graph(atoms, ads_indices_=ads_indices)
    chem_envs = stru_creator.extract_chem_envs(atoms)

    # NOTE: for single atom adsorption,
    assert len(chem_envs) == len(ads_indices), "Single atoms group into one adsorbate. Try reducing the covalent radii."
    # TODO: for molecule adsorption

    # - find unique sites to remove for this structure
    unique_indices = get_unique_environments_based_on_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # - create sctructures
    unique_frames = []
    for g in unique_envs:
        for (u, d) in g.nodes.data():
            if d["central_ads"]:
                chem_sym, idx, offset = unpack_node_name(u)
                if chem_sym == species:
                    new_atoms = atoms.copy()
                    del new_atoms[idx]
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs

def single_exchange_adsorbate(graph_params: dict, spec_params: dict, atoms: Atoms):
    """Exchange selected particles from the structure with target species.

    Currently, only single atom can be removed. TODO: molecule.

    Args:
        graph_params: Parameters for creating graphs.
        spec_params: Parameters for finding species to remove.

    """
    # - create graph from structure
    stru_creator = StruGraphCreator(
        **graph_params
    )

    # - check if spec_indices are all species
    species = spec_params["species"]

    chemical_symbols = atoms.get_chemical_symbols()

    spec_indices = spec_params.get("spec_indices", None)
    if spec_indices is None:
        region = spec_params.get("region", None)
        if region is None:
            ads_indices = [] # selected indices
            for i, sym in enumerate(chemical_symbols):
                if sym == species:
                    ads_indices.append(i)
        else:
            ads_indices = []
            #print(region)
            (ox, oy, oz, xl, yl, zl, xh, yh, zh) = region
            for i, a in enumerate(atoms):
                if a.symbol == species:
                    pos = a.position
                    if (
                        (ox+xl <= pos[0] <= ox+xh) and
                        (oy+yl <= pos[1] <= oy+yh) and
                        (oz+zl <= pos[2] <= oz+zh)
                    ):
                        ads_indices.append(i)
    else:
        ads_indices = copy.deepcopy(spec_indices)
    #print(ads_indices)

    # TODO: tags for molecule?
    for i in ads_indices:
        if chemical_symbols[i] != species:
            raise RuntimeError("Species to remove is inconsistent for those by indices.")
    
    target_species = spec_params["target"] # TODO: a molecule?

    # NOTE: compare environments of host species
    # - get chem envs
    _ = stru_creator.generate_graph(atoms, ads_indices_=ads_indices)
    chem_envs = stru_creator.extract_chem_envs(atoms)
    #print("ex chem envs: ", len(chem_envs))
    #print(chem_envs[0])
    #print(chem_envs[1])
    #for i, g in enumerate(chem_envs):
    #    print(g)
    #    #plot_graph(g, f"graph-{i}.png")
    #    for (u, d) in g.nodes.data():
    #        if d["central_ads"]:
    #            print(u, d)
    # NOTE: for single atom adsorption,
    assert len(chem_envs) == len(ads_indices), "Single atoms group into one adsorbate. Try reducing the covalent radii."
    # TODO: for molecule adsorption

    # - find unique sites to remove for this structure
    unique_indices = get_unique_environments_based_on_bonds(chem_envs)
    unique_envs = [chem_envs[i] for i in unique_indices]

    # - create sctructures
    unique_frames = []
    for g in unique_envs:
        for (u, d) in g.nodes.data():
            if d["central_ads"]:
                chem_sym, idx, offset = unpack_node_name(u)
                if chem_sym == species:
                    #new_atoms = make_clean_atoms(atoms)
                    new_atoms = atoms.copy()
                    new_atoms[idx].symbol = target_species # TODO: a molecule?
                    unique_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            ...

    return unique_frames, unique_envs


class AdsorbateGraphGenerator(StructureGenerator):

    """Generate surface structures with adsorbates.

    It generates unique structures based on graph connectivity. This generator 
    produces fixed number of structures.

    """


    def __init__(
        self, params, directory: Union[str,pathlib.Path]="./",
        *args, **kwargs
    ):
        """"""
        super().__init__(directory=directory, *args, **kwargs)

        # - read substrate
        substrate_fpath = params.get("substrate", None)
        assert substrate_fpath is not None, "AdsGen needs substrates."
        substrates_ = read(substrate_fpath, ":")
        self.substrates = substrates_

        # - how to deal with adsorptions
        operations = params.get("operations", None)
        assert operations is not None, "AdsGen needs operations."

        # TODO: only support one species now
        self.operations = operations
        #for data in operations:
        #    self.species = data["species"] # atom or molecule
        #    self.action = data["action"]
        #    self.distance_to_site = data.get("distance_to_site", 1.5)
        #    break
        #else:
        #    pass
        
        # - graph params
        graph_params = params.get("graph", None)
        assert graph_params is not None, "AdsGen needs graph_params."

        self.check_site_unique = graph_params.pop("check_site_unique", True)
        self.graph_params = graph_params

        # - parallel
        self.njobs = config.NJOBS

        return
    
    def run(self, *args, **kwargs) -> List[Atoms]:
        """"""
        super().run(*args, **kwargs) # for the logger...

        # TODO: better check cached results
        if not (self.directory/"enumerated-last.xyz").exists():
            ads_frames = self.substrates
            for i, params in enumerate(self.operations):
                self.pfunc(f"===== Operation {i} =====")
                self.op_num = i # serial number of the operation
                if not (self.directory/f"enumerated-{i}.xyz").exists():
                    ads_frames = self._irun(ads_frames, params, *args, **kwargs)
                else:
                    self.pfunc(f"Use cached results of operation {i}.")
                    ads_frames = read(self.directory/f"enumerated-{i}.xyz", ":")

            write(self.directory/"enumerated-last.xyz", ads_frames)
        else:
            self.pfunc("Use cached results.")
            ads_frames = read(self.directory/"enumerated-last.xyz", ":")

        return ads_frames
    
    def _irun(self, frames_, params: dict, *args, **kwargs) -> List[Atoms]:
        """Run single operation.

        Args:
            frames: Input structures as the substrate.
            params: Parameters for the operation.
        
        Returns:
            Modified structures with given operation.

        """
        # -- make clean substrates and add info
        frames = []
        for i, atoms_ in enumerate(frames_):
            atoms = make_clean_atoms(atoms_)
            subid_ = atoms.info.get("subid", None)
            if subid_ is None:
                subid = i
            else:
                subid = subid_ + "-" + i
            atoms.info["subid"] = subid
            frames.append(atoms)

        nsubstrates = len(frames)
        self.pfunc(f"number of input substrates: {nsubstrates}")

        # - create structures
        action = params.pop("action", "insert") # [insert, remove, exchange]
        if action == "insert":
            self.pfunc("---run insert---")
            # -- parameters for finding adsorption sites
            site_params = copy.deepcopy(params["site_params"])

            # -- parameters for site graph
            graph_params = copy.deepcopy(self.graph_params) # to create site graph
            adsorbate_elements = copy.deepcopy(params.get("adsorbate_elements", []))
            #adsorbate_elements.append(params.get("adsorbate_elemnt", species)) # TODO: not an atom?
            graph_params.update( 
                dict(
                    adsorbate_elements = adsorbate_elements,
                    #coordination_numbers = params.get("coordination_numbers"),
                    site_radius = params.get("site_radius", 2),
                    #check_site_unique = params.get("check_site_unique", True)
                )
            )

            # -- parameters for species used for comparison
            species = params.get("species", None) # to insert
            spec_params = dict(
                species = species,
                selected_species = [],
                spec_indices = params.get("adsorbate_indices", None),
                region = params.get("region", None)
            )

            created_frames = self._insert_adsorbate(
                frames, graph_params, spec_params, site_params
            )

        elif action == "remove":
            # - params for graph creator
            self.pfunc("---run remove---")
            graph_params = copy.deepcopy(self.graph_params)
            adsorbate_elements = copy.deepcopy(params.get("adsorbate_elements", []))
            graph_params.update(adsorbate_elements=adsorbate_elements)

            # - find species to remove
            species = params.get("species")
            spec_params = dict(
                species = species,
                selected_species = [species],
                spec_indices = params.get("adsorbate_indices", None),
                region = params.get("region", None)
            )

            # - run remove
            created_frames = self._remove_adsorbate(frames, graph_params, spec_params)

        elif action == "exchange":
            self.pfunc("---run exchange---")
            # - params for graph creator
            graph_params = copy.deepcopy(self.graph_params)
            adsorbate_elements = copy.deepcopy(params.get("adsorbate_elements", []))
            graph_params.update(adsorbate_elements=adsorbate_elements)

            # - find species to exchange
            species = params.get("species") # host species
            target = params.get("target") # parasite species
            spec_params = dict(
                species = species,
                target = target,
                selected_species = [species, target],
                spec_indices = params.get("adsorbate_indices", None),
                region = params.get("region", None)
            )

            # - run exchange
            created_frames = self._exchange_adsorbate(frames, graph_params, spec_params)

        else:
            raise RuntimeError(f"Unimplemented action {action}.")

        self.pfunc(f"number of output structures: {len(created_frames)}")

        # - add confid
        for i, a in enumerate(created_frames):
            a.info["confid"] = i
        write(self.directory/f"enumerated-{self.op_num}.xyz", created_frames)

        return created_frames

    def _insert_adsorbate(self, frames: List[Atoms], graph_params: dict, spec_params: dict, site_params: dict):
        """Insert adsorbate into the graph.

        Args:
            frames: Subsrate structures.
            species: Adsorbate to insert.
            disatnce_to_site: Distance between site COP and adsorabte COP.
        
        """
        species = spec_params.get("species", None)
        self.pfunc(f"start to insert adsorbate {species}.")

        # - build adsorbate (single atom or molecule)
        #   and update selected_species
        if isinstance(species, str):
            # simple species
            adsorbate = build_species(species)
        else: # dict
            adsorbate = read(species["adsorbate"]) # only one structure
        symbols = list(set(adsorbate.get_chemical_symbols()))

        selected_species = copy.deepcopy(graph_params.get("adsorbate_elements", []))
        selected_species.extend(symbols)
        selected_species = list(set(selected_species))

        spec_params = copy.deepcopy(spec_params)
        spec_params["selected_species"] = selected_species

        # - get structures with inserted species
        with CustomTimer(name="insert-adsorbate", func=self.pfunc):
            #ret = Parallel(n_jobs=self.njobs)(
            ret = Parallel(n_jobs=1)(
                delayed(single_insert_adsorbate)(
                    graph_params, idx, a, adsorbate, site_params,
                    pfunc=self.pfunc
                ) for idx, a in enumerate(frames)
            )
            #ret = []
            #for idx, a in enumerate(frames):
            #    ret.append(
            #        single_insert_adsorbate(
            #            graph_params, idx, a, adsorbate, site_params, pfunc=self.pfunc
            #        )
            #    )
            #print(ads_frames)
        ret_frames = []
        for frames in ret:
            ret_frames.extend(frames)

        write(self.directory/f"possible_frames-{self.op_num}.xyz", ret_frames)

        # NOTE: It is unnecessary to compare among substrates if the spectator
        #       adsorbates are not the same as the inserted one. Otherwise, 
        #       comparasion should be performed.
        created_frames = self._compare_structures(ret_frames, graph_params, spec_params)

        return created_frames
    
    def _remove_adsorbate(self, frames: List[Atoms], graph_params: dict, spec_params: dict):
        """Remove valid adsorbates and check graph differences.
        """
        # - get chem envs of selected species that may be removed
        with CustomTimer(name="remove-adsorbate", func=self.pfunc):
            subids = [a.info["subid"] for a in frames]
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_remove_adsorbate)(graph_params, spec_params, a) 
                for idx, a in enumerate(frames)
            )

            ret_frames, ret_envs = [], []
            for i, (frames, envs) in enumerate(ret):
                nenvs = len(envs)
                subid = subids[i]
                # TODO: add info since it may be lost in atoms.copy() function
                #for a in frames:
                #    a.info["subid"] = subid
                # -- add data
                ret_envs.extend(envs)
                ret_frames.extend(frames)
                self.pfunc(f"number of sites {nenvs} to remove for substrate {subid}.")
        #nsites = len(ret_frames)
        #self.pfunc(f"Total number of chemical environments: {nsites}")

        # - further unique envs among different substrates
        #   only compare chemical environments
        #unique_indices = get_unique_environments_based_on_bonds(ret_envs)
        #created_frames = [ret_frames[i] for i in unique_indices]

        # - compare the graph of chemical environments in the structure
        #   NOTE: if O atoms were to remove, the chem envs of the rest O atoms 
        #         are used to compare the structure difference.
        write(self.directory/f"possible_frames-{self.op_num}.xyz", ret_frames)

        # - get unique structures among substrates
        created_frames = self._compare_structures(ret_frames, graph_params, spec_params)

        return created_frames
    
    def _exchange_adsorbate(self, frames: List[Atoms], graph_params: dict, spec_params: dict, selected_indices=None):
        """Exchange an adsorbate to another species.
        """
        # - get possible sites to exchange
        with CustomTimer(name="exchange-adsorbate", func=self.pfunc):
            subids = [a.info["subid"] for a in frames]
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_exchange_adsorbate)(graph_params, spec_params, a) 
                for a in frames
            )

            ret_frames, ret_envs = [], []
            for i, (frames, envs) in enumerate(ret):
                nenvs = len(envs)
                subid = subids[i]
                # TODO: add info since it may be lost in atoms.copy() function
                #for a in frames:
                #    a.info["subid"] = subid
                # -- add data
                ret_envs.extend(envs)
                ret_frames.extend(frames)
                self.pfunc(f"number of sites {nenvs} to exchange for substrate {subid}.")
        # not unique across substrates
        write(self.directory/f"possible_frames-{self.op_num}.xyz", ret_frames)

        # - compare the graph of chemical environments in the structure
        #   NOTE: if Zn atoms were to exchange with Cr, the chem envs of 
        #         the rest Zn atoms are used to compare the structure difference.
        #         TODO: consider Cr as well?
        created_frames = self._compare_structures(ret_frames, graph_params, spec_params)

        return created_frames
    
    def _compare_structures(self, ret_frames: List[Atoms], graph_params, spec_params):
        """"""
        with CustomTimer(name="create-graphs", func=self.pfunc):
            ret = Parallel(n_jobs=self.njobs)(
                delayed(single_create_structure_graph)(graph_params, spec_params, a)
                for a in ret_frames
            )
        # not unique across substrates
        #write(self.directory/f"possible_frames-{self.op_num}.xyz", ret_frames)

        # - check if the ret is empty
        #   it happens when all species are removed/exchanged...
        ret_envs = []
        for x in ret:
            ret_envs.extend(x)

        if ret_envs:
            ret_env_groups = ret
            self.pfunc("Typical Chemical Environment "+str(ret_envs[0]))
            with CustomTimer(name="check-uniqueness"):
                # compare chem envs
                #unique_envs, unique_groups = unique_chem_envs(
                #    chem_groups, list(enumerate(frames))
                #)
                unique_envs, unique_groups = paragroup_unique_chem_envs(
                    ret_env_groups, list(enumerate(ret_frames)), directory=self.directory, 
                    #n_jobs=self.njobs
                    n_jobs=1
                )
                #self.pfunc("number of unique groups: ", len(unique_groups))

            # - get unique structures
            created_frames = [] # graphly unique
            for x in unique_groups:
                created_frames.append(x[0][1])
            ncandidates = len(created_frames)

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

            unique_info_path = self.directory / f"unique-info-{self.op_num}.txt"
            with open(unique_info_path, "w") as fopen:
                fopen.write(content)
        else:
            self.pfunc("Cant find valid species...")
            created_frames = ret_frames
            ncandidates = len(created_frames)

        return created_frames


if __name__ == "__main__":
    ...