#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path

import numpy as np

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write

from GDPy.utils.command import parse_input_file

from GDPy.graph.creator import StruGraphCreator, SiteGraphCreator
from GDPy.graph.creator import unique_chem_envs, compare_chem_envs
from GDPy.graph.utils import plot_graph, unpack_node_name


def add_two():
    xyz_path = "/users/40247882/scratch2/oxides/graph/MyCode/s44s-O1.xyz"
    atoms = read(xyz_path, "16")
    print(atoms)

    site_creator = SiteGraphCreator(
        pbc_grid = [2, 2, 0],
        covalent_ratio = 1.1,
        skin = 0.25,
        adsorbate_elements = ["O"],
        coordination_numbers = [3],
        site_radius=3
    )
    sites = site_creator.convert_atoms(atoms, True)

    print("unique groups: ", len(sites))
    print([len(s) for s in sites])

    print("\n\n===== check site =====")
    ads = Atoms("O", positions=[[0, 0, 0]])
    frames = []
    for ig, sg in enumerate(sites):
        for s in sg: 
            print(s)
            #s.is_occupied(["O"])

            #print(s.known)
            new_atoms = s.adsorb(ads, [64], height=1.3)
            if not isinstance(new_atoms, Atoms):
                print("!!! site may be already occupied!!!", new_atoms)
            else:
                frames.append(new_atoms)
    write("s44s-O2-2.xyz", frames)

    return

def check_unique():
    """"""
    xyz_path = "/mnt/scratch2/users/40247882/oxides/graph/s44s-O2.xyz"
    frames = read(xyz_path, ":")

    stru_creator = StruGraphCreator(
        pbc_grid = [2, 2, 0],
        graph_radius = 1,
        covalent_ratio = 1.1,
        skin = 0.25,
        adsorbate_elements = ["O"]
    )

    def cmp_two(a, b, frames):
        print(f"compare index {a} and {b}")
        frames = [frames[a], frames[b]]

        # check f1 and f2
        atoms1 = frames[0]
        _ = stru_creator.generate_graph(atoms1)
        chem_envs_1 = stru_creator.extract_chem_envs()

        plot_graph(chem_envs_1[0], fig_name=f"ads-1.png")

        atoms2 = frames[1]
        _ = stru_creator.generate_graph(atoms2)
        chem_envs_2 = stru_creator.extract_chem_envs()

        plot_graph(chem_envs_2[0], fig_name=f"ads-2.png")

        print("comparasion: ", compare_chem_envs(chem_envs_1, chem_envs_2))
    
    #cmp_two(8, 9, frames)
    #exit()

    chem_groups = []
    for i, atoms in enumerate(frames):
        print(f"check frame {i}")
        _ = stru_creator.generate_graph(atoms)
        chem_envs = stru_creator.extract_chem_envs()
        chem_groups.append(chem_envs)

        print("number of adsorbate graphs: ", len(chem_envs))
    
    unique_envs, unique_groups = unique_chem_envs(
        chem_groups, list(enumerate(frames))
    )
    print("number of unique groups: ", len(unique_groups))

    unique_frames = []
    for ug in unique_groups:
        print("{} {} {}".format(ug[0][0], ug[0][1], len(ug)-1))
        for duplicate in ug[1:]:
            print(duplicate[0])

    write("unique.xyz", unique_frames)

    return

def add_adsorbate_single(site_creator, frames):
    created_frames = []
    for i, atoms in enumerate(frames):
        print(f"====== create sites {i} =====")
        sites = site_creator.convert_atoms(atoms, check_unique=True)

        print("unique sites: ", len(sites))
        print([len(s) for s in sites])
        print("nsites: ", sum([len(s) for s in sites]))

        print("\n\n===== check site =====")
        ads = Atoms("O", positions=[[0, 0, 0]]) # TODO: make this an input structure

        for ig, sg in enumerate(sites):
            # NOTE: only choose unique site
            print("number of uniques: ", len(sg))
            for s in sg[:1]: 
                print(s)

                new_atoms = s.adsorb(ads, site_creator.ads_indices, height=1.3)
                if not isinstance(new_atoms, Atoms):
                    print(s, "!!! site may be already occupied!!!", new_atoms)
                else:
                    new_atoms.info["cycle"] = s.cycle
                    new_atoms.arrays["order"] = np.array(range(len(new_atoms)))
                    created_frames.append(new_atoms)

    write("created-O3.xyz", created_frames)

    return

def create_structure_graphs(input_dict, idx, atoms):
    """"""
    stru_creator = StruGraphCreator(
        **input_dict
    )
    _ = stru_creator.generate_graph(atoms)
    chem_envs = stru_creator.extract_chem_envs()

    print(f"check frame {idx}")
    print("number of adsorbate graphs: ", len(chem_envs))

    return chem_envs

def add_adsorbate(input_dict, idx, atoms, ads, check_unique=False):
    #print(f"====== create sites {i} =====")
    site_creator = SiteGraphCreator(**input_dict)
    sites = site_creator.convert_atoms(atoms, check_unique=check_unique) # TODO: custom?

    noccupied = 0

    created_frames = []
    for ig, sg in enumerate(sites):
        # NOTE: only choose unique site
        #print("number of uniques: ", len(sg))
        for s in sg[:1]: 
            # print(s)

            new_atoms = s.adsorb(ads, site_creator.ads_indices, height=1.3)
            if not isinstance(new_atoms, Atoms):
                noccupied += 1
                #print(s, "!!! site may be already occupied!!!", new_atoms)
            else:
                new_atoms.info["cycle"] = s.cycle
                new_atoms.arrays["order"] = np.array(range(len(new_atoms)))
                created_frames.append(new_atoms)

    content = f"===== frame {idx} =====\n"
    content += "unique sites: %d\n" %len(sites)
    content += str([len(s) for s in sites]) + "\n"
    content += "nsites: %d\n" %sum([len(s) for s in sites])
    content += "noccupied: %d\n" %noccupied
    content += "\n\n"
    print(content)
    
    return created_frames

def del_adsorbate(graph_params, atoms, ads_chem_sym):
    """"""
    stru_creator = StruGraphCreator(
        **graph_params
    )

    created_frames = []

    _ = stru_creator.generate_graph(atoms)
    chem_envs = stru_creator.extract_chem_envs()
    print("delete adsorbate number of chem envs: ", len(chem_envs))
    for g in chem_envs:
        for (u, d) in g.nodes.data():
            if d["central_ads"]:
                chem_sym, idx, offset = unpack_node_name(u)
                if chem_sym == ads_chem_sym:
                    new_atoms = atoms.copy()
                    del new_atoms[idx]
                    created_frames.append(new_atoms)
                    break
        else:
            # no valid adsorbate for this structure
            pass

    return created_frames

def extract_unique_structures(chem_groups):
    """ parallel
    """

    return

def graph_main(n_jobs, graph_input_file, stru_path, indices, choice):
    """ sift unique adsorbate structures
        frames must have same chemical formula
    """
    print(f"*** number of processors {n_jobs} ***")

    input_dict = parse_input_file(graph_input_file)
    print(input_dict)

    frames = read(stru_path, indices)
    nframes = len(frames)

    print(f"===== number of frames {nframes}=====")

    if choice == "diff": # analyse chemical environment and output unique structures
        stru_creator = StruGraphCreator(
            **input_dict
        )

        # calculate chem envs
        st = time.time()

        chem_groups = []

        chem_groups = Parallel(n_jobs=n_jobs)(delayed(create_structure_graphs)(input_dict, idx,a) for idx, a in enumerate(frames))

        et = time.time()
        print("*time* calc chem envs: ", et - st)
    
        # compare chem envs - serial
        unique_envs, unique_groups = unique_chem_envs(
            chem_groups, list(enumerate(frames))
        )
        print("number of unique groups: ", len(unique_groups))

        et = time.time()
        print("*time* cmp chem envs: ", et - st)

        # --- compare chem envs - parallel
        pair_indices = []
        for i in range(nframes):
            for j in range(i+1,nframes):
                pair_indices.append((i,j))
        print("number of comparasions: ", len(pair_indices))
        cmp_results = Parallel(n_jobs=n_jobs)(delayed(compare_chem_envs)(chem_groups[i], chem_groups[j]) for i, j in pair_indices)

        # merge results
        last_idx = nframes - 1
        found_last = False
        end_idx = np.cumsum(np.arange(nframes-1,0,-1))
        start_idx = [0] + list(end_idx)[:-1]
        unique_groups = []
        for i, (s, e) in enumerate(zip(start_idx, end_idx)):
            for x in unique_groups:
                if i in x:
                    break
            else:
                cur_res = cmp_results[s:e]
                cur_pairs = pair_indices[s:e]
                ug = [i]
                ug += [cur_pairs[j][1] for j, x in enumerate(cur_res) if x]
                if len(ug) == 0:
                    ug = [i]
                if last_idx in ug:
                    found_last = True
                unique_groups.append(ug)
                print(ug)
        # check last
        if not found_last:
            ug.append([last_idx])
        # reformat
        unique_groups = [[[m, frames[m]]for m in x] for x in unique_groups]

        et = time.time()
        print("*time* cmp chem envs: ", et - st)

        unique_data = []
        for i, x in enumerate(unique_groups):
            data = ["ug"+str(i)]
            data.extend([a[0] for a in x])
            unique_data.append(data)
        content = "# unique, indices\n"
        content += f"# ncandidates {nframes}\n"
        for d in unique_data:
            content += ("{:<8s}  "+"{:<8d}  "*(len(d)-1)+"\n").format(*d)

        with open(Path.cwd() / "unique-g.txt", "w") as fopen:
            fopen.write(content)

        #all_unique = []
        #for iu, ug in enumerate(unique_groups):
        #    unique_frames = []
        #    for x in ug:
        #        x[1].info["confid"] = x[0]
        #        unique_frames.append(x[1])
        #    unique_frames = sorted(unique_frames, key=lambda a: a.get_potential_energy(), reverse=False)
        #    write(f"unique-g{iu}.xyz", unique_frames)
        #    all_unique.append(unique_frames[0])
        #    #print("{} {} {}".format(ug[0][0], ug[0][1], len(ug)-1))
        #    print("{} {}".format(ug[0][0], len(ug)-1))
        #    for duplicate in ug[1:]:
        #        print(duplicate[0])
        #write("all-unique.xyz", all_unique)

    elif choice == "add":
        site_creator = SiteGraphCreator(
            **input_dict
        )
        print("input surface mask: ", site_creator.surface_mask)

        # TODO: how to use clean? 
        print("input clean: ", site_creator.clean)

        # joblib version
        st = time.time()

        ads = Atoms("O", positions=[[0, 0, 0]]) # TODO: make this an input structure

        ads_frames = Parallel(n_jobs=n_jobs)(delayed(add_adsorbate)(input_dict, idx, a, ads) for idx, a in enumerate(frames))
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)
        write("created-O3.xyz", created_frames)

        et = time.time()
        print("time: ", et - st)

    else:
        pass

    return

if __name__ == "__main__":
    add_two()
    #check_unique()
    pass