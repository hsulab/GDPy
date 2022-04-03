#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
from collections import Counter

import numpy as np

from joblib import Parallel, delayed

from ase import Atoms
from ase.io import read, write
from ase.constraints import FixAtoms
from ase.calculators.singlepoint import SinglePointCalculator

from GDPy.potential.manager import PotManager
from GDPy.utils.command import parse_input_file
from GDPy.graph.creator import StruGraphCreator, SiteGraphCreator
from GDPy.graph.utils import unique_chem_envs, compare_chem_envs
from GDPy.graph.graph_main import create_structure_graphs, add_adsorbate, del_adsorbate
from GDPy.graph.utils import unpack_node_name

from ase.ga.standard_comparators import EnergyComparator
from ase.ga.standard_comparators import InteratomicDistanceComparator, get_sorted_dist_list

class AdsorbateEvolution():

    def __init__(self, input_dict, njobs, calc):
        """"""
        self.njobs = njobs
        self.calc = calc

        self.graph_params = input_dict["system"]["graph"]

        self.pot_params = input_dict["potential"]

        # read substrate 
        self.substrates = read(input_dict["system"]["substrate"], ":")

        # mutation
        mutations = input_dict["mutation"] # add, delete, exchange
        assert len(mutations) == 1, "only one mutation can be added to this evolution..."
        self.mut_op = list(mutations.keys())[0]

        # create adsorbate
        composition = input_dict["system"]["composition"]
        assert len(composition) == 1, "only one element is support"

        self.ads_chem_sym = list(composition.keys())[0]
        self.adsorbate = Atoms(self.ads_chem_sym, positions=[[0., 0., 0.]])
        self.ads_number = composition[self.ads_chem_sym]

        # selection
        self.energy_cutoff = input_dict["selection"]["energy_cutoff"]

        self.check_site_unique = input_dict["system"].get("check_site_uniqe", True)

        return
    
    def run(self):
        """"""
        # check start
        start, end = self.ads_number
        if isinstance(self.substrates, Atoms):
            self.substrates = [self.substrates]
        chemical_symbols = self.substrates[0].get_chemical_symbols()
        chem_dict = Counter(chemical_symbols)
        start = chem_dict[self.ads_chem_sym]
        for i in range(1, len(self.substrates)):
            chemical_symbols = self.substrates[i].get_chemical_symbols()
            chem_dict = Counter(chemical_symbols)
            cur_nads = chem_dict[self.ads_chem_sym]
            if cur_nads != start:
                raise RuntimeError("substrates should have same number of adsorbates..")
        start += 1
        ntop = start

        # first generation
        new_substrates = [a.copy() for a in self.substrates]
        #for nads in range(start, end):
        #    print(f"===== generation for {nads} adsorbates =====")
        #    created_frames = self.add_adsorbate(new_substrates)
        #    

        print(f"===== generation for {start} adsorbates =====")
        # use monte carlo to select substrates
        nsubstrates = len(new_substrates)
        print("number of substrates: ", nsubstrates)
        if nsubstrates > 100: # TODO: preset value
            new_substrates = sorted(new_substrates, key=lambda a: a.info["energy"], reverse=False) # BUG???
            putative_minimum = new_substrates[0].info["energy"]
            upper_energy = putative_minimum + self.energy_cutoff 
            for i, a in enumerate(new_substrates):
                if a.info["energy"] > upper_energy:
                    upper_idx = i
                    break
            else:
                upper_idx = nsubstrates
            new_substrates = new_substrates[:upper_idx]

        nsubstrates = len(new_substrates)
        print("number of substrates after selection: ", nsubstrates)

        res_dir = Path.cwd() / "results"
        if not res_dir.exists():
            res_dir.mkdir()
        else:
            pass
        ug_path = res_dir / "ug-candidates.xyz"

        if not ug_path.exists():
            # --- test single run
            print("--- adsorbate creation ---")
            if self.mut_op == "add":
                created_frames = self.add_adsorbate(new_substrates)
            elif self.mut_op == "delete":
                created_frames = self.del_adsorbate(new_substrates)
            elif self.mut_op == "exchange":
                pass

            print(f"number of adsorbate structures: {len(created_frames)}")
            # add confid
            for i, a in enumerate(created_frames):
                a.info["confid"] = i

            ncandidates = len(created_frames)
            write(res_dir / "possible_candidates.xyz", created_frames)

            # compare structures
            # --- graph
            unique_groups = self.compare_graphs(created_frames)
            print(f"number of unique groups: {len(unique_groups)}")
            unique_data = []
            for i, x in enumerate(unique_groups):
                data = ["ug"+str(i)]
                data.extend([a[0] for a in x])
                unique_data.append(data)
            content = "# unique, indices\n"
            content += f"# ncandidates {ncandidates}\n"
            for d in unique_data:
                content += ("{:<8s}  "+"{:<8d}  "*(len(d)-1)+"\n").format(*d)

            with open(res_dir / "unique-g.txt", "w") as fopen:
                fopen.write(content)

            # --- distance

            # only calc unique ones
            unique_candidates = [] # graphly unique
            for x in unique_groups:
                unique_candidates.append(x[0][1])
            write(res_dir / "ug-candidates.xyz", unique_candidates)
        else:
            print("use previous ug-candidates.xyz !!!")
            unique_candidates = read(ug_path, ":")
        nugcands = len(unique_candidates)
        print("number of unique candidates: ", nugcands)

        # --- energy
        # calc every candidate if the number is smaller than a preset value
        if self.calc:
            # use worker to min structures
            tmp_folder = Path.cwd() / "tmp_folder"
            if not tmp_folder.exists():
                tmp_folder.mkdir()
            worker, run_params = self.__register_worker()
            #cons = FixAtoms(indices = list(range(16)))

            with open(res_dir / "calc_candidates.xyz", "w") as fopen:
                fopen.write("")

            new_frames = []
            for i, atoms in enumerate(unique_candidates):
                confid = atoms.info["confid"]
                print("structure ", confid)
                # TODO: skip structures
                dump_path = tmp_folder / ("cand"+str(confid)) / "surface.dump"
                new_atoms = atoms.copy()
                worker.reset()
                worker.set_output_path(tmp_folder / ("cand"+str(confid)))
                if not dump_path.exists():
                    new_atoms = worker.minimise(new_atoms, **run_params)

                    # NOTE: move this to dynamics calculator?
                    #energy = new_atoms.get_potential_energy()
                    #forces = new_atoms.get_forces().copy()
                    #calc = SinglePointCalculator(
                    #    new_atoms, energy=energy, forces=forces
                    #)
                    #new_atoms.calc = calc
                else:
                    print("read existing...")
                    new_atoms = worker.run(new_atoms, read_exists=True, **run_params)

                write(res_dir / "calc_candidates.xyz", new_atoms, append=True)
                new_frames.append(new_atoms)

            new_frames = sorted(new_frames, key=lambda a: a.get_potential_energy(), reverse=False)

            write(res_dir / "calc_candidates.xyz", new_frames)

            # compare by energies
            all_unique = []
            unique_groups = []
            for i, en in enumerate(new_frames):
                for j, (u_indices, u_frames, u_ens) in enumerate(unique_groups):
                    new_en = new_frames[i].get_potential_energy()
                    en_diff = np.fabs(new_en - np.mean(u_ens))
                    en_flag = (en_diff <= 2e-4) # TODO
                    if en_flag:
                        dis_diff = self.__compare_distances(new_frames[i], u_frames[0], ntop=ntop)
                        dis_flag = (dis_diff <= 0.01) # TODO
                        if dis_flag:
                            u_indices.append(i)
                            u_frames.append(new_frames[i])
                            u_ens.append(new_en)
                            break
                else:
                    new_en = new_frames[i].get_potential_energy()
                    unique_groups.append(
                        ([i], [new_frames[i]], [new_en])
                    )
            #print(unique_groups)
            #print("!!!nuique: ", len(unique_groups))
            all_unique.extend(unique_groups)

            # write true unique frames
            unique_data = []
            unique_frames = []
            for i, (u_indices, u_frames, u_ens) in enumerate(all_unique):
                print(i, "energy: {} indices: {}".format(u_ens, u_indices))
                unique_frames.append(u_frames[0])
                content = ("uged{:s}  "+"{:<8.4f}  ").format(str(i), u_ens[0])+("{:<6d}  "*len(u_indices)).format(*u_indices)
                unique_data.append(content)
                #write(f"./ged-uniques/u-ged-{i}.xyz", u_frames)
            unique_frames = sorted(unique_frames, key=lambda a: a.get_potential_energy(), reverse=False)
            write(res_dir / "uged-calc_candidates.xyz", unique_frames)
            with open(res_dir / "unique-ged.txt", "w") as fopen:
                fopen.write("\n".join(unique_data))


        return
    
    def add_adsorbate(self, frames):
        """"""
        # joblib version
        st = time.time()

        ads_frames = Parallel(n_jobs=self.njobs)(
            delayed(add_adsorbate)(self.graph_params, idx, a, self.adsorbate, self.check_site_unique) for idx, a in enumerate(frames)
        )
        #print(ads_frames)

        created_frames = []
        for af in ads_frames:
            created_frames.extend(af)

        et = time.time()
        print("add_adsorbate time: ", et - st)

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
    
    def compare_graphs(self, frames):
        """"""
        # calculate chem envs
        st = time.time()

        chem_groups = []
        #for i, atoms in enumerate(frames):
        #    print(f"check frame {i}")
        #    _ = stru_creator.generate_graph(atoms)
        #    chem_envs = stru_creator.extract_chem_envs()
        #    chem_groups.append(chem_envs)
        #    print("number of adsorbate graphs: ", len(chem_envs))

        chem_groups = Parallel(n_jobs=self.njobs)(delayed(create_structure_graphs)(self.graph_params, idx, a) for idx, a in enumerate(frames))

        et = time.time()
        print("calc chem envs: ", et - st)
    
        # compare chem envs
        unique_envs, unique_groups = unique_chem_envs(
            chem_groups, list(enumerate(frames))
        )
        print("number of unique groups: ", len(unique_groups))

        et = time.time()
        print("cmp chem envs: ", et - st)

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

        return unique_groups
    
    def __compare_distances(self, a1, a2, ntop):
        """ compare distances
        """
        pc1 = get_sorted_dist_list(a1[-ntop:], mic=True)
        pc2 = get_sorted_dist_list(a2[-ntop:], mic=True)

        diffs = []
        for key in pc1.keys():
            diffs.append(
                np.max(np.fabs(pc1[key] - pc2[key]))
            )

        return np.max(diffs)

    # --- compare structures ---
    def cmp_structures(self, frames):
        # compare by energies
        all_unique = []
        unique_groups = []
        for i, en in enumerate(frames):
            for j, (u_indices, u_frames, u_ens) in enumerate(unique_groups):
                new_en = frames[i].get_potential_energy()
                #if en_comp.looks_like(frames[i], u_frames[0]): # TODO
                en_diff = np.fabs(new_en - np.mean(u_ens))
                if en_diff <= 2e-4:
                    u_indices.append(i)
                    u_frames.append(frames[i])
                    u_ens.append(new_en)
                    break
                elif en_diff <= 1e-2:
                    # TODO: interatomic distance comparator
                    if compare_distances(frames[i], u_frames[0]):
                        u_indices.append(i)
                        u_frames.append(frames[i])
                        u_ens.append(new_en)
                        break
                else:
                    pass
            else:
                new_en = frames[i].get_potential_energy()
                unique_groups.append(
                    ([i], [frames[i]], [new_en])
                )
        #print(unique_groups)
        #print("!!!nuique: ", len(unique_groups))
        all_unique.extend(unique_groups)

        # write true unique frames
        unique_frames = []
        for i, (u_indices, u_frames, u_en) in enumerate(all_unique):
            print(i, "energy: {} indices: {}".format(u_en, u_indices))
            unique_frames.append(u_frames[0])
            write(f"./ged-uniques/u-ged-{i}.xyz", u_frames)
        unique_frames = sorted(unique_frames, key=lambda a: a.get_potential_energy(), reverse=False)
        write("unique-ged-O6.xyz", unique_frames)

        return
    
    def __register_worker(self):
        #input_dict = parse_input_file("../../potential-ase.yaml")
        #input_dict = parse_input_file("../../potential-lammps.yaml")
        #print(input_dict)

        input_dict = self.pot_params

        atype_map = {}
        for i, a in enumerate(input_dict["calc_params"]["type_list"]):
            atype_map[a] = i

        # create potential
        mpm = PotManager() # main potential manager
        eann_pot = mpm.create_potential(
            pot_name = input_dict["name"],
            # TODO: remove this kwargs
            backend = "ase",
            models = input_dict["calc_params"]["pair_style"]["model"],
            type_map = atype_map
        )

        worker, run_params = eann_pot.create_worker(
            backend = input_dict["backend"],
            calc_params = input_dict["calc_params"],
            dyn_params = input_dict["dyn_params"]
        )
        print(run_params)

        return worker, run_params


    # --- run calculation ---
    def run_calc():
        # use worker to min structures
        #cons = FixAtoms(indices = list(range(16)))
        new_frames = []
        frames = read("./unique-O7-g.xyz", ":") # TODO...
        for i, atoms in enumerate(frames):
            print(f"structure {i}")
            new_atoms = atoms.copy()
            new_atoms.set_constraint(cons)
            worker.reset()
            worker.set_output_path(f"./tmp_folder/cand{i}")

            new_atoms = worker.minimise(new_atoms, **run_params)

            energy = new_atoms.get_potential_energy()
            forces = new_atoms.get_forces().copy()
            calc = SinglePointCalculator(
                new_atoms, energy=energy, forces=forces
            )
            new_atoms.calc = calc
            new_frames.append(new_atoms)

        # TODO: sort energies
        new_frames = sorted(new_frames, key=lambda a: a.get_potential_energy(), reverse=False)

        write("new_frames.xyz", new_frames)

        return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "INPUT"
    )
    parser.add_argument(
        "-nj", "--njobs", default=8, type=int
    )
    parser.add_argument(
        "--calc", action="store_true",
        help = "calculate candidates"
    )
    args = parser.parse_args()

    #input_file = "/mnt/scratch2/users/40247882/oxides/graph/NewTest/input.yaml"
    input_file = args.INPUT

    input_dict = parse_input_file(input_file)
    ae = AdsorbateEvolution(input_dict, args.njobs, args.calc)
    ae.run()
    pass