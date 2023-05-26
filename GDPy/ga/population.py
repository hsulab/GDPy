#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

import numpy as np

from ase import Atoms
from ase.io import read, write
from ase.ga.population import Population


class AbstractPopulationManager():

    """An abstract population manager for evolutionary algorithms.

    For structure exploration, there are generally two formulations. ASE forms 
    current population from all candidates while USPEX forms it based on the 
    previous generation. Furthermore, USPEX uses fracGene, fracRand, fracTopRand, 
    fracPerm, fracAtomsMut, fracRotMut, fracLatMut, fracSpinMut...

    Example:
        Parameters needed

        $ cat ga.yaml
        population:
            init: # for the initial population
                size: 50 # not necessarily equal to size if set
                seed_file: ./seed.xyz # seed structures for the initial population
            gen: # for the following generations
                size: 20 # number of structures in each generation
                reproduce: 20 # crossover + mutate
                random: 0

    """

    pfunc = print

    #: Reproduction and mutation.
    MAX_REPROC_TRY: int = 1

    def __init__(self, params: dict) -> None:
        """"""
        # - gen params
        gen_params = params.get(
            "gen", dict(
                size = 20,
                reproduce = 20,
                random = 0
            )
        )
        self.gen_size = gen_params.get("size", None)
        assert isinstance(self.gen_size, int), "size of generaton needs to be an integer."

        self.gen_ran_size = gen_params.get("random", 0)
        self.gen_ran_max_try = gen_params.get("max_random_try", self.gen_ran_size*10)

        self.gen_rep_size = gen_params.get("reprod", self.gen_size-self.gen_ran_size)
        self.gen_rep_max_try = gen_params.get("max_reprod_try", self.gen_rep_size*10)

        self.gen_mut_size = gen_params.get("mutate", self.gen_size-self.gen_ran_size-self.gen_rep_size)
        self.gen_mut_max_try = gen_params.get("max_mutate_try", self.gen_mut_size*10)

        # - init params
        init_params = params.get(
            "init", dict(
                size = 20,
                seed_file = None
            )
        )
        self.init_size = init_params.get("size", None)
        self.init_seed_file = init_params.get("seed_file", None)

        # - check if params were valid
        assert (self.gen_rep_size+self.gen_ran_size+self.gen_mut_size) == self.gen_size, "In each generation, the sum of each component does not equal the total size."
        assert self.gen_ran_size <= self.gen_size, "In each generation, the random size should not be larger than the total size."
        assert self.gen_rep_size <= self.gen_size, "In each generation, the reprod size should not be larger than the total size."
        assert self.gen_mut_size <= self.gen_size, "In each generation, the mutate size should not be larger than the total size."

        # - mutation
        self.pmut = params.get("pmut", 0.5)

        return
    
    def _prepare_initial_population(
            self, generator
        ) -> List[Atoms]:
        self.pfunc("\n\n===== Prepare Initial Population =====")
        starting_population = []

        # - try to read seed structures
        # NOTE: seed structures would be re-optimised by the worker
        self.pfunc("\n----- try to add seed structures -----")
        seed_frames = []
        if self.init_seed_file is not None:
            seed_frames = read(self.init_seed_file, ":")
            seed_size = len(seed_frames)
            assert (seed_size > 0 and seed_size <= self.init_size), "The number of seed structures is invalid."
        else:
            seed_size = 0

        # TODO: check force convergence and only add converged structures
        # TODO: check atom permutation
        for i, atoms in enumerate(seed_frames):
            # TODO: check atom order
            atoms.info["data"] = {}
            atoms.info["key_value_pairs"] = {}
            atoms.info["key_value_pairs"]["origin"] = "seed {}".format(i)
            atoms.info["key_value_pairs"]["raw_score"] = -atoms.get_potential_energy()
            # TODO: check geometric convergence
        self.pfunc(f"number of seed structures: {len(seed_frames)}")
        starting_population.extend(seed_frames)

        # - generate random structures
        self.pfunc("\n----- try to generate random structures -----")
        random_frames = generator.run(size=self.init_size - seed_size)
        self.pfunc(f"number of random structures: {len(random_frames)}")
        starting_population.extend(random_frames)

        if len(starting_population) != self.init_size:
            raise RuntimeError("It fails to generate the initial population. Check the seed file and the system setting.")

        self.pfunc(f"finished creating initial population...")

        #self.init_seed_size = seed_size
        #self.init_ran_size = len(random_frames)

        return starting_population

    def _prepare_current_population(
        self, cur_gen, database, population, generator, pairing, mutations
    ):
        """Prepare current population.

        Usually, it should be the same as the initial size. However, for variat 
        composition search, a large init size can be useful.

        """
        current_candidates = []

        # - reproduction and then mutation
        paired_structures = []
        for i in range(self.gen_rep_max_try):
            atoms = self._reproduce(database, population, pairing, mutations)
            if atoms is not None:
                paired_structures.append(atoms)
                self.pfunc("  --> confid %d\n" %(atoms.info["confid"]))
            if len(paired_structures) == self.gen_rep_size:
                break
        else:
            self.pfunc(f"There is not enough paired structures after {self.gen_rep_max_try} attempts.")
        current_candidates.extend(paired_structures)

        if len(paired_structures) < self.gen_rep_size:
            self.pfunc("There is not enough reproduced (paired) structures.")
            self.pfunc(f"Only {len(paired_structures)} are reproduced. The rest would be generated randomly.")
            cur_ran_size = self.gen_size - len(paired_structures) - self.gen_mut_size
        else:
            cur_ran_size = self.gen_ran_size

        # - random
        random_structures = []
        for i in range(self.gen_ran_max_try):
            frames = generator.run(size=1, soft_error=True)
            if frames:
                atoms = frames[0]
                self.pfunc("  reproduce randomly ")
                atoms.info["key_value_pairs"] = {}
                atoms.info["data"] = {}
                confid = database.c.write(atoms, origin="RandomCandidateUnrelaxed",
                    relaxed=0, extinct=0, generation=cur_gen, 
                    key_value_pairs=atoms.info["key_value_pairs"], 
                    data=atoms.info["data"]
                )
                database.c.update(confid, gaid=confid)
                atoms.info["confid"] = confid

                self.pfunc("  --> confid %d\n" %(atoms.info["confid"]))
                random_structures.append(atoms)
            if len(random_structures) == cur_ran_size:
                break
        else:
            if self.gen_ran_size > 0: # NOTE: no break when random size is 0
                self.pfunc(f"There is not enough random structures after {self.gen_ran_max_try} attempts.")
        current_candidates.extend(random_structures)

        if len(current_candidates) < (self.gen_rep_size+self.gen_ran_size):
            self.pfunc("There is not enough reproduced+random structures.")
            self.pfunc(f"Only {len(current_candidates)} are generated. The rest would be generated by mutations.")
            cur_mut_size = self.gen_size - len(current_candidates)
        else:
            cur_mut_size = self.gen_mut_size
        
        # - mutate
        mutated_structures = []
        for i in range(self.gen_mut_max_try):
            atoms = population.get_one_candidate(with_history=True)
            a3, desc = mutations.get_new_individual([atoms])
            if atoms is not None:
                database.add_unrelaxed_step(a3, desc)
                self.pfunc("  Mutate cand{} by {}".format(atoms.info["confid"], desc))
                self.pfunc("  --> confid %d\n" %(a3.info["confid"]))
                mutated_structures.append(a3)
            if len(mutated_structures) == cur_mut_size:
                break
        else:
            if self.gen_mut_size > 0: # NOTE: no break when random size is 0
                self.pfunc(f"There is not enough mutated structures after {self.gen_mut_max_try} attempts.")
        current_candidates.extend(mutated_structures)

        if len(current_candidates) != self.gen_size:
            self.pfunc("Not enough candidates for the next generation.")
            raise RuntimeError("Not enough candidates for the next generation.")

        return current_candidates
    
    def _update_generation_settings(self, population, mutations, pairing):
        """Update some generation-specific settings."""
        # - operations at the end of each generation
        cur_pop = population.get_current_population()
        #find_strain = False
        #from ase.ga.standardmutations import StrainMutation
        for mut in mutations.oplist:
            #if issubclass(mut, StrainMutation):
            #    find_strain = True
            #    mut.update_scaling_volume(cur_pop, w_adapt=0.5, n_adapt=0)
            #    self.pfunc(f"StrainMutation Scaling Volume: {mut.scaling_volume}")
            if hasattr(mut, "update_scaling_volume"):
                mut.update_scaling_volume(cur_pop, w_adapt=0.5, n_adapt=0)
                self.pfunc(f"{mut.__class__.__name__} Scaling Volume: {mut.scaling_volume}")
        if hasattr(pairing, "update_scaling_volume"):
            pairing.update_scaling_volume(cur_pop, w_adapt=0.5, n_adapt=0)
            self.pfunc(f"{pairing.__class__.__name__} Scaling Volume: {pairing.scaling_volume}")
        
        return
    
    def _reproduce(self, database, population, pairing, mutations) -> Atoms:
        """Reproduce a structure from the current population."""
        a3 = None
        for i in range(self.MAX_REPROC_TRY):
            # try 10 times
            parents = population.get_two_candidates()
            a3, desc = pairing.get_new_individual(parents) # NOTE: this also adds key_value_pairs to a.info
            if a3 is not None:
                database.add_unrelaxed_candidate(
                    a3, description=desc # here, desc is used to add "pairing": 1 to database
                ) # if mutation happens, it will not be relaxed

                mut_desc = ""
                if np.random.random() < self.pmut:
                    a3_mut, mut_desc = mutations.get_new_individual([a3])
                    #self.pfunc("a3_mut: ", a3_mut.info)
                    #self.pfunc("mut_desc: ", mut_desc)
                    if a3_mut is not None:
                        database.add_unrelaxed_step(a3_mut, mut_desc)
                        a3 = a3_mut
                self.pfunc(f"  reproduce offspring with {desc} \n  {mut_desc} after {i+1} attempts..." )
                break
            else:
                mut_desc = ""
                self.pfunc(f"  failed to reproduce offspring with {desc} \n  {mut_desc} after {i+1} attempts..." )
        else:
            self.pfunc("cannot reproduce offspring a3 after {0} attempts".format(self.MAX_REPROC_TRY))

        return a3


if __name__ == "__main__":
    pass