#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import pathlib
from typing import List, Optional, Union

import numpy as np
from ase import Atoms
from ase.io import read
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga.data import DataConnection

from .population import Population

#: Retained keys in key_value_pairs when get_atoms from the database.
RETAINED_KEYS: List[str] = ["extinct", "origin"]


def clean_seed_structures(prev_frames: List[Atoms]) -> List[Atoms]:
    """"""
    curr_frames = []
    energies, forces = [], []
    for i, prev_atoms in enumerate(prev_frames):
        # - copy geometry
        curr_atoms = Atoms(
            symbols=copy.deepcopy(prev_atoms.get_chemical_symbols()),
            positions=copy.deepcopy(prev_atoms.get_positions()),
            cell=copy.deepcopy(prev_atoms.get_cell(complete=True)),
            pbc=copy.deepcopy(prev_atoms.get_pbc()),
            tags=prev_atoms.get_tags(),  # retain this for molecules
        )
        # if prev_atoms.get_kinetic_energy() > 0.: # retain this for MD
        #    curr_atoms.set_momenta(prev_atoms.get_momenta())
        curr_frames.append(curr_atoms)

        # - save properties
        try:
            ene = prev_atoms.get_potential_energy()
            energies.append(ene)
        except:
            raise RuntimeError(f"Cannot get energy for seed structure {i}.")

        try:
            frc = prev_atoms.get_forces()
            forces.append(frc)
        except:
            raise RuntimeError(f"Cannot get forces for seed structure {i}.")

    for a, e, f in zip(curr_frames, energies, forces):
        calc = SinglePointCalculator(a, energy=e, forces=f)
        a.calc = calc

    return curr_frames


class AbstractPopulationManager:
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
                reprod: 20 # crossover + mutate
                random: 0
                mutate: 0

    """

    _print = print

    #: Reproduction and mutation.
    MAX_REPROC_TRY: int = 1

    #: Maximum attempts to generate new structures.
    MAX_ATTEMPTS_MULTIPLIER: int = 10

    def __init__(self, params: dict, rng=np.random.default_rng()) -> None:
        """"""
        self.rng = rng

        # - gen params
        gen_params = params.get("gen", dict(size=20, reproduce=20, random=0))
        self.gen_size = gen_params.get("size", None)
        assert isinstance(
            self.gen_size, int
        ), "size of generaton needs to be an integer."

        self.gen_ran_size = gen_params.get("random", 0)
        self.gen_ran_max_try = gen_params.get(
            "max_random_try", self.gen_ran_size * self.MAX_ATTEMPTS_MULTIPLIER
        )

        self.gen_rep_size = gen_params.get("reprod", self.gen_size - self.gen_ran_size)
        self.gen_rep_max_try = gen_params.get(
            "max_reprod_try", self.gen_rep_size * self.MAX_ATTEMPTS_MULTIPLIER
        )

        self.gen_mut_size = gen_params.get(
            "mutate", self.gen_size - self.gen_ran_size - self.gen_rep_size
        )

        # - init params
        init_params = params.get("init", dict(size=20, seed_file=None))
        self.init_size = init_params.get("size", None)
        self.init_seed_file: Optional[Union[str, pathlib.Path, List[Atoms]]] = (
            init_params.get("seed_file", None)
        )

        # - check if params were valid
        assert (
            self.gen_rep_size + self.gen_ran_size + self.gen_mut_size
        ) == self.gen_size, "In each generation, the sum of each component does not equal the total size."
        assert (
            self.gen_ran_size <= self.gen_size
        ), "In each generation, the random size should not be larger than the total size."
        assert (
            self.gen_rep_size <= self.gen_size
        ), "In each generation, the reprod size should not be larger than the total size."
        assert (
            self.gen_mut_size <= self.gen_size
        ), "In each generation, the mutate size should not be larger than the total size."

        # - mutation
        self.pmut = params.get("pmut", 0.5)
        self.pmut_custom = params.get("params", 0.5)

        return

    def _get_current_candidates(self, database: DataConnection, curr_gen: int):
        """Get offsprings in the current generation.

        Mutataed candidates do not have `generation` keyword.

        Args:
            database: DataConnection.
            curr_gen: The current generation number.

        """
        candidate_groups = {"paired": [], "random": [], "mutated": []}
        num_paired, num_mutated, num_random = 0, 0, 0

        # unrelaxed_strus_gen_ = list(database.c.select(f"relaxed=0"))
        unrelaxed_strus_gen_ = list(
            database.c.select(f"relaxed=0,generation={curr_gen}")
        )
        for row in unrelaxed_strus_gen_:
            if row.formula:
                # print(row["gaid"], row)
                confid = row["gaid"]
                curr_rows = sorted(
                    database.c.select(f"relaxed=0,gaid={confid}"), key=lambda x: x.mtime
                )
                curr_rows = [x for x in curr_rows if x.formula]
                # - get atoms
                curr_atoms = database.get_atoms(curr_rows[-1].id, add_info=True)
                # NOTE: candidates should not have description info...
                #       otherwise, queued row also has them and failed in
                #       database.c.get_participation_in_pairing()
                kvp = {
                    k: v
                    for k, v in curr_atoms.info["key_value_pairs"].items()
                    if k in RETAINED_KEYS
                }
                data = curr_atoms.info.get(
                    "data", {}
                )  # not every cand has data that stores parents
                curr_atoms.info = {
                    "key_value_pairs": kvp,
                    "data": data,
                    "confid": confid,
                }
                # print(curr_atoms)
                # - count and add atoms
                if "Pairing" in curr_rows[0]["origin"]:
                    num_paired += 1
                    candidate_groups["paired"].append(curr_atoms)
                elif "Mutation" in curr_rows[0]["origin"]:
                    num_mutated += 1
                    candidate_groups["mutated"].append(curr_atoms)
                elif "Random" in curr_rows[0]["origin"]:
                    num_random += 1
                    candidate_groups["random"].append(curr_atoms)
                else:
                    ...

        return candidate_groups, num_paired, num_mutated, num_random

    def _prepare_initial_population(self, generator) -> List[Atoms]:
        self._print("===== Prepare Initial Population =====")
        starting_population = []

        # - try to read seed structures
        # NOTE: seed structures would be re-optimised by the worker
        self._print("----- try to add seed structures -----")
        seed_frames = []
        if self.init_seed_file is not None:
            self._print(str(self.init_seed_file))
            if isinstance(self.init_seed_file, str):
                seed_frames = read(self.init_seed_file, ":")
            elif isinstance(self.init_seed_file, pathlib.Path):
                seed_frames = read(self.init_seed_file, ":")
            elif isinstance(self.init_seed_file, list):  # List[Atoms]
                seed_frames = self.init_seed_file
            else:
                raise RuntimeError(
                    f"Init_seed_file {self.init_seed_file} formst is unsuppoted."
                )
            seed_frames = clean_seed_structures(seed_frames)
            seed_size = len(seed_frames)
            self._print(f"number of seed frames: {seed_size}")
            assert (
                seed_size > 0 and seed_size <= self.init_size
            ), "The number of seed structures is invalid."
        else:
            seed_size = 0

        # TODO: seed structures will be calculated again??
        # TODO: check atom permutation
        for i, atoms in enumerate(seed_frames):
            # TODO: check atom order
            atoms.info["data"] = {}
            atoms.info["key_value_pairs"] = {}
            atoms.info["key_value_pairs"]["origin"] = "seed {}".format(i)
            atoms.info["key_value_pairs"]["raw_score"] = -atoms.get_potential_energy()
            # TODO: check geometric convergence
        self._print(f"number of seed structures: {len(seed_frames)}")
        starting_population.extend(seed_frames)

        # - generate random structures
        self._print("----- try to generate random structures -----")
        random_frames = generator.run(size=self.init_size - seed_size)
        self._print(f"number of random structures: {len(random_frames)}")
        starting_population.extend(random_frames)

        if len(starting_population) != self.init_size:
            raise RuntimeError(
                "It fails to generate the initial population. Check the seed file and the system setting."
            )

        self._print(f"finished creating initial population...")

        return starting_population

    def _prepare_current_population(
        self,
        database: DataConnection,
        curr_gen: int,
        population: Population,
        generator,
        operators: dict,
        candidate_groups: dict = {},
        num_paired: int = 0,
        num_mutated: int = 0,
        num_random: int = 0,
    ) -> List[Atoms]:
        """Prepare current population.

        Usually, it should be the same as the initial size. However, for variable
        composition search, a large init size can be useful.

        Args:
            database: database
            curr_gen: current generation
            population: current population
            generator: generator
            pairing: pairing
            mutations: mutations
            candidate_groups: candidate groups
            num_paired: number of paired
            num_mutated: number of mutated
            num_random: number of random

        Returns:
            A list of Atoms.

        """
        current_candidates = []

        # We need adjust n_top for the variable composition search.
        slab = database.get_slab()
        num_atoms_substrate = len(slab)

        # - reproduction and then mutation
        rest_rep_size = self.gen_rep_size - num_paired
        paired_structures = []
        paired_structures.extend(candidate_groups.get("paired", []))
        if rest_rep_size > 0 and num_random == 0:
            # pair finished but not enough, random already starts...
            for i in range(self.gen_rep_max_try):
                self._print(f"Reproduction attempt {i} ->")
                atoms = self._reproduce(database, curr_gen, population, operators, num_atoms_substrate)
                if atoms is not None:
                    paired_structures.append(atoms)
                    parents = " ".join([str(x) for x in atoms.info["data"]["parents"]])
                    self._print(
                        f"  confid={atoms.info['confid']:>6d} parents={parents:<14s} origin={atoms.info['key_value_pairs']['origin']:<20s} extinct={atoms.info['key_value_pairs']['extinct']:<4d}"
                    )
                else:
                    ...  # Reproduction failed.
                if len(paired_structures) == self.gen_rep_size:
                    break
            else:
                self._print(
                    f"There is not enough paired structures after {self.gen_rep_max_try} attempts."
                )
        else:
            ...
        current_candidates.extend(paired_structures)

        # - random
        if len(paired_structures) < self.gen_rep_size:
            self._print("There is not enough reproduced (paired) structures.")
            self._print(
                f"Only {len(paired_structures)} are reproduced. The rest would be generated randomly."
            )
            curr_ran_size = self.gen_size - len(paired_structures) - self.gen_mut_size
        else:
            curr_ran_size = self.gen_ran_size

        rest_ran_size = curr_ran_size - num_random
        gen_ran_max_try = rest_ran_size * self.MAX_ATTEMPTS_MULTIPLIER

        random_structures = []
        random_structures.extend(candidate_groups.get("random", []))
        if rest_ran_size > 0 and num_mutated == 0:
            # random finished but not enough, mutation already starts...
            for i in range(gen_ran_max_try):
                self._print(f"Random attempt {i} ->")
                frames = generator.run(size=1, soft_error=True)
                if frames:
                    atoms = frames[0]
                    atoms.info["key_value_pairs"] = {
                        "extinct": 0,
                        "origin": "RandomCandidateUnrelaxed",
                    }
                    atoms.info["data"] = {}
                    confid = database.c.write(
                        atoms,
                        relaxed=0,
                        extinct=0,
                        random=1,
                        description="random",
                        generation=curr_gen,
                        key_value_pairs=atoms.info["key_value_pairs"],
                        data=atoms.info["data"],
                    )
                    database.c.update(confid, gaid=confid)
                    atoms.info["confid"] = confid

                    random_structures.append(atoms)
                    self._print(
                        f"  confid={atoms.info['confid']:>6d} parents={'none':<14s} origin={atoms.info['key_value_pairs']['origin']:<20s} extinct={atoms.info['key_value_pairs']['extinct']:<4d}"
                    )
                else:
                    ...  # Random failed.
                if len(random_structures) == curr_ran_size:
                    break
            else:
                if self.gen_ran_size > 0:  # NOTE: no break when random size is 0
                    self._print(
                        f"There is not enough random structures after {self.gen_ran_max_try} attempts."
                    )
        else:
            ...
        current_candidates.extend(random_structures)

        # - mutate
        if len(current_candidates) < (self.gen_rep_size + self.gen_ran_size):
            self._print("There is not enough reproduced+random structures.")
            self._print(
                f"Only {len(current_candidates)} are generated. The rest would be generated by mutations."
            )
            curr_mut_size = self.gen_size - len(current_candidates)
        else:
            curr_mut_size = self.gen_mut_size

        rest_mut_size = curr_mut_size - num_mutated
        gen_mut_max_try = rest_mut_size * self.MAX_ATTEMPTS_MULTIPLIER

        mutated_structures = []
        mutated_structures.extend(candidate_groups.get("mutated", []))
        for i in range(gen_mut_max_try):
            self._print(f"Mutation attempt {i} ->")
            parent = population.get_one_candidate(with_history=True)
            atoms, desc = operators["mobile"]["mutations"].get_new_individual([parent])
            if atoms is not None:
                t, desc = desc.split(":")
                atoms.info["key_value_pairs"]["generation"]= curr_gen
                atoms.info["data"] = {
                    "parents": [parent.info["confid"]]
                }
                confid = database.c.write(
                    atoms,
                    relaxed=0,
                    extinct=0,
                    mutation=1,
                    description=desc,
                    generation=curr_gen,
                    key_value_pairs=atoms.info["key_value_pairs"],
                    data=atoms.info["data"],
                )
                database.c.update(confid, gaid=confid)
                atoms.info["confid"] = confid

                mutated_structures.append(atoms)

                parents = " ".join([str(x) for x in atoms.info["data"]["parents"]])
                self._print(
                    f"  confid={atoms.info['confid']:>6d} parents={parents:<14s} origin={atoms.info['key_value_pairs']['origin']:<20s} extinct={atoms.info['key_value_pairs']['extinct']:<4d}"
                )
            else:
                ...  # mutation failed...
            if len(mutated_structures) == curr_mut_size:
                break
        else:
            if self.gen_mut_size > 0:  # NOTE: no break when random size is 0
                self._print(
                    f"There is not enough mutated structures after {gen_mut_max_try} attempts."
                )
        current_candidates.extend(mutated_structures)

        if len(current_candidates) != self.gen_size:
            self._print("Not enough candidates for the next generation.")
            raise RuntimeError("Not enough candidates for the next generation.")

        return current_candidates

    def _update_generation_settings(self, population, mutations, pairing):
        """Update some generation-specific settings."""
        # - operations at the end of each generation
        cur_pop = population.get_current_population()
        # find_strain = False
        # from ase.ga.standardmutations import StrainMutation
        for mut in mutations.oplist:
            # if issubclass(mut, StrainMutation):
            #    find_strain = True
            #    mut.update_scaling_volume(cur_pop, w_adapt=0.5, n_adapt=0)
            #    self._print(f"StrainMutation Scaling Volume: {mut.scaling_volume}")
            if hasattr(mut, "update_scaling_volume"):
                mut.update_scaling_volume(cur_pop, w_adapt=0.5, n_adapt=0)
                self._print(
                    f"{mut.__class__.__name__} Scaling Volume: {mut.scaling_volume}"
                )
        if hasattr(pairing, "update_scaling_volume"):
            pairing.update_scaling_volume(cur_pop, w_adapt=0.5, n_adapt=0)
            self._print(
                f"{pairing.__class__.__name__} Scaling Volume: {pairing.scaling_volume}"
            )

        return

    def _reproduce(
        self, database: DataConnection, curr_gen: int, population, operators: dict, num_atoms_substrate: int
    ) -> Optional[Atoms]:
        """Reproduce a structure from the current population.

        Args:
            curr_gen: The current generation number.

        Returns:
            An atoms.

        """
        pairing = operators["mobile"]["pairing"]
        mutations = operators["mobile"]["mutations"]

        custom_mutations = None
        if operators.get("custom", None) is not None:
            custom_mutations = operators["custom"]["mutations"]

        if hasattr(pairing , "n_top"):
            prev_ntop = pairing.n_top
        else:
            prev_ntop = None  # HACK: In the end, it should not None.

        a3 = None
        for _ in range(self.MAX_REPROC_TRY):
            # Get two parents.
            if pairing.allow_variable_composition:
                parents = population.get_two_candidates()
                natoms_p0, natoms_p1 = len(parents[0]), len(parents[1])
                self._print(f"  p0_natoms: {natoms_p0} p1_natoms: {natoms_p1}")
            else:
                # TODO: If there is no two structures with the number of atoms?
                for _ in range(100):
                    parents = population.get_two_candidates()
                    natoms_p0, natoms_p1 = len(parents[0]), len(parents[1])
                    if natoms_p0 == natoms_p1:
                        self._print(f"  p0_natoms: {natoms_p0} p1_natoms: {natoms_p1}")
                        break
                else:
                    raise RuntimeError(
                        "Different number of atoms in the two parents after 100 attempts."
                    )
            # We need adjust n_top of some operators for comptability.
            curr_ntop = natoms_p0 - num_atoms_substrate
            if hasattr(pairing, "n_top"):
                prev_ntop = pairing.n_top
                self._print(f"  pairing  {prev_ntop =} -> {curr_ntop =}")
                pairing.n_top = curr_ntop
                assert natoms_p0 == pairing.n_top + num_atoms_substrate
            else:
                prev_ntop = curr_ntop
            # Perform the crossover and the mutations.
            a3, desc = pairing.get_new_individual(
                parents
            )  # This also adds key_value_pairs to a.info
            if a3 is not None:
                # We need update curr_ntop as a variable crossover may be performed.
                curr_ntop = len(a3) - num_atoms_substrate

                a3.info["key_value_pairs"]["generation"] = curr_gen
                database.add_unrelaxed_candidate(
                    a3,
                    description=desc,  # here, desc is used to add "pairing": 1 to database
                )  # if mutation happens, it will not be relaxed
                self._print(f"  confid= {a3.info['confid']} ")

                # mutate atoms in the mobile group
                for mutation in mutations.oplist:
                    if hasattr(mutation, "n_top"):
                        self._print(f"  mutation  {mutation.n_top =} -> {curr_ntop =}")
                        mutation.n_top = curr_ntop
                        assert natoms_p0 == mutation.n_top + num_atoms_substrate

                curr_prob = self.rng.random()
                if curr_prob < self.pmut:
                    a3_mut, mut_desc = mutations.get_new_individual([a3])
                    if a3_mut is not None:
                        database.add_unrelaxed_step(a3_mut, mut_desc)
                        a3 = a3_mut
                        self._print(f"  mobile: {desc}  {mut_desc}")
                    else:
                        self._print(f"  mobile: {desc}")  # Mutate failed.
                else:
                    self._print(f"  mobile: {desc}")  # No mutation is applied.

                # mutate atoms in the custom group
                if custom_mutations is not None:
                    curr_prob = self.rng.random()
                    if curr_prob < self.pmut_custom:
                        a3_bmut, bmut_desc = custom_mutations.get_new_individual([a3])
                        if a3_bmut is not None:
                            database.add_unrelaxed_step(a3_bmut, bmut_desc)
                            a3 = a3_bmut
                            self._print(f"  custom: {bmut_desc}")
                        else:
                            ...
                    else:
                        ...
                else:
                    ...

                break
            else:
                ...  # Reproduce failed.
        else:
            self._print(
                f"  cannot reproduce offspring a3 after {self.MAX_REPROC_TRY} attempts"
            )

        # restorre n_top, custom mutations should not have n_top...
        if hasattr(pairing, "n_top"):
            pairing.n_top = prev_ntop
        for mutation in mutations.oplist:
            if hasattr(mutation, "n_top"):
                mutation.n_top = prev_ntop

        return a3


if __name__ == "__main__":
    ...
