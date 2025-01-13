#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import collections
import copy
import inspect
import itertools
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.ga.data import DataConnection, PrepareDB
from ase.ga.offspring_creator import OperationSelector
from ase.io import read, write

from .. import convert_indices, get_tags_per_species, registers
from ..expedition import AbstractExpedition
from .population.manager import AbstractPopulationManager
from .population.population import Population, PopulationWithVariableComposition

"""
Workflow
    check current calculation
        |
    check population and generate offspring
        |
    submit unrelaxed structures

Systems
    bulk
    slab
    cluster

Reserved Keywords in Database
    generation
    relaxed
    queued
    extinct
    description
    pairing

Operators
    comparator
    crossover (pairing)
    mutation

"""


def get_generation_number(da: DataConnection) -> int:
    """Check the number of generation based on the number of relaxed candidates.

    The population size of the first generation can be different from the following ones.

    Args:
        da: The ga data connection.

    Returns:
        The generation number.

    """
    init_pop_size: int = da.get_param("initial_population_size")  # type: ignore
    pop_size: int = da.get_param("population_size")  # type: ignore

    all_candidates = list(da.c.select(relaxed=1))
    counter = collections.Counter([c.generation for c in all_candidates])
    generations = sorted(list(counter.keys()))
    num_generations = len(generations)
    if num_generations == 0:
        curr_gen = 0
    else:
        if num_generations == 1:
            if counter[0] < init_pop_size:
                curr_gen = 0
            else:
                assert counter[0] == init_pop_size
                curr_gen = 1
        else:
            curr_gen = max(generations)
            if counter[curr_gen] < pop_size:
                ...
            else:
                assert counter[curr_gen] == pop_size
                curr_gen += 1

    return curr_gen


def plot_evolution_figure(rdir, data, gen_num, target):
    """"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))
    ax.set_title("Population Evolution")  # type: ignore
    for i, properties in data:
        ax.scatter([i] * len(properties), properties, alpha=0.5)  # type: ignore
    ax.set(xlabel="generation", xticks=range(gen_num), ylabel=target)  # type: ignore
    fig.savefig(rdir / "pop.png", bbox_inches="tight")
    plt.close()

    return


class GeneticAlgorithmBroadcaster:
    """Broadcast genetic_algorithm_engine by parameters."""

    def __init__(self, builder, params, worker=None, random_seed=None):
        """"""
        new_params_list = self._broadcast_parameters(params)

        input_params_list = []
        for new_params in new_params_list:
            input_params = dict(
                builder=copy.deepcopy(builder),
                worker=copy.deepcopy(worker),
                params=new_params,
                random_seed=copy.deepcopy(random_seed),
            )
            input_params_list.append(input_params)
        self.input_params_list = input_params_list

        return

    def __iter__(self):
        """"""
        for input_params in self.input_params_list:
            engine = GeneticAlgorithmEngine(**input_params)
            yield engine

    def _broadcast_parameters(self, params):
        """Broadcast input parameters that can form several engines.

        Note:
            Currently, we only support `chempot` in formation_energy optimisation.

        """
        new_params_list = []

        property_setting = params.get("property", dict(target="energy"))
        target = property_setting.get("target", "energy")
        if target == "energy":
            new_params = copy.deepcopy(params)
            new_params_list.append(new_params)
        elif target == "formation_energy":
            chempot = []
            for k, v in property_setting.get("chempot").items():
                if isinstance(v, list):
                    chempot.append([(k, v_i) for v_i in v])
                else:  # This must be a number.
                    chempot.append([(k, v)])
            broadcasted_chempots = list(itertools.product(*chempot))
            for chempot in broadcasted_chempots:
                new_params = copy.deepcopy(params)
                new_params["property"]["chempot"] = {k: v for k, v in chempot}
                new_params_list.append(new_params)
        else:
            ...

        return new_params_list


class GeneticAlgorithmEngine(AbstractExpedition):
    """Genetic Algorithem Engine."""

    _directory = pathlib.Path.cwd()

    # local optimisation directory
    CALC_DIRNAME = "tmp_folder"

    #: Prefix of each generation's directory.
    GEN_PREFIX: str = "gen"

    # TODO: Neighbor list and parametrization parameters to screen
    # candidates before relaxation can be added. Default is not to use.
    find_neighbors = None
    perform_parametrization = None

    def __init__(
        self,
        builder: dict,
        params: dict,
        directroy="./",
        random_seed=None,
        *args,
        **kwargs,
    ):
        """Initialise engine.

        Args:
            builder: Define the system to explore.

        """
        ga_dict = params  # For compat

        # --- database ---
        self.db_name = ga_dict.get("database", "mydb.db")

        # -
        self.directory = directroy

        # Store initial parameters
        self.ga_dict = copy.deepcopy(ga_dict)

        population_params = self.ga_dict.get("population", None)
        if population_params is not None:
            if "init" in population_params:
                seed_file = population_params["init"].get("seed_file", None)
                if seed_file is not None:
                    self.ga_dict["population"]["init"]["seed_file"] = str(
                        pathlib.Path(seed_file).resolve()
                    )

        # Check random consistency, generator and population
        if random_seed is None:
            random_seed = np.random.randint(0, 1e8)  # type: ignore
        self.random_seed = random_seed
        self._print(f"GA RANDOM SEED {random_seed}")
        self.rng = np.random.Generator(np.random.PCG64(seed=random_seed))

        # Check builder for random structure generation
        if isinstance(builder, dict):
            # -- generator will reset np.random by the given random_seed
            builder_params = copy.deepcopy(builder)
            builder_method = builder_params.pop("method")
            prev_seed = builder_params.get("random_seed")
            builder_params.update(random_seed=random_seed)
            self.generator = registers.create(
                "builder", builder_method, convert_name=False, **builder_params
            )
        else:
            # NOTE: randomBuilder uses deprecated np.random
            #       if multi engines are running at the same time
            #       the builders will intervine each other on random!!!
            prev_seed = builder.random_seed
            self.generator = builder
            np.random.seed(random_seed)
            if (self.generator, "rng"):
                self.generator.rng = self.rng
        self._print(f"OVERWRITE BUILDER SEED FROM {prev_seed} TO {random_seed}")

        # - worker info
        self.worker = None

        # --- population ---
        self.pop_manager = AbstractPopulationManager(
            ga_dict["population"], rng=self.rng
        )

        # Sanity check on target property
        self.prop_dict = ga_dict.get("property", dict(target="energy"))
        target = self.prop_dict.get("target", None)
        assert target in [
            "energy",
            "formation_energy",
        ], f"Target `{target}` is not supported yet."
        if target == "formation_energy":
            if "chempot" not in self.prop_dict:
                raise RuntimeError(
                    "The `chempot` is not provided in the property section."
                )
        else:
            ...

        # The ase built-in cut_and_splice reinits tags from 0 if use_tags is false,
        # Here, no matter what type of system is explored, we enforce use_tags to be true
        # as it retains the tags information.
        if hasattr(self.generator, "use_tags"):
            if self.generator.use_tags:
                ...
            else:
                self.generator.use_tags = True
                self._print(
                    f"Builder `{self.generator.name}` changes `use_tags` to true for formation energy computation."
                )
        else:
            raise RuntimeError(
                f"Builder `{self.generator.name}` does not have true `use_tags`."
            )

        self.target = target

        # --- convergence ---
        self.conv_dict = ga_dict["convergence"]

        # - misc
        self.use_archive = ga_dict.get("use_archive", True)

        return

    @property
    def directory(self):
        return self._directory

    @directory.setter
    def directory(self, directory_):
        self._directory = pathlib.Path(directory_)
        self.db_path = self._directory / self.db_name
        return

    def report(self):
        """Write reports of this GA search.

        One file contains all relaxed structures and one figure shows the target properties
        in each generation.

        """
        self._print("restart the database...")
        self.da = DataConnection(self.db_path)
        results = self.directory / "results"
        if not results.exists():
            results.mkdir()

        # write structures that are already sorted by raw_score
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        write(results / "all_candidates.xyz", all_relaxed_candidates)

        # plot population evolution
        data = []
        gen_num = get_generation_number(self.da)  # equals finished generation plus one
        self._print(f"Genetic Algorithm Statistics with {gen_num-1} generations: ")
        for i in range(gen_num):
            current_candidates = [
                atoms
                for atoms in all_relaxed_candidates
                if atoms.info["key_value_pairs"]["generation"] == i
            ]
            properties = np.array(
                [a.info["key_value_pairs"]["target"] for a in current_candidates]
            )
            stats = dict(
                min=np.min(properties),
                max=np.max(properties),
                avg=np.mean(properties),
                std=np.std(properties),
            )
            self._print(
                f"num {properties.shape[0]:>4d} min {stats['min']:>12.4f} max {stats['max']:>12.4f} avg {stats['avg']:>12.4f} std {stats['std']:>12.4f}"
            )
            data.append([i, properties])

        plot_evolution_figure(results, data, gen_num, self.target)

        return

    def update_active_params(self, prev_wdir, *args, **kwargs):
        """"""
        candidates = read(prev_wdir / "results" / "all_candidates.xyz", ":")

        init_size = self.pop_manager.init_size
        self.pop_manager.init_seed_file = candidates[:init_size]

        return

    def run(self, *args, **kwargs):
        """Run the GA procedure several steps.

        Default setting would run the algorithm many times until its convergence.
        This is useful for running optimisations with serial worker.

        """
        # Search target
        self._print(f"===== Genetic Algorithm =====")
        self._print(f"Target of Global Optimisation is {self.target}")

        # Check worker
        self._print("===== register worker =====")
        assert self.worker is not None, "GA has not set its worker properly."
        self.worker.directory = self.directory / self.CALC_DIRNAME

        # - outputs
        self.pop_manager._print = self._print

        # - generator info
        self._print("===== register builder =====")
        for l in str(self.generator).split("\n"):
            self._print(l)
        self._print(f"random_state: f{self.generator.random_seed}")

        # TODO: move this part to where before generator is created
        # HACK: As the substrate is lazy-evaluated, it is unknown until
        #       generator.run() is called. The unknwon substrate will
        #       cause the crossover giving inconsistent pbc.
        #       Thus, we initialise a substrate by default in generator's setting.
        try:
            self.generator._update_settings()
        except:
            ...

        # NOTE: check database existence and generation number to determine restart
        self._print("===== register database =====")
        self._debug(f"database path: {str(self.db_path)}")
        if not self.db_path.exists():
            self._print("create a new database...")
            self._create_initial_population()
        else:
            self._print("restart the database...")
            self.da = DataConnection(self.db_path)

        num_atoms_substrate = self.da.get_param("num_atoms_substrate")
        self._print(f"{num_atoms_substrate=}")

        # --- mutation and comparassion operators
        self._print("===== register operators =====")
        self._register_operators()

        # - run
        for _ in range(1000):
            self._check_generation()
            if self.read_convergence():
                self._print("reach maximum generation...")
                self.report()
                break
            curr_convergence = self._irun()
            if not curr_convergence:
                self._print("current generation does not converge...")
                break

        return

    def _check_generation(self):
        """Check the generation status."""
        # self._print(f"{self.cur_gen =}")

        self.cur_gen = get_generation_number(self.da)

        unrelaxed_strus_gen_ = list(
            self.da.c.select("relaxed=0,generation=%d" % self.cur_gen)
        )
        unrelaxed_strus_gen = []
        for row in unrelaxed_strus_gen_:
            # NOTE: mark_as_queue unrelaxed_candidate will have relaxed field too...
            if "queued" not in row:
                unrelaxed_strus_gen.append(row)
        self.unrelaxed_confids = [row["gaid"] for row in unrelaxed_strus_gen]
        self.num_unrelaxed_gen = len(self.unrelaxed_confids)

        relaxed_strus_gen = list(
            self.da.c.select("relaxed=1,generation=%d" % self.cur_gen)
        )
        for row in relaxed_strus_gen:
            self._debug(row)
        self.relaxed_confids = [row["gaid"] for row in relaxed_strus_gen]
        self.num_relaxed_gen = len(self.relaxed_confids)

        # check if this is the begin or the end of the current generation
        self.beg_of_gen = (self.num_relaxed_gen == self.num_unrelaxed_gen) and (
            self.num_relaxed_gen == 0
        )
        self.end_of_gen = (self.num_relaxed_gen == self.num_unrelaxed_gen) and (
            self.num_relaxed_gen != 0
        )

        return

    def _irun(self):
        """main procedure"""
        if not hasattr(self, "cur_gen"):
            raise RuntimeError(
                "The current genertion is unknown. Check generation before."
            )
        # - generation
        self._print("===== Generation Info =====")
        self._print(f"current generation number: {self.cur_gen}")
        self._print(f"number of relaxed in current generation: {self.num_relaxed_gen}")
        self._print(convert_indices(sorted(self.relaxed_confids)))
        self._print(
            f"number of unrelaxed in current generation: {self.num_unrelaxed_gen}"
        )
        self._print(convert_indices(sorted(self.unrelaxed_confids)))
        self._print(f"end of current generation: {self.end_of_gen}")

        # - population
        self._print("===== Population Info =====")
        content = "For generation > 0,\n"
        content += "{:>8s}  {:>8s}  {:>8s}  {:>8s}\n".format(
            "Reprod", "Random", "Mutate", "Total"
        )
        content += "{:>8d}  {:>8d}  {:>8d}  {:>8d}\n".format(
            self.pop_manager.gen_rep_size,
            self.pop_manager.gen_ran_size,
            self.pop_manager.gen_mut_size,
            self.pop_manager.gen_size,
        )
        content += "Note: Reproduced structure has a chance (pmut) to mutate.\n"
        for l in content.split("\n"):
            self._print(l)

        # - minimise
        if self.cur_gen == 0:
            self._print("===== Initial Population Calculation =====")
            frames_to_work = []
            while (
                self.da.get_number_of_unrelaxed_candidates()
            ):  # NOTE: this uses GADB get_atoms which adds extra_info
                # calculate structures from init population
                atoms = self.da.get_an_unrelaxed_candidate()
                frames_to_work.append(atoms)
                self.da.mark_as_queued(atoms)  # this marks relaxation is in the queue
            confids = [a.info["confid"] for a in frames_to_work]
            self._print(f"start to run structure {convert_indices(confids)}")
            # NOTE: provide unified interface to mlp and dft
            if frames_to_work:
                self.worker.directory = (
                    self.directory / self.CALC_DIRNAME / f"gen{self.cur_gen}"
                )
                _ = self.worker.run(frames_to_work)  # retrieve later
        else:
            # --- update population
            self._print("===== Update Population =====")
            # - create the population used for crossover and mutation
            candidate_groups, num_paired, num_mutated, num_random = (
                self.pop_manager._get_current_candidates(
                    database=self.da, curr_gen=self.cur_gen
                )
            )
            # candidate_groups, num_paired, num_mutated, num_random = self.pop_manager._get_current_candidates(
            #    database=self.da, curr_gen=self.cur_gen-1
            # )
            # for a in candidate_groups["paired"]:
            #    print(a.info)

            # TODO: random seed...
            if self.pop_manager.name == "constant":
                current_population = Population(
                    data_connection=self.da,
                    population_size=self.pop_manager.gen_size,
                    comparator=self.operators["mobile"]["comparing"],
                    rng=self.rng,
                )
            else:
                assert self.pop_manager.name == "variable"
                current_population = PopulationWithVariableComposition(
                    data_connection=self.da,
                    population_size=self.pop_manager.gen_size,
                    comparator=self.operators["mobile"]["comparing"],
                    rng=self.rng,
                )
                for tribe in current_population.tribes:
                    self._print(f"tribe: {tribe[0]} number: {len(tribe[1])}")

            pop_confids = [a.info["confid"] for a in current_population.pop]
            self._print(f"number of structures in population: {len(pop_confids)}")
            self._print(f"confids in population: {convert_indices(pop_confids)}")

            self.pop_manager._update_generation_settings(
                current_population,
                self.operators["mobile"]["mutations"],
                self.operators["mobile"]["pairing"],
            )

            # ----
            current_candidates = []
            if self.beg_of_gen:  # (num_relaxed == num_unrelaxed == 0)
                current_candidates = self.pop_manager._prepare_current_population(
                    database=self.da,
                    curr_gen=self.cur_gen,
                    population=current_population,
                    generator=self.generator,
                    operators=self.operators,
                )
            else:
                self._print("Current generation has not finished...")
                # NOTE: The current candidates have not been created completely.
                #       For example, num_relaxed != num_unrelaxed,
                #       need create more candidates...
                if self.num_relaxed_gen == 0 and (
                    self.num_unrelaxed_gen < self.pop_manager.gen_size
                ):
                    current_candidates = self.pop_manager._prepare_current_population(
                        database=self.da,
                        curr_gen=self.cur_gen,
                        population=current_population,
                        generator=self.generator,
                        operators=self.operators,
                        candidate_groups=candidate_groups,
                        num_paired=num_paired,
                        num_mutated=num_mutated,
                        num_random=num_random,
                    )
                elif self.num_relaxed_gen == 0 and (
                    self.num_unrelaxed_gen == self.pop_manager.gen_size
                ):
                    # no relaxed, and finished creation, num_relaxed == gen_size?
                    current_candidates = self.pop_manager._prepare_current_population(
                        database=self.da,
                        curr_gen=self.cur_gen,
                        population=current_population,
                        generator=self.generator,
                        operators=self.operators,
                        candidate_groups=candidate_groups,
                        num_paired=num_paired,
                        num_mutated=num_mutated,
                        num_random=num_random,
                    )
                else:
                    ...

            # -- validate current candidates
            # candidate_groups, num_paired, num_mutated, num_random = self.pop_manager._get_current_candidates(
            #    database=self.da, curr_gen=self.cur_gen
            # )
            # for ia, a in enumerate(candidate_groups["paired"]):
            #    self._print(f"{ia} {a.info}")

            # TODO: send candidates directly to worker that respects the batchsize
            self._print("===== Optimisation =====")
            for ia, a in enumerate(current_candidates):
                parents = "none"
                if "parents" in a.info["data"]:
                    parents = " ".join([str(x) for x in a.info["data"]["parents"]])
                self._print(
                    f"{ia:>4d} confid={a.info['confid']:>6d} parents={parents:<14s} origin={a.info['key_value_pairs']['origin']:<32s} extinct={a.info['key_value_pairs']['extinct']:<4d}"
                )
            if not (self.directory / self.CALC_DIRNAME / f"gen{self.cur_gen}").exists():
                frames_to_work = []
                for atoms in current_candidates:
                    frames_to_work.append(atoms)
                    self.da.mark_as_queued(
                        atoms
                    )  # this marks relaxation is in the queue
                if frames_to_work:
                    confids = [a.info["confid"] for a in frames_to_work]
                    self._print(f"start to run structure {convert_indices(confids)}")
                    self.worker.directory = (
                        self.directory / self.CALC_DIRNAME / f"gen{self.cur_gen}"
                    )
                    _ = self.worker.run(frames_to_work)  # retrieve later
            else:
                self._print(
                    f"calculation directory for generation {self.cur_gen} exists."
                )

        # --- check if there were finished jobs
        curr_convergence = False
        self.worker.directory = (
            self.directory / self.CALC_DIRNAME / f"gen{self.cur_gen}"
        )
        self.worker.inspect(resubmit=True)
        if self.worker.get_number_of_running_jobs() == 0:
            self._print("===== Retrieve Relaxed Population =====")
            converged_candidates = [
                t[-1] for t in self.worker.retrieve(use_archive=self.use_archive)
            ]
            for cand in converged_candidates:
                # update extra info
                extra_info = dict(
                    data={}, key_value_pairs={"generation": self.cur_gen, "extinct": 0}
                )
                cand.info.update(extra_info)
                # get tags
                confid = cand.info["confid"]
                if self.generator.use_tags:
                    rows = list(self.da.c.select(f"relaxed=0,gaid={confid}"))
                    rows = sorted(
                        [row for row in rows if row.formula], key=lambda row: row.mtime
                    )
                    if len(rows) > 0:
                        previous_atoms = rows[-1].toatoms(
                            add_additional_information=True
                        )
                        previous_tags = previous_atoms.get_tags()
                    else:
                        raise RuntimeError(f"Cannot find tags for candidate {confid}")
                    cand.set_tags(previous_tags)
                    identities = get_tags_per_species(cand)
                    identity_stats = {}
                    for k, v in identities.items():
                        identity_stats[k] = len(v)
                    cand.info["identity_stats"] = identity_stats
                else:
                    ...
                # evaluate raw score
                self.evaluate_candidate(cand)
                fitness = cand.info["key_value_pairs"]["raw_score"]
                self._print(
                    f"confid {confid:<6d} relaxed with fitness {fitness:>16.4f}"
                )
                if "identity_stats" in cand.info:
                    identity_info = "  " + " ".join(
                        [f"{k}: {v}" for k, v in cand.info["identity_stats"].items()]
                    )
                    self._print(identity_info)
                self.da.add_relaxed_step(
                    cand,
                    find_neighbors=self.find_neighbors,
                    perform_parametrization=self.perform_parametrization,
                )
            curr_convergence = True
        else:
            self._print("Worker is unfinished.")

        return curr_convergence

    def get_workers(self):
        """Get all workers used by this expedition."""
        if not hasattr(self, "da"):
            self.da = DataConnection(self.db_path)
            self._check_generation()

        num_gen = self.cur_gen
        if self.end_of_gen:
            num_gen += 1

        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()

        workers = []
        for i in range(num_gen):
            curr_worker = copy.deepcopy(self.worker)
            curr_worker.directory = (
                self.directory / self.CALC_DIRNAME / (f"{self.GEN_PREFIX}{i}")
            )
            workers.append(curr_worker)

        return workers

    def read_convergence(self):
        """check whether the search is converged"""
        if not hasattr(self, "cur_gen"):
            self.da = DataConnection(self.db_path)
            self._check_generation()
        max_gen = self.conv_dict["generation"]
        if self.cur_gen > max_gen and (self.num_relaxed_gen == self.num_unrelaxed_gen):
            return True
        else:
            return False

    def _create_operator(
        self, op_params: dict, specific_params: dict, mod_name: str, convert_name=False
    ):
        """Create operators such as comparator, crossover, and mutation.

        Args:
            op_params: Operator parameters loaded from input file.
            specific_params: Operator parameters obtained based on system.

        """
        op_params = copy.deepcopy(op_params)
        method = op_params.pop("method", None)
        if method is None:
            raise RuntimeError(f"There is no operator {method}.")
        op_cls = registers.get(mod_name, method, convert_name=convert_name)
        init_args = inspect.getargspec(op_cls.__init__).args[1:]  # skip self
        for k, v in specific_params.items():
            if k in init_args:
                op_params.update(**{k: v})
        op = op_cls(**op_params)

        return op

    def _register_operators(self):
        """"""
        self.operators = {}

        op_dict = copy.deepcopy(self.ga_dict.get("operators", None))
        if op_dict is None:
            op_dict = {
                "mobile": {
                    "comparator": {"name": "InteratomicDistanceComparator"},
                    "crossover": {"name": "CutAndSplicePairing"},
                }
            }
        else:
            if "mobile" not in op_dict:  # This is for compatibility.
                op_dict_ = dict(mobile=op_dict)
                op_dict = op_dict_
            else:
                ...

        specific_params = dict(
            slab=self.da.get_slab(),
            # n_top=len(self.da.get_atom_numbers_to_optimize()),
            n_top=0,  # We will determine `n_top` on-the-fly when crossover and mutation.
            used_modes_file=self.directory
            / self.CALC_DIRNAME
            / "used_modes.json",  # SoftMutation
            # rng = self.rng # TODO: ase operators need np.random
        )

        # For compatibility,
        for attr in [
            "blmin",
            "number_of_variable_cell_vectors",
            "cell_bounds",
            "test_dist_to_slab",
            "use_tags",
        ]:
            if hasattr(self.generator, attr):
                specific_params.update(**{attr: getattr(self.generator, attr)})
            else:
                ...

        # We may not overwrite operators' covalent_ratio setting.
        # specific_params.update(covalent_ratio=self.generator.covalent_ratio)

        # The operators from ase (should be deprecated) use blmin and can only
        # check too_close since cov_max is not given while
        # the newly implemented operators by us use bond_distance_dict and
        # can check too_close and too_far based on covalent_ratio.
        # Also, the new random structure generator (random_surface_improved) uses
        # bond_distance_dict and covalent_ratio.
        # Thus, be careful when using random structure generator and operator in a
        # mixed way, either old generator with new operator or vice versa,
        # leading inconsistency in bond distance check.
        if hasattr(self.generator, "get_bond_distance_dict"):
            blmin = self.generator.get_bond_distance_dict(
                ratio=self.generator.covalent_ratio[0]
            )
            bond_distance_dict = self.generator.get_bond_distance_dict()
            specific_params.update(
                blmin=blmin,
                bond_distance_dict=bond_distance_dict,
            )
        else:
            specific_params.update(
                blmin=self.generator.blmin,
                bond_distance_dict=self.generator.blmin,
            )

        # StrainMutation uses cellbounds instead of cell_bounds
        if "cell_bounds" in specific_params:
            specific_params.update(cellbounds=specific_params["cell_bounds"])

        # Get operators for each group
        groups = ["mobile", "custom"]
        for g in groups:
            g_op_dict = op_dict.get(g, None)
            if g_op_dict is not None:
                self._print(f"operators for group {g} ->")
                group_operators = self._parse_group_operators(
                    g_op_dict, specific_params
                )
                self.operators[g] = group_operators
            else:
                ...

        return

    def _parse_group_operators(self, op_dict: dict, specific_params: dict):
        """Parse operators for a given group.

        Returns:
            A dict with comparing, pairing, and mutations.

        """
        # --- comparator
        comp_params = op_dict.get("comparator", None)
        if comp_params is not None:
            comparing = self._create_operator(
                comp_params, specific_params, "comparator", convert_name=True
            )

            self._print("  --- comparator ---")
            self._print(f"  Use comparator {comparing.__class__.__name__}.")
        else:
            comparing = None

        # --- crossover
        crossover_params = op_dict.get("crossover", None)
        if crossover_params is not None:
            pairing = self._create_operator(
                crossover_params, specific_params, "builder", convert_name=False
            )
            # For some ase-builtin operators, we manually set allow_variable_composition to False
            # by default. For others, we can set it through the input file.
            if hasattr(pairing, "allow_variable_composition"):
                ...
            else:
                pairing.allow_variable_composition = False

            self._print("  --- crossover ---")
            self._print(f"  Use crossover {pairing.__class__.__name__}.")
            self._print(
                f"  allow_variable_composition: {pairing.allow_variable_composition}."
            )
        else:
            pairing = None

        # --- mutations
        use_tags = specific_params.get("use_tags", False)

        mutation_list = op_dict.get("mutation", [])
        if mutation_list:
            mutations, probs = [], []
            if not isinstance(mutation_list, list):
                mutation_list = [mutation_list]
            for mut_params in mutation_list:
                prob = mut_params.pop("prob", 1.0)
                probs.append(prob)
                mut = self._create_operator(
                    mut_params, specific_params, "builder", convert_name=False
                )
                # Check whether mutation accepts molecules
                if use_tags:
                    if hasattr(mut, "use_tags"):
                        assert (
                            mut.use_tags
                        ), f"use_tags `{use_tags}` in mutation `{mut}` must be true."
                    else:
                        raise RuntimeError(
                            f"Mutation `{mut}` cannot be used in a search with tags."
                        )
                mutations.append(mut)

            self._print("  --- mutations ---")
            # self._print(f"mutation probability: {self.pmut}")
            for mut, prob in zip(mutations, probs):
                self._print(f"  Use mutation {mut.descriptor} with prob {prob}.")
            mutations = OperationSelector(probs, mutations, rng=np.random)
        else:
            mutations = None

        return dict(comparing=comparing, pairing=pairing, mutations=mutations)

    def _create_initial_population(
        self,
    ):
        # For all targets, we must have tags to infer num_atoms_substrate.
        # Thus, we can have a crystal substrate and include part of its atoms
        # for further crossover and mutation.
        substrate = self.generator._substrate
        num_atoms = len(substrate)

        tags = substrate.get_tags()
        if tags.shape[0] == 0:
            num_atoms_substrate = num_atoms
        else:
            substrate_atomic_indices = [i for i in range(num_atoms) if tags[i] == 0]
            if sorted(substrate_atomic_indices) == list(
                range(min(substrate_atomic_indices), max(substrate_atomic_indices) + 1)
            ):
                ...
            else:
                raise Exception(
                    "The atoms (tag == 0) in the substrate must be consecutive."
                )
            num_atoms_substrate = len(substrate_atomic_indices)

        canonicalised_substrate = substrate[:num_atoms_substrate]

        # Create the database to store information in
        da = PrepareDB(
            db_file_name=self.db_path,
            simulation_cell=canonicalised_substrate,
        )

        # Generate structures for the initial population
        starting_population = self.pop_manager._prepare_initial_population(
            generator=self.generator
        )

        self._print(f"save population {len(starting_population)} to database")
        for a in starting_population:
            da.add_unrelaxed_candidate(a)

        # Save some global information in the database
        # TODO: change this to the DB interface
        row = da.c.get(1)
        new_data = row["data"].copy()
        new_data["population_size"] = self.pop_manager.gen_size
        new_data["initial_population_size"] = self.pop_manager.init_size
        new_data["num_atoms_substrate"] = num_atoms_substrate

        da.c.update(1, data=new_data)

        self.da = DataConnection(self.db_path)

        return

    def evaluate_candidate(self, atoms: Atoms) -> None:
        """Evaluate the candidate's fitness.

        The fitness is stored in atoms.infop['raw_score']. The candidate
        is better with a larger raw_score.

        The supported properties are

            1. energy (potential energy)
            2. enthalpy (potential energy plus pressure correction)
            3. formation_energy (grand canonical)
            4. reaction_energy (TODO)

        Args:
            atoms: The candidate with calculated properties.

        Returns:
            None.

        """
        assert (
            atoms.info["key_value_pairs"].get("raw_score", None) is None
        ), "candidate already has raw_score before evaluation"

        # evaluate based on target property
        if self.target == "energy":
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            atoms.info["key_value_pairs"]["raw_score"] = -energy
            atoms.info["key_value_pairs"]["target"] = energy

            # Reduce the cell in the bulk structure search.
            # Check whether the cell is in the bound.
            from ase.build import niggli_reduce
            from ase.calculators.singlepoint import SinglePointCalculator

            if hasattr(self.generator, "cell_bounds"):
                stress = atoms.get_stress()
                niggli_reduce(atoms)
                calc = SinglePointCalculator(
                    atoms, energy=energy, forces=forces, stress=stress
                )
                atoms.calc = calc
                if self.generator.cell_bounds.is_within_bounds(atoms.get_cell()):
                    atoms.info["key_value_pairs"]["raw_score"] = -energy
                else:
                    atoms.info["key_value_pairs"]["raw_score"] = -1e8
        elif self.target == "formation_energy":
            identity_stats = atoms.info.get("identity_stats", None)
            assert (
                identity_stats is not None
            ), "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."
            chempot_dict = self.prop_dict["chempot"]

            energy = atoms.get_potential_energy()

            formation_energy = energy - np.sum(
                [chempot_dict[k] * v for k, v in identity_stats.items()]
            )
            atoms.info["key_value_pairs"]["raw_score"] = -formation_energy
            atoms.info["key_value_pairs"]["target"] = formation_energy
        elif self.target == "reaction_energy":
            ...  # TODO: ...
        else:
            raise RuntimeError(f"Unknown target {self.target}...")

        return

    def as_dict(self) -> dict:
        """"""
        engine_params = {}
        engine_params["random_seed"] = self.random_seed
        engine_params["method"] = "genetic_algorithm"
        engine_params["builder"] = self.generator.as_dict()
        engine_params["worker"] = self.worker.as_dict()
        engine_params["params"] = self.ga_dict

        engine_params = copy.deepcopy(engine_params)

        return engine_params


if __name__ == "__main__":
    ...
