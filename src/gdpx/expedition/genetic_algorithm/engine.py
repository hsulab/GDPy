import copy
import itertools
import pathlib
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from ase.ga.offspring_creator import OperationSelector
from ase.ga.utilities import CellBounds
from ase.io import read, write

from gdpx.factory.builder import canonicalise_builder
from gdpx.utils.atoms_tags import get_tags_per_species
from gdpx.utils.strconv import integers_to_string

from ..expedition import BaseExpedition
from ..persist.database import GenerationInfo
from ..persist.database import GlobalOptimisationDatabase as GODB
from .operators import instantiate_a_genetic_operator
from .population.manager import PopulationManager


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


def reduce_cell_by_bounds(atoms: Atoms, cell_bounds: CellBounds) -> Atoms:
    """Reduce the cell of atoms based on the cell bounds.

    If the cell is not within the bounds, the raw score is set to -1e8,
    which is small enough to not be selected in the population.

    Args:
        atoms: The atoms object.
        cell_bounds: The cell bounds.

    Returns:
        The reduced atoms object.

    """
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    raw_score = atoms.info["key_value_pairs"]["raw_score"]

    niggli_reduce(atoms)
    calc = SinglePointCalculator(atoms, energy=energy, forces=forces, stress=stress)
    atoms.calc = calc
    if cell_bounds.is_within_bounds(atoms.get_cell()):
        atoms.info["key_value_pairs"]["raw_score"] = raw_score
    else:
        atoms.info["key_value_pairs"]["raw_score"] = -1e8

    return atoms


class GeneticAlgorithmBroadcaster:
    """Broadcast genetic_algorithm_engine by parameters."""

    def __init__(self, builder, params, random_seed=None):
        """"""
        new_params_list = self._broadcast_parameters(params)

        input_params_list = []
        for new_params in new_params_list:
            input_params = dict(
                builder=copy.deepcopy(builder),
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
        elif target == "cohesive_energy" or target == "formation_energy":
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
            raise Exception(f"Cannot broadcast unknown target {target}.")

        return new_params_list


class GeneticAlgorithmEngine(BaseExpedition):
    """The genetic algorithm engine for structure search.

    The systems include bulk, surface, cluster, and surface with adsorbates.
    A database is used to store the information along the search, and the
    reserved keywords in the database including generation, relaxed, queued,
    extinct, description, and pairing.
    Three types of operators are used, namely, comparator, crossover (pairing)
    and mutation.

    """

    # local optimisation directory
    CALC_DIRNAME = "tmp_folder"

    #: Prefix of each generation's directory.
    GEN_PREFIX: str = "gen"

    def __init__(
        self,
        builder: dict,
        params: dict,
        *args,
        **kwargs,
    ):
        """Initialise engine.

        Args:
            builder: Define the system to explore.

        """
        super().__init__(*args, **kwargs)

        # For compatibility
        ga_dict = params

        # Database
        self.db_name = ga_dict.get("database", "mydb.db")

        # Store initial parameters
        self.ga_dict = copy.deepcopy(ga_dict)

        population_params = self.ga_dict.get("population", None)
        if population_params is not None:
            if "init" in population_params:
                seed_file = population_params["init"].get("seed_file", None)
                if seed_file is not None:
                    self.ga_dict["population"]["init"]["seed_file"] = str(pathlib.Path(seed_file).resolve())

        # Check random consistency, generator and population
        self._print(f"GA RANDOM SEED {self.random_seed}")

        # Check builder for random structure generation
        if isinstance(builder, dict):
            builder_params = copy.deepcopy(builder)
        else:  # assume it is a StructureBuilder
            builder_params = builder.as_dict()

        # The builder has its own rng but it is initialised from the engine's random_seed.
        # If random_bulk is used, due to its deprecated np.random,
        # the results may not be reproducible.
        prev_seed = builder_params.get("random_seed", None)
        builder_params.update(random_seed=self.random_seed)
        self.generator = canonicalise_builder(builder_params)

        self._print(f"OVERWRITE BUILDER SEED FROM {prev_seed} TO {self.random_seed}")

        # The ase built-in cut_and_splice reinits tags from 0 if use_tags is false,
        # Here, no matter what type of system is explored, we enforce the builder's use_tags
        # to be true as it retains the tags information.
        assert self.generator is not None, "Builder is not properly initialised."
        if hasattr(self.generator, "use_tags"):
            if self.generator.use_tags:
                ...
            else:
                self.generator.use_tags = True
                self._print(
                    f"Builder `{self.generator.name}` changes `use_tags` to true for formation energy computation."
                )
        else:
            raise RuntimeError(f"Builder `{self.generator.name}` does not have true `use_tags`.")

        # Worker will be lazily checked in run
        self.worker = None

        # Sanity check on target property
        self.prop_dict = ga_dict.get("property", dict(target="energy"))
        target = self.prop_dict.get("target", None)
        assert target in (
            "energy",
            "cohesive_energy",
            "formation_energy",
        ), f"Target `{target}` is not supported yet."
        if target == "cohesive_energy" or target == "formation_energy":
            if "chempot" not in self.prop_dict:
                raise RuntimeError("The `chempot` is not provided in the property section.")
        else:
            ...

        self.target = target

        # Population and check target-population consistency
        self.pop_manager = PopulationManager(ga_dict["population"], rng=self.rng)
        if self.pop_manager.name == "variable":
            if self.target not in ("cohesive_energy", "formation_energy"):
                raise RuntimeError(
                    f"Population manager `{self.pop_manager.name}` is only compatible with "
                    + "formation energy or cohesive energy target."
                )
        else:
            ...

        # Convergence
        self.conv_dict = ga_dict["convergence"]

        # Misc
        self.use_archive = ga_dict.get("use_archive", True)

        return

    @BaseExpedition.directory.setter
    def directory(self, directory: Union[str, pathlib.Path]) -> None:
        """"""
        self._directory = pathlib.Path(directory).resolve()
        self.db_path = self._directory / self.db_name

        return

    def report(self):
        """Write reports of this GA search.

        One file contains all relaxed structures and one figure shows the target properties
        in each generation.

        """
        self._print("restart the database...")
        self.da = GODB(self.db_path)
        results = self.directory / "results"
        if not results.exists():
            results.mkdir()

        # write structures that are already sorted by raw_score
        all_relaxed_candidates = self.da.get_all_relaxed_candidates()
        write(results / "all_candidates.xyz", all_relaxed_candidates)

        # plot population evolution
        data = []
        gen_num = self.da.get_generation_number()  # equals finished generation plus one
        self._print(f"Genetic Algorithm Statistics with {gen_num - 1} generations: ")
        for i in range(gen_num):
            current_candidates = [
                atoms for atoms in all_relaxed_candidates if atoms.info["key_value_pairs"]["generation"] == i
            ]
            properties = np.array([a.info["key_value_pairs"]["target"] for a in current_candidates])
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

    def update_active_params(self, prev_wdir: pathlib.Path) -> None:
        """"""
        candidates = read(prev_wdir / "results" / "all_candidates.xyz", ":")
        selected_candidates = candidates[: self.pop_manager.init_size]
        assert isinstance(selected_candidates, list)
        assert all(isinstance(c, Atoms) for c in selected_candidates)

        self.pop_manager.init_seed_file = selected_candidates

        return

    def run(self) -> None:
        """Run the GA procedure several steps.

        Default setting would run the algorithm many times until its convergence.
        This is useful for running optimisations with serial worker.

        """
        # Search target
        self._print(f"===== Genetic Algorithm =====")
        self._print(f"Target of Global Optimisation is {self.target}")

        # Update output functions
        self.pop_manager._print = self._print
        self.pop_manager._debug = self._debug

        # Check random structure builder (generator)
        self._print("===== register builder =====")
        for l in str(self.generator).split("\n"):
            self._print(l)
        assert self.generator is not None, "GA has not set its builder properly."
        self._print(f"random_state: {self.generator.random_seed}")

        # Check worker
        self._print("===== register worker =====")
        assert self.worker is not None, "GA has not set its worker properly."
        self.worker.directory = self.directory / self.CALC_DIRNAME

        if self.generator.name == "random_bulk" and self.worker.driver.setting.task != "cmin":
            content = "*" * 50 + "\n"
            content += "*    " + f"{'':<44s}" + "*\n"
            content += "*    " + f"{'YOU ARE EXPLORING RANDOM BULK STRUCTURES':<44s}" + "*\n"
            content += "*    " + f"{'BETTER USE `task: cmin` IN THE DRIVER':<44s}" + "*\n"
            content += "*    " + f"{'OTHERWISE THE CELL WILL NOT BE CHANGED':<44s}" + "*\n"
            content += "*    " + f"{'':<44s}" + "*\n"
            content += "*" * 50 + "\n"
            for l in content.split("\n"):
                self._print(l)
        else:
            self._print("")

        # Check database existence and generation number to determine restart
        self._print("===== register database =====")
        self._debug(f"database path: {str(self.db_path)}")
        if not self.db_path.exists():
            self._print("create a new database...")
            self._create_initial_population()
        else:
            self._print("restart the database...")
            self.da = GODB(self.db_path)

        num_atoms_substrate = self.da.get_param("num_atoms_substrate")
        self._print(f"{num_atoms_substrate=}")

        # Register mutation and comparassion operators
        self._print("===== register operators =====")
        self._register_operators()

        # Run genetic
        gen_info = None
        for _ in range(1000):
            gen_info = self.da.get_generation_info()
            if self.read_convergence(gen_info):
                self._print("reach maximum generation...")
                self.report()
                break
            curr_convergence = self._irun(gen_info)
            if not curr_convergence:
                self._print("current generation does not converge...")
                break

        return

    def _irun(self, gen_info: GenerationInfo) -> bool:
        """main procedure"""
        # Generation information
        gen_num = gen_info.num
        self._print(f"===== Generation {gen_num:>04d} =====")
        self._print(f"  {gen_info.state}")
        self._print(f"  num_relaxed: {gen_info.num_relaxed}")
        self._print("  confids: " + integers_to_string(sorted(gen_info.relaxed_confids), inp_convention="lmp"))
        self._print(f"  num_unrelaxed: {gen_info.num_unrelaxed}")
        self._print("  confids: " + integers_to_string(sorted(gen_info.unrelaxed_confids), inp_convention="lmp"))

        # Relax structures
        assert self.worker is not None, "GA has not set its worker properly."
        if gen_num == 0:
            # mark_as_queued later before optimisation
            current_candidates = self.da.get_all_unrelaxed_candidates(mark_as_queued=False)
        else:
            # --- update population
            # Check candidate origin for the current generation
            candidate_groups, num_paired, num_mutated, num_random = self.pop_manager._get_current_candidates(
                database=self.da, curr_gen=gen_num
            )
            self._print("candidate origin distribution before:")
            self._print("  " + "".join([f"{k:<8s}: {len(v):<4d}  " for k, v in candidate_groups.items()]))

            self.pop_manager.update_population(
                database=self.da,
                comparing=self.operators["mobile"]["comparing"],
            )
            assert self.pop_manager.population is not None

            pop_confids = [a.info["confid"] for a in self.pop_manager.population.pop]
            self._print(f"number of structures in population: {len(pop_confids)}")
            self._print(f"confids in population: {integers_to_string(pop_confids, inp_convention='lmp')}")

            self.pop_manager._update_generation_settings(
                self.operators["mobile"]["mutations"],
                self.operators["mobile"]["pairing"],
            )

            # Generate candidates for the current generation
            is_prodcution_complete = (num_paired + num_mutated + num_random) >= self.pop_manager.gen_size
            if not is_prodcution_complete:
                self._print("Current generation has not finished...")
            # The current candidates have not been created completely.
            # For example, num_relaxed != num_unrelaxed, need create more candidates...
            current_candidates = self.pop_manager._prepare_current_population(
                database=self.da,
                curr_gen=gen_num,
                generator=self.generator,
                operators=self.operators,
                candidate_groups=candidate_groups,
                num_paired=num_paired,
                num_mutated=num_mutated,
                num_random=num_random,
            )

            # Validate candidate origins for the current generation
            candidate_groups, num_paired, num_mutated, num_random = self.pop_manager._get_current_candidates(
                database=self.da, curr_gen=gen_num
            )
            self._print("candidate origin distribution after:")
            self._print("  " + "".join([f"{k:<8s}: {len(v):<4d}  " for k, v in candidate_groups.items()]))

        self._print(">>>>> Optimisation >>>>>")
        generation_directory = self.directory / self.CALC_DIRNAME / f"gen{gen_num}"
        self.worker.directory = generation_directory

        for ia, a in enumerate(current_candidates):
            parents = "none"
            if "parents" in a.info["data"]:
                parents = " ".join([str(x) for x in a.info["data"]["parents"]])
            self._print(
                f"{ia:>4d} confid={a.info['confid']:>6d} parents={parents:<14s} origin={a.info['key_value_pairs']['origin']:<32s} extinct={a.info['key_value_pairs']['extinct']:<4d}"
            )

        # TODO: We need check if optimisation task is already created.
        if not generation_directory.exists():
            for atoms in current_candidates:
                self.da.mark_as_queued(atoms)  # It only marks when not queued before.
            if current_candidates:
                confids = [a.info["confid"] for a in current_candidates]
                self._print(f"start to run structure {integers_to_string(confids, inp_convention='lmp')}")
                _ = self.worker.run(current_candidates)  # retrieve later
        else:
            self._print(f"calculation directory for generation {gen_num} exists.")

        # Check if there were finished jobs
        assert self.generator is not None, "GA has not set its builder properly."
        curr_convergence = False
        self.worker.inspect(resubmit=True)
        if self.worker.get_number_of_running_jobs() == 0:
            self._print(">>>>> Evaluation >>>>>")
            # TODO: If stop during evaluation?
            whether_reduce_cell = hasattr(self.generator, "cell_bounds")
            if whether_reduce_cell:
                self._print("The candidates will be reduced by cell bounds.")
            converged_candidates = [t[-1] for t in self.worker.retrieve(use_archive=self.use_archive)]
            for ia, cand in enumerate(converged_candidates):
                # update extra info
                extra_info = dict(
                    data={},
                    key_value_pairs={"generation": gen_num, "extinct": 0},
                )
                cand.info.update(extra_info)
                # get tags
                confid = cand.info["confid"]
                if self.generator.use_tags:
                    rows = list(self.da.connection.select(f"relaxed=0,confid={confid}"))
                    rows = sorted(
                        [row for row in rows if row.formula],
                        key=lambda row: row.mtime,
                    )
                    if len(rows) > 0:
                        previous_atoms = rows[-1].toatoms(add_additional_information=True)
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
                if whether_reduce_cell:
                    cand = reduce_cell_by_bounds(cand, self.generator.cell_bounds)
                fitness = cand.info["key_value_pairs"]["raw_score"]
                cand_stat = f"{ia:>4d} confid {confid:<6d} fitness {fitness:>16.4f} "
                if "identity_stats" in cand.info:
                    identity_info = "  " + " ".join([f"{k}: {v}" for k, v in cand.info["identity_stats"].items()])
                    cand_stat += identity_info
                self._print(cand_stat)
                self.da.add_relaxed_step(cand)
            curr_convergence = True
        else:
            self._print("Worker is unfinished.")

        return curr_convergence

    def get_workers(self, gen_info: Optional[GenerationInfo] = None) -> list:
        """Get all workers used by this expedition."""
        if gen_info is None:
            da = self.da if hasattr(self, "da") else GODB(self.db_path)
            gen_info = da.get_generation_info()

        assert self.worker is not None, "GA has not set its worker properly."
        if hasattr(self.worker.potter, "remove_loaded_models"):
            self.worker.potter.remove_loaded_models()

        workers = []
        for i in range(gen_info.num):
            curr_worker = copy.deepcopy(self.worker)
            curr_worker.directory = self.directory / self.CALC_DIRNAME / (f"{self.GEN_PREFIX}{i}")
            workers.append(curr_worker)

        return workers

    def read_convergence(self, gen_info: Optional[GenerationInfo] = None) -> bool:
        """check whether the search is converged"""
        if gen_info is None:
            da = self.da if hasattr(self, "da") else GODB(self.db_path)
            gen_info = da.get_generation_info()

        is_converged = False

        max_gen = self.conv_dict["generation"]
        if gen_info.num > max_gen:
            is_converged = True
        else:
            is_converged = False

        return is_converged

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

        specific_params: dict[str, Any] = dict(
            slab=self.da.get_substrate(),
            # n_top=len(self.da.get_atom_numbers_to_optimize()),
            n_top=0,  # We will determine `n_top` on-the-fly when crossover and mutation.
            used_modes_file=self.directory / self.CALC_DIRNAME / "used_modes.json",  # SoftMutation
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
        assert self.generator is not None, "GA has not set its builder properly."
        if hasattr(self.generator, "get_bond_distance_dict"):
            if hasattr(self.generator, "covalent_ratio"):
                # make sure it is a tuple of two floats
                cov_min = self.generator.covalent_ratio[0]  # type: ignore
            else:
                cov_min = 0.8  # default value
            blmin = self.generator.get_bond_distance_dict(ratio=cov_min)
            bond_distance_dict = self.generator.get_bond_distance_dict(ratio=1.0)
            specific_params.update(
                blmin=blmin,
                bond_distance_dict=bond_distance_dict,
            )
        else:
            raise Exception(
                f"Genetic only supports builder with "
                + "get_bond_distance_dict, for example, "
                + "random_bulk and random_structure_improved."
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
                group_operators = self._parse_group_operators(g_op_dict, specific_params)
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
            comparing = instantiate_a_genetic_operator("comparator", comp_params, specific_params)

            self._print("  --- comparator ---")
            self._print(f"  Use comparator {comparing.__class__.__name__}.")
        else:
            comparing = None

        # --- crossover
        crossover_params = op_dict.get("crossover", None)
        if crossover_params is not None:
            pairing = instantiate_a_genetic_operator(
                "crossover",
                crossover_params,
                specific_params,
            )
            # For some ase-builtin operators, we manually set allow_variable_composition to False
            # by default. For others, we can set it through the input file.
            if hasattr(pairing, "allow_variable_composition"):
                ...
            else:
                pairing.allow_variable_composition = False

            self._print("  --- crossover ---")
            self._print(f"  Use crossover {pairing.__class__.__name__}.")
            self._print(f"  allow_variable_composition: {pairing.allow_variable_composition}.")
        else:
            pairing = None

        # --- mutations
        mutation_list = op_dict.get("mutation", [])
        if mutation_list:
            mutations, probs = [], []
            if not isinstance(mutation_list, list):
                mutation_list = [mutation_list]
            for mut_params in mutation_list:
                prob = mut_params.pop("prob", 1.0)
                probs.append(prob)
                mut_use_tags = mut_params.get("use_tags", True)
                specific_params_ = copy.deepcopy(specific_params)
                sys_use_tags = specific_params_.pop("use_tags", True)
                mut_params["use_tags"] = sys_use_tags and mut_use_tags
                mut = instantiate_a_genetic_operator("mutation", mut_params, specific_params_)
                # Check whether mutation accepts molecules
                if hasattr(mut, "use_tags"):
                    if mut_use_tags:
                        assert mut.use_tags, f"use_tags `{mut.use_tags}` in mutation `{mut}` must be true."
                    else:
                        # HACK: We may disbale the use_tags in the mutation if all our fragments
                        # are just atoms, and the mutation does not mess up with tags.
                        ...
                else:
                    raise RuntimeError(f"Mutation `{mut}` cannot be used in a search with tags.")
                assert "Mutation" in mut.descriptor, f"{mut} must have `Mutation` in its descriptor."
                mutations.append(mut)

            self._print("  --- mutations ---")
            # self._print(f"mutation probability: {self.pmut}")
            for mut, prob in zip(mutations, probs):
                self._print(f"  Use mutation {mut.descriptor} with prob {prob}.")
            mutations = OperationSelector(probs, mutations, rng=np.random)
        else:
            mutations = OperationSelector([], [], rng=np.random)

        return dict(comparing=comparing, pairing=pairing, mutations=mutations)

    def _create_initial_population(
        self,
    ):
        self._print("===== Population Info =====")
        content = "For generation > 0,\n"
        content += "{:>8s}  {:>8s}  {:>8s}  {:>8s}\n".format("Reprod", "Random", "Mutate", "Total")
        content += "{:>8d}  {:>8d}  {:>8d}  {:>8d}\n".format(
            self.pop_manager.gen_rep_size,
            self.pop_manager.gen_ran_size,
            self.pop_manager.gen_mut_size,
            self.pop_manager.gen_size,
        )
        content += "Note: Reproduced structure has a chance (pmut) to mutate.\n"
        for l in content.split("\n"):
            self._print(l)

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
                range(
                    min(substrate_atomic_indices),
                    max(substrate_atomic_indices) + 1,
                )
            ):
                ...
            else:
                raise Exception("The atoms (tag == 0) in the substrate must be consecutive.")
            num_atoms_substrate = len(substrate_atomic_indices)

        canonicalised_substrate = substrate[:num_atoms_substrate]

        # Initialise database
        da = GODB(self.db_path)
        da.init_task(
            canonicalised_substrate,
            data=dict(
                population_size=self.pop_manager.gen_size,
                initial_population_size=self.pop_manager.init_size,
                num_atoms_substrate=num_atoms_substrate,
            ),
        )

        # Generate structures for the initial population
        starting_population = self.pop_manager._prepare_initial_population(generator=self.generator)

        self._print(f"save population {len(starting_population)} to database")
        for a in starting_population:
            da.add_unrelaxed_candidate(a, generation=0)

        self.da = da

        return

    def evaluate_candidate(self, atoms: Atoms) -> None:
        """Evaluate the candidate's fitness.

        The fitness is stored in atoms.infop['raw_score']. The candidate is better with a larger raw_score.

        The supported properties are

            - energy (potential energy)
            - enthalpy (potential energy plus pressure correction)
            - cohesive energy (grand canonical)
            - formation_energy (grand canonical)
            - reaction_energy (TODO)

        Args:
            atoms: The candidate with calculated properties.

        Returns:
            None.

        """
        assert atoms.info["key_value_pairs"].get("raw_score", None) is None, (
            "candidate already has raw_score before evaluation"
        )

        # evaluate based on target property
        if self.target == "energy":
            energy = atoms.get_potential_energy()
            atoms.info["key_value_pairs"]["raw_score"] = -energy
            atoms.info["key_value_pairs"]["target"] = energy
        elif self.target == "cohesive_energy":
            chempot_dict = self.prop_dict["chempot"]

            energy = atoms.get_potential_energy()
            cohesive_energy = energy - np.sum([chempot_dict[s] for s in atoms.get_chemical_symbols()])
            atoms.info["key_value_pairs"]["raw_score"] = -cohesive_energy
            atoms.info["key_value_pairs"]["target"] = cohesive_energy
        elif self.target == "formation_energy":
            identity_stats = atoms.info.get("identity_stats", None)
            assert identity_stats is not None, (
                "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."
            )
            chempot_dict = self.prop_dict["chempot"]

            energy = atoms.get_potential_energy()

            formation_energy = energy - np.sum([chempot_dict[k] * v for k, v in identity_stats.items()])
            atoms.info["key_value_pairs"]["raw_score"] = -formation_energy
            atoms.info["key_value_pairs"]["target"] = formation_energy
        elif self.target == "reaction_energy":
            raise NotImplementedError()
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
