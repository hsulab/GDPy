import copy
import inspect
import itertools
import pathlib
from collections.abc import Mapping
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.build import niggli_reduce
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from gdpx.structures.builders import REGISTER as BUILDER_REGISTER
from gdpx.structures.builders.factory import canonicalise_builder
from gdpx.structures.geometry.ga import CellBounds
from gdpx.utils.atoms_tags import get_tags_per_species
from gdpx.utils.strconv import integers_to_string

from ..expedition import BaseExpedition
from ..objective import is_default_objective, normalise_objective, reject_legacy_property
from ..persist.database import (
    CANDIDATES_DATABASE_FILENAME,
    GenerationInfo,
    GenerationState,
)
from ..persist.database import GlobalOptimisationDatabase as GODB
from .operators import instantiate_a_genetic_operator
from .core import OperationSelector, RandomStreamRegistry
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

    def __init__(
        self,
        population: dict,
        convergence: dict,
        operators: Optional[dict] = None,
        objective: Optional[dict] = None,
        use_archive: bool = True,
        random_seed=None,
        **legacy_kwargs,
    ):
        """"""
        reject_legacy_property(legacy_kwargs)
        if "database" in legacy_kwargs:
            raise ValueError(
                "The genetic-algorithm database filename is no longer configurable; "
                f"remove 'database'. GDPy uses {CANDIDATES_DATABASE_FILENAME!r}."
            )
        if legacy_kwargs:
            key = next(iter(legacy_kwargs))
            raise TypeError(f"Unexpected genetic-algorithm recipe key {key!r}.")

        objective = normalise_objective(
            objective,
            {"energy", "cohesive_energy", "formation_energy"},
        )
        recipe = dict(
            population=population,
            operators=operators,
        )
        if not is_default_objective(objective):
            recipe["objective"] = objective
        recipe.update(convergence=convergence, use_archive=use_archive)
        new_params_list = self._broadcast_parameters(recipe)

        input_params_list = []
        for new_params in new_params_list:
            input_params = copy.deepcopy(new_params)
            input_params["random_seed"] = copy.deepcopy(random_seed)
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
            List-valued chemical potentials create independent objectives.

        """
        new_params_list = []

        objective = params.get("objective", dict(target="energy"))
        target = objective.get("target", "energy")
        if target == "energy":
            new_params = copy.deepcopy(params)
            new_params_list.append(new_params)
        elif target == "cohesive_energy" or target == "formation_energy":
            chemical_potentials = []
            for k, v in objective["chemical_potentials"].items():
                if isinstance(v, list):
                    chemical_potentials.append([(k, v_i) for v_i in v])
                else:  # This must be a number.
                    chemical_potentials.append([(k, v)])
            broadcasted_chemical_potentials = list(
                itertools.product(*chemical_potentials)
            )
            for chemical_potential in broadcasted_chemical_potentials:
                new_params = copy.deepcopy(params)
                new_params["objective"]["chemical_potentials"] = {
                    k: v for k, v in chemical_potential
                }
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
        population: dict,
        convergence: dict,
        operators: Optional[dict] = None,
        objective: Optional[dict] = None,
        use_archive: bool = True,
        *args,
        **kwargs,
    ):
        """Initialise engine.

        Args:
            population: Define population creation and evolution.

        """
        reject_legacy_property(kwargs)
        if "database" in kwargs:
            raise ValueError(
                "The genetic-algorithm database filename is no longer configurable; "
                f"remove 'database'. GDPy uses {CANDIDATES_DATABASE_FILENAME!r}."
            )
        super().__init__(*args, **kwargs)
        self.random_streams = RandomStreamRegistry(self.random_seed)
        self.rng = self.random_streams.get("engine")

        population = copy.deepcopy(population)
        if "random_generator" in population:
            raise ValueError("Legacy GA population key 'random_generator' is not supported; use 'builders'.")
        builders_config = population.pop("builders", None)
        reference_builder = population.pop("reference_builder", "random")
        if not isinstance(builders_config, Mapping) or not builders_config:
            raise ValueError("GA population configuration requires a non-empty 'builders' mapping.")
        if not all(isinstance(name, str) and name for name in builders_config):
            raise ValueError("GA population builder names must be non-empty strings.")
        if not isinstance(reference_builder, str):
            raise ValueError("GA population.reference_builder must name one of population.builders.")
        if reference_builder not in builders_config:
            if reference_builder == "random":
                raise ValueError(
                    "GA population defaults reference_builder to 'random', but population.builders "
                    "does not define 'random'; set reference_builder to another builder name."
                )
            raise ValueError("GA population.reference_builder must name one of population.builders.")

        self.pop_manager = PopulationManager(
            population, rng=self.random_streams.get("population")
        )
        self.periodic = self.pop_manager.periodic
        self.preserve_fragments = self.pop_manager.preserve_fragments

        objective = normalise_objective(
            objective,
            {"energy", "cohesive_energy", "formation_energy"},
        )
        ga_dict = dict(
            population=population,
            operators=operators,
        )
        if not is_default_objective(objective):
            ga_dict["objective"] = objective
        ga_dict.update(convergence=convergence, use_archive=use_archive)

        # Store initial parameters
        self.ga_dict = copy.deepcopy(ga_dict)

        # Check random consistency, builders and population
        self._print(f"GA RANDOM SEED {self.random_seed}")
        self.builders = {}
        for name, builder_config in builders_config.items():
            child_seed = self.random_streams.seed(f"builder/{name}")
            if isinstance(builder_config, Mapping):
                params = copy.deepcopy(dict(builder_config))
                population_owned = {"pbc", "use_tags"}.intersection(params)
                if population_owned:
                    migrations = {
                        "pbc": "population.periodic",
                        "use_tags": "population.preserve_fragments",
                    }
                    details = ", ".join(
                        f"{key} -> {migrations[key]}" for key in sorted(population_owned)
                    )
                    raise ValueError(
                        f"GA population builder {name!r} contains population-owned keys: {details}."
                    )
                method = params.get("method", "direct")
                builder_class = BUILDER_REGISTER[method]
                if "pbc" in inspect.signature(builder_class.__init__).parameters:
                    params["pbc"] = self.periodic
                params["random_seed"] = child_seed
                builder = canonicalise_builder(params)
            else:
                builder = builder_config
                if not hasattr(builder, "set_rng"):
                    raise TypeError(f"Population builder {name!r} must be a builder configuration or instance.")
                builder.set_rng(child_seed)
            if builder is None:
                raise ValueError(f"Population builder {name!r} could not be initialised.")
            if hasattr(builder, "use_tags"):
                builder.use_tags = True
            builder.rng = self.random_streams.get(f"builder/{name}")
            self.builders[name] = builder
            self._print(f"SET BUILDER {name!r} SEED TO {child_seed}")

        self.reference_builder_name = reference_builder
        # Keep this alias for operator/system metadata that comes from one reference builder.
        self.generator = self.builders[reference_builder]

        # Worker will be lazily checked in run
        self.worker = None

        # Search objective
        self.objective = objective
        self.target = objective["target"]

        # Population and check target-population consistency
        configured_builder_names = [
            allocation["builder"] for allocation in self.pop_manager.initial_builder_allocations
        ] + [item["builder"] for item in self.pop_manager.completion_builder_proportions]
        self.pop_manager._require_builders(self.builders, configured_builder_names)
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
        self.db_path = self._directory / CANDIDATES_DATABASE_FILENAME

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
        candidates_path = (prev_wdir / "results" / "all_candidates.xyz").resolve()
        candidates = read(candidates_path, ":")
        selected_candidates = candidates[: self.pop_manager.init_size]
        assert isinstance(selected_candidates, list)
        assert all(isinstance(c, Atoms) for c in selected_candidates)
        if len(selected_candidates) != self.pop_manager.init_size:
            raise RuntimeError(
                f"Active population contains {len(selected_candidates)} structures; "
                f"initial.total_size requires {self.pop_manager.init_size}."
            )
        name = "active_population"
        self.builders[name] = canonicalise_builder(
            dict(method="direct", frames=str(candidates_path), indices=list(range(self.pop_manager.init_size)))
        )
        self.pop_manager.initial_builder_allocations = [
            dict(
                builder=name,
                size=self.pop_manager.init_size,
                maximum_attempts=self.pop_manager.init_size * self.pop_manager.MAX_ATTEMPTS_MULTIPLIER,
            )
        ]
        self.ga_dict["population"]["initial"]["builder_allocations"] = [
            dict(builder=name, size=self.pop_manager.init_size)
        ]

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

        self._print("===== register builders =====")
        for name, builder in self.builders.items():
            self._print(f"--- {name} ---")
            for line in str(builder).split("\n"):
                self._print(line)
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
        self._register_database()
        assert self.da is not None, "GA has not set its database properly."

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
                self._print("The search reaches maximum generation or extinction...")
                self.report()
                break
            gen_state = self._irun(gen_info)
            if gen_state == GenerationState.OPT_UNFINISHED:
                self._print("The optimisation has not finished yet.")
                break

        return

    def _irun(self, gen_info: GenerationInfo) -> GenerationState:
        """main procedure"""
        # Generation information
        gen_num = gen_info.num
        self._print(f"===== Generation {gen_num:>04d} =====")
        self._print(f"  {gen_info.state}")
        self._print(f"  num_relaxed: {gen_info.num_relaxed}")
        self._print("  confids: " + integers_to_string(sorted(gen_info.relaxed_confids), inp_convention="lmp"))
        self._print(f"  num_unrelaxed: {gen_info.num_unrelaxed}")
        self._print("  confids: " + integers_to_string(sorted(gen_info.unrelaxed_confids), inp_convention="lmp"))

        if gen_info.state == GenerationState.EXTINCTED:
            self._print("All candidates extincted, cannot proceed further.")
            return GenerationState.EXTINCTED

        # Get structures for the current generation
        assert self.worker is not None, "GA has not set its worker properly."
        if gen_num == 0:
            current_candidates = self._get_candidates_for_the_first_generation(gen_num)
        else:
            current_candidates = self._get_candidates_for_the_other_generation(gen_num)

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
        gen_state = GenerationState.OPT_UNFINISHED
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
                self.pop_manager.validate_candidate(cand, "relaxed", ia)
                # evaluate raw score
                self.evaluate_candidate(cand)
                self.pop_manager._extinct_candidate(cand)
                if whether_reduce_cell:
                    cand = reduce_cell_by_bounds(cand, self.generator.cell_bounds)
                fitness = cand.info["key_value_pairs"]["raw_score"]
                cand_stat = f"{ia:>4d} confid {confid:<6d} fitness {fitness:>16.4f} extinct {cand.info['key_value_pairs']['extinct']:<2d}"
                if "identity_stats" in cand.info:
                    identity_info = "  " + " ".join([f"{k}: {v}" for k, v in cand.info["identity_stats"].items()])
                    cand_stat += identity_info
                self._print(cand_stat)
                self.da.add_relaxed_step(cand)
            num_extincted = sum([cand.info["key_value_pairs"]["extinct"] for cand in converged_candidates])
            self._print(f"extinct {num_extincted} candidates.")
            gen_state = GenerationState.OPT_FINISHED
        else:
            self._print("Worker is unfinished.")

        return gen_state

    def _get_candidates_for_the_first_generation(self, gen_num: int) -> list[Atoms]:
        """The main procedure for the first generation."""
        assert gen_num == 0, "This function is only for the first generation."

        candidate_groups = self.pop_manager._get_current_candidates(database=self.da, curr_gen=gen_num)
        starting_population = list(candidate_groups["initial"])
        plan = self.da.get_generation_plan(gen_num)
        if plan is None:
            plan = {"stage": "initial", "random_states": self.random_streams.snapshot()}
            self.da.set_generation_plan(gen_num, plan)
        elif "random_states" in plan:
            self.random_streams.restore(plan["random_states"])

        existing_by_builder: dict[str, int] = {}
        for atoms in starting_population:
            builder_name = atoms.info.get("data", {}).get("builder")
            if not isinstance(builder_name, str):
                raise RuntimeError("Persisted initial structure does not identify its builder.")
            existing_by_builder[builder_name] = existing_by_builder.get(builder_name, 0) + 1

        for allocation in self.pop_manager.initial_builder_allocations:
            name = allocation["builder"]
            remaining = allocation["size"] - existing_by_builder.get(name, 0)
            if remaining < 0:
                raise RuntimeError(f"Too many persisted initial structures for builder {name!r}.")
            frames = self.pop_manager._generate_from_builder(
                name, self.builders[name], remaining, allocation["maximum_attempts"]
            )
            for atoms in self.pop_manager.clean_initial_structures(frames, name):
                self.da.add_unrelaxed_candidate(atoms, generation=gen_num)
                starting_population.append(atoms)
                plan["random_states"] = self.random_streams.snapshot()
                self.da.set_generation_plan(gen_num, plan)

        if len(starting_population) != self.pop_manager.init_size:
            raise RuntimeError(
                f"Initial generation contains {len(starting_population)} candidates; "
                f"expected {self.pop_manager.init_size}."
            )
        plan["stage"] = "complete"
        plan["random_states"] = self.random_streams.snapshot()
        self.da.set_generation_plan(gen_num, plan)

        # Validate candidate origins for the current generation
        candidate_groups = self.pop_manager._get_current_candidates(database=self.da, curr_gen=gen_num)
        self._print("candidate origin distribution after:")
        self._print("  " + "".join([f"{k:<8s}: {len(v):<4d}  " for k, v in candidate_groups.items()]))

        return starting_population

    def _get_candidates_for_the_other_generation(self, gen_num: int) -> list[Atoms]:
        """The main procedure for other generations."""
        # Check candidate origin for the current generation
        candidate_groups = self.pop_manager._get_current_candidates(database=self.da, curr_gen=gen_num)
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
        num_candidates = sum(len(group) for group in candidate_groups.values())
        is_prodcution_complete = num_candidates >= self.pop_manager.gen_size
        if not is_prodcution_complete:
            self._print("Current generation has not finished...")
        # The current candidates have not been created completely.
        # For example, num_relaxed != num_unrelaxed, need create more candidates...
        current_candidates = self.pop_manager._prepare_current_population(
            database=self.da,
            curr_gen=gen_num,
            builders=self.builders,
            operators=self.operators,
            candidate_groups=candidate_groups,
            random_state_getter=self.random_streams.snapshot,
            random_state_restorer=self.random_streams.restore,
        )

        # Validate candidate origins for the current generation
        candidate_groups = self.pop_manager._get_current_candidates(database=self.da, curr_gen=gen_num)
        self._print("candidate origin distribution after:")
        self._print("  " + "".join([f"{k:<8s}: {len(v):<4d}  " for k, v in candidate_groups.items()]))

        return current_candidates

    def get_workers(self, gen_info: Optional[GenerationInfo] = None) -> list:
        """Get all workers used by this expedition."""
        if gen_info is None:
            da = self.da if hasattr(self, "da") else GODB(self.db_path)
            gen_info = da.get_generation_info()

        assert self.worker is not None, "GA has not set its worker properly."
        potential = self.worker.runtime.provider_potential
        if hasattr(potential, "remove_loaded_models"):
            potential.remove_loaded_models()

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
        if gen_info.num > max_gen or gen_info.state == GenerationState.EXTINCTED:
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
                    "crossover": {"method": "periodic_cut_and_splice"},
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
            pbc=self.periodic,
            mic=self.periodic,
        )

        # For compatibility,
        for attr in [
            "blmin",
            "number_of_variable_cell_vectors",
            "cell_bounds",
            "test_dist_to_slab",
        ]:
            if hasattr(self.generator, attr):
                specific_params.update(**{attr: getattr(self.generator, attr)})
            else:
                ...

        # We may not overwrite operators' covalent_ratio setting.
        # specific_params.update(covalent_ratio=self.generator.covalent_ratio)

        # Standard operators use blmin and check minimum distances, while
        # the geometry-aware operators use bond_distance_dict and
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
                group_operators = self._parse_group_operators(g, g_op_dict, specific_params)
                self.operators[g] = group_operators
            else:
                ...

        return

    def _parse_group_operators(self, group: str, op_dict: dict, specific_params: dict):
        """Parse operators for a given group.

        Returns:
            A dict with comparing, pairing, and mutations.

        """
        # --- comparator
        comp_params = op_dict.get("comparator", None)
        if comp_params is not None:
            self._reject_population_owned_operator_keys(comp_params, f"operators.{group}.comparator")
            comp_specific = dict(
                specific_params,
                rng=self.random_streams.get(f"operator/{group}/comparator"),
            )
            comparing = instantiate_a_genetic_operator("comparator", comp_params, comp_specific)

            self._print("  --- comparator ---")
            self._print(f"  Use comparator {comparing.__class__.__name__}.")
        else:
            comparing = None

        # --- crossover
        crossover_params = op_dict.get("crossover", None)
        if crossover_params is not None:
            self._reject_population_owned_operator_keys(crossover_params, f"operators.{group}.crossover")
            crossover_specific = dict(
                specific_params,
                rng=self.random_streams.get(f"operator/{group}/crossover"),
            )
            pairing = instantiate_a_genetic_operator(
                "crossover",
                crossover_params,
                crossover_specific,
            )
            self._configure_fragment_policy(pairing, f"operators.{group}.crossover")
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
            for mutation_index, mut_params in enumerate(mutation_list):
                mut_params = copy.deepcopy(mut_params)
                self._reject_population_owned_operator_keys(
                    mut_params, f"operators.{group}.mutation[{mutation_index}]"
                )
                if "prob" in mut_params:
                    raise ValueError("Legacy mutation key 'prob' is not supported; use 'probability'.")
                prob = mut_params.pop("probability", 1.0)
                probs.append(prob)
                specific_params_ = copy.deepcopy(specific_params)
                specific_params_["rng"] = self.random_streams.get(
                    f"operator/{group}/mutation/{mutation_index}"
                )
                mut = instantiate_a_genetic_operator("mutation", mut_params, specific_params_)
                self._configure_fragment_policy(
                    mut, f"operators.{group}.mutation[{mutation_index}]"
                )
                assert "Mutation" in mut.descriptor, f"{mut} must have `Mutation` in its descriptor."
                mutations.append(mut)

            self._print("  --- mutations ---")
            # self._print(f"mutation probability: {self.pmut}")
            for mut, prob in zip(mutations, probs):
                self._print(f"  Use mutation {mut.descriptor} with prob {prob}.")
            mutations = OperationSelector(
                probs,
                mutations,
                rng=self.random_streams.get(f"operator/{group}/mutation_selector"),
            )
        else:
            mutations = OperationSelector(
                [], [], rng=self.random_streams.get(f"operator/{group}/mutation_selector")
            )

        return dict(comparing=comparing, pairing=pairing, mutations=mutations)

    @staticmethod
    def _reject_population_owned_operator_keys(params: Mapping, path: str) -> None:
        population_owned = {"pbc", "use_tags"}.intersection(params)
        if population_owned:
            migrations = {
                "pbc": "population.periodic",
                "use_tags": "population.preserve_fragments",
            }
            details = ", ".join(
                f"{key} -> {migrations[key]}" for key in sorted(population_owned)
            )
            raise ValueError(f"{path} contains population-owned keys: {details}.")

    def _configure_fragment_policy(self, operator, path: str) -> None:
        supports_fragments = getattr(operator, "supports_fragment_preservation", False)
        if self.preserve_fragments and not supports_fragments:
            raise ValueError(
                f"{path} uses {operator.__class__.__name__}, which cannot guarantee "
                "population.preserve_fragments=true."
            )
        if getattr(operator, "fragment_mode_configurable", False):
            operator.use_tags = self.preserve_fragments

    def _register_database(
        self,
    ):
        self._print("===== Population Info =====")
        content = "For generation > 0,\n"
        content += "{:>12s}  {:>12s}  {:>12s}  {:>8s}\n".format(
            "Reproduction", "Mutation", "Completion", "Total"
        )
        content += "{:>12d}  {:>12d}  {:>12d}  {:>8d}\n".format(
            self.pop_manager.gen_rep_size,
            self.pop_manager.gen_mut_size,
            self.pop_manager.gen_size - self.pop_manager.gen_rep_size - self.pop_manager.gen_mut_size,
            self.pop_manager.gen_size,
        )
        content += "Note: Reproduced structures mutate according to mutation_probability.\n"
        content += f"use_extinct: {self.pop_manager.use_extinct}\n"
        content += f"thanos: {self.pop_manager.extinct_callbacks}\n"
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

        # Evaluate the configured objective.
        if self.target == "energy":
            energy = atoms.get_potential_energy()
            atoms.info["key_value_pairs"]["raw_score"] = -energy
            atoms.info["key_value_pairs"]["target"] = energy
        elif self.target == "cohesive_energy":
            chemical_potentials = self.objective["chemical_potentials"]

            energy = atoms.get_potential_energy()
            cohesive_energy = energy - np.sum(
                [chemical_potentials[s] for s in atoms.get_chemical_symbols()]
            )
            atoms.info["key_value_pairs"]["raw_score"] = -cohesive_energy
            atoms.info["key_value_pairs"]["target"] = cohesive_energy
        elif self.target == "formation_energy":
            identity_stats = atoms.info.get("identity_stats", None)
            assert identity_stats is not None, (
                "Fail to compute `formation_energy` as no `identity_stats` is found in atoms.info."
            )
            chemical_potentials = self.objective["chemical_potentials"]

            energy = atoms.get_potential_energy()

            formation_energy = energy - np.sum(
                [chemical_potentials[k] * v for k, v in identity_stats.items()]
            )
            atoms.info["key_value_pairs"]["raw_score"] = -formation_energy
            atoms.info["key_value_pairs"]["target"] = formation_energy
        elif self.target == "reaction_energy":
            raise NotImplementedError()
        else:
            raise RuntimeError(f"Unknown target {self.target}...")

        return

    def as_dict(self) -> dict:
        """"""
        population = copy.deepcopy(self.ga_dict["population"])
        builders = {}
        for name, builder in self.builders.items():
            builder_config = copy.deepcopy(builder.as_dict())
            builder_config.pop("pbc", None)
            builder_config.pop("use_tags", None)
            builders[name] = builder_config
        population_prefix: dict[str, Any] = {
            "builders": builders
        }
        if self.reference_builder_name != "random":
            population_prefix["reference_builder"] = self.reference_builder_name
        population = dict(**population_prefix, **population)
        ga_dict = copy.deepcopy(self.ga_dict)
        ga_dict["population"] = population
        recipe = dict(
            random_seed=self.random_seed,
            **ga_dict,
        )
        engine_params = {
            "method": "genetic_algorithm",
            "recipe": recipe,
            "runtime": self.worker.as_dict(),
        }
        return copy.deepcopy(engine_params)
