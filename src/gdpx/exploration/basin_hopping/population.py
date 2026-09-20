"""Basin-hopping chain-start policy over the shared candidate pool."""
from ..population.config import PopulationConfig
from ..population.comparators import create_population_comparator
from ..population.pool import CandidatePool


class HoppingPopulation(PopulationConfig):
    def __init__(self, params, streams):
        super().__init__(params, rng=streams.get("population"))
        unsupported = {"reproduction", "mutation", "completion"} & params["generation"].keys()
        if unsupported:
            raise ValueError("BH generation does not accept GA policies: " + ", ".join(sorted(unsupported)))
        self.initialise_builders(params, streams)
        self.comparator = create_population_comparator(
            self.comparator_config, self.periodic, streams.get("population/comparator"))

    def get_current_generation(self, database, rng=None, with_history=True, use_extinct=False):
        pool = CandidatePool(database, self.retained_size, self.comparator, use_extinct)
        return pool.select(self.gen_size, self.rng if rng is None else rng, with_history)
