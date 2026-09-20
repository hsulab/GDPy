"""Algorithm-independent generation progress and evaluation outcomes."""
from dataclasses import dataclass
from enum import Enum, auto


class GenerationState(Enum):
    BEG_OF_GEN = auto()
    MID_OF_GEN = auto()
    END_OF_GEN = auto()
    EXTINCTED = auto()


class EvaluationStatus(Enum):
    PENDING = auto()
    FINISHED = auto()


@dataclass(frozen=True)
class GenerationInfo:
    num: int
    state: GenerationState
    unrelaxed_confids: list[int]
    relaxed_confids: list[int]

    @property
    def num_unrelaxed(self):
        return len(self.unrelaxed_confids)

    @property
    def num_relaxed(self):
        return len(self.relaxed_confids)

    def converged(self, maximum_generation):
        return self.state is GenerationState.EXTINCTED or self.num > maximum_generation


def restore_generation_random_states(database, generation, streams):
    """Resume this production plan or continue the preceding completed plan."""
    plan = database.get_generation_plan(generation)
    if plan is None and generation > 0:
        plan = database.get_generation_plan(generation - 1)
    if plan and "random_states" in plan:
        streams.restore(plan["random_states"])
