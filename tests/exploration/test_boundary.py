from gdpx.exploration import Exploration, ExplorationStrategy, Proposal


class Strategy(ExplorationStrategy):
    def __init__(self):
        self.result = None

    def propose(self):
        return Proposal("candidate")

    def observe(self, proposal, result):
        self.result = (proposal.inputs, result)

    def converged(self):
        return self.result is not None


class Service:
    def submit(self, runtime, inputs):
        return (runtime, inputs)

    def retrieve(self, handle):
        return "evaluated:" + handle[1]


def test_exploration_chooses_tasks_but_delegates_execution():
    strategy = Strategy()
    exploration = Exploration(strategy, Service(), runtime="runtime")

    assert exploration.step() == "evaluated:candidate"
    assert strategy.result == ("candidate", "evaluated:candidate")
    assert strategy.converged()

