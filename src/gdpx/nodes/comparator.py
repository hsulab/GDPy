from gdpx.core.register import registers
from gdpx.session.operation import Operation
from gdpx.session.variable import DummyVariable, Variable


@registers.variable.register
class ComparatorVariable(Variable):
    def __init__(self, directory="./", *args, **kwargs):
        """"""
        method = kwargs.pop("method", None)
        comparator = registers.create("comparator", method, convert_name=False, **kwargs)
        super().__init__(initial_value=comparator, directory=directory)

        return


@registers.operation.register
class compare(Operation):
    status = "finished"  # Always finished since it is not time-consuming

    def __init__(self, reference, prediction=DummyVariable(), comparator=DummyVariable(), directory="./") -> None:
        """"""
        super().__init__(input_nodes=[reference, prediction, comparator], directory=directory)

        return

    def forward(self, reference, prediction, comparator):
        """"""
        super().forward()

        comparator.directory = self.directory
        comparator.run(prediction, reference)

        return
