from gdpx.workflow.session.node import NodeKind, WorkflowNode
from gdpx.workflow.session.once import OnceSession
from gdpx.workflow.session.operation import Operation
from gdpx.workflow.session.variable import Variable


class AddOne(Operation):
    def forward(self, value):
        super().forward()
        self.status = "finished"
        return value + 1


def test_session_executes_explicit_node_kinds(tmp_path):
    variable = Variable(2)
    operation = AddOne([variable])

    assert isinstance(variable, WorkflowNode)
    assert variable.node_kind is NodeKind.VARIABLE
    assert operation.node_kind is NodeKind.OPERATION

    session = OnceSession(tmp_path / "session")
    session.run(operation)

    assert operation.output == 3
    assert session.is_finished()
