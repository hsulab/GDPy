from gdpx.workflow.session.registry import workflow_registers as registers
from gdpx.structures.regions.factory import create_region
from gdpx.workflow.session.variable import Variable


@registers.variable.register
class RegionVariable(Variable):
    def __init__(self, directory="./", *args, **kwargs):
        """"""
        region = create_region(kwargs)

        super().__init__(initial_value=region, directory=directory)

        return
