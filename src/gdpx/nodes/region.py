from gdpx.session.registry import workflow_registers as registers
from gdpx.factory.region import create_region
from gdpx.session.variable import Variable


@registers.variable.register
class RegionVariable(Variable):
    def __init__(self, directory="./", *args, **kwargs):
        """"""
        region = create_region(kwargs)

        super().__init__(initial_value=region, directory=directory)

        return
