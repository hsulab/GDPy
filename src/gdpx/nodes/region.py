from gdpx.core.register import registers
from gdpx.session.variable import Variable


class RegionVariable(Variable):
    def __init__(self, directory="./", *args, **kwargs):
        """"""
        name = kwargs.pop("method", "auto")
        region = registers.create("region", name, convert_name=True, **kwargs)

        super().__init__(initial_value=region, directory=directory)

        return
