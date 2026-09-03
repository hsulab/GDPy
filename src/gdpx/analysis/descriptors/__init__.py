from gdpx.core.registry import Registry

REGISTER = Registry("describer")

from .spc import SpcDescriber

REGISTER.register("spc")(SpcDescriber)

from .coordinate import CoordinateDescriber

REGISTER.register("coordinate")(CoordinateDescriber)

from .coordination import CoordinationDescriber

REGISTER.register("coordination")(CoordinationDescriber)

from .connectivity import ConnectivityDescriber

REGISTER.register("connectivity")(ConnectivityDescriber)

from .dissociative import DissociativeDescriber

REGISTER.register("dissociative")(DissociativeDescriber)

from .formation_energy import FormationEnergyDescriber

REGISTER.register("formation_energy")(FormationEnergyDescriber)

REGISTER.register_lazy("soap", "gdpx.analysis.descriptors.soap", "SoapDescriber")

from .cluster import ClusterDescriber

REGISTER.register("cluster")(ClusterDescriber)

from .colvar import ColvarDescriber

REGISTER.register("colvar")(ColvarDescriber)

from .describer import BaseDescriber


def create_describer(config):
    import copy
    from collections.abc import Mapping

    if isinstance(config, BaseDescriber):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Describer configuration must be a mapping or BaseDescriber.")
    parameters = copy.deepcopy(dict(config))
    name = parameters.pop("name", "soap")
    return REGISTER[name](**parameters)


__all__ = ["BaseDescriber", "REGISTER", "create_describer"]
