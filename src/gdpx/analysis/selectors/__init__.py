from gdpx.core.registry import Registry

REGISTER = Registry("selector")

from .compare import CompareSelector

REGISTER.register("compare")(CompareSelector)

from .interval import IntervalSelector

REGISTER.register("interval")(IntervalSelector)

from .invariant import InvariantSelector

REGISTER.register("invariant")(InvariantSelector)

from .locate import LocateSelector

REGISTER.register("locate")(LocateSelector)

from .property import PropertySelector

REGISTER.register("property")(PropertySelector)

from .random import RandomSelector

REGISTER.register("random")(RandomSelector)

from .scf import ScfSelector

REGISTER.register("scf")(ScfSelector)

from .sinfo import StructureInfoSelector

REGISTER.register("structure_info")(StructureInfoSelector)

REGISTER.register_lazy("descriptor", "gdpx.analysis.selectors.descriptor", "DescriptorSelector")

from .selector import BaseSelector


def create_selector(config):
    import copy
    from collections.abc import Mapping

    if isinstance(config, Mapping) and "selection" in config:
        config = config["selection"]
    if isinstance(config, list):
        selectors = [create_selector(item) for item in copy.deepcopy(config)]
        if not selectors:
            raise ValueError("At least one selector definition is required.")
        if len(selectors) == 1:
            return selectors[0]
        from .composition import ComposedSelector

        return ComposedSelector(selectors)
    if isinstance(config, BaseSelector):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Selector configuration must be a mapping or BaseSelector.")
    parameters = copy.deepcopy(dict(config))
    method = parameters.pop("method", "random")
    return REGISTER[method](**parameters)


__all__ = ["BaseSelector", "REGISTER", "create_selector"]
