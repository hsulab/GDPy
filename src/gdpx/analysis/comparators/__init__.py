from gdpx.core.registry import Registry

REGISTER = Registry("comparator")

REGISTER.register_lazy("cartesian_coordinate", "gdpx.analysis.comparators.cartesian", "CartesianCoordinateComparator")
REGISTER.register_lazy("coordination", "gdpx.analysis.comparators.coordination", "CoordinationComparator")
REGISTER.register_lazy("graph", "gdpx.analysis.comparators.graph", "GraphComparator")
REGISTER.register_lazy("single_point", "gdpx.analysis.comparators.singlepoint", "SinglePointComparator")
REGISTER.register_lazy("reaction", "gdpx.analysis.comparators.reaction", "ReactionComparator")
REGISTER.register_lazy("oganov_fingerprint", "gdpx.analysis.comparators.oganov_fingerprint", "OganovFingerprintComparator")
REGISTER.register_lazy("permutation_rmsd", "gdpx.analysis.comparators.rmsd", "PermutationRMSDComparator")
REGISTER.register_lazy("energy", "gdpx.analysis.comparators.energy", "EnergyComparator")
REGISTER.register_lazy("combined", "gdpx.analysis.comparators.combined", "CombinedComparator")
REGISTER.register_lazy("energetic_oganov_fingerprint", "gdpx.analysis.comparators.presets", "energetic_oganov_fingerprint")

from .comparator import BaseComparator


def create_comparator(config):
    import copy
    from collections.abc import Mapping

    if isinstance(config, (list, tuple)):
        return REGISTER["combined"](comparators=config)
    if isinstance(config, BaseComparator):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("Comparator configuration must be a mapping, list, or BaseComparator.")
    parameters = copy.deepcopy(dict(config))
    method = parameters.pop("method")
    return REGISTER[method](**parameters)


__all__ = ["BaseComparator", "REGISTER", "create_comparator"]
