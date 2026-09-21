"""Optional TACE ASE adapter for checkpoints and foundation models."""

import copy

from ..manager_base import BasePotentialManager
from ..potential_utils import build_a_committee_calculator, canonicalise_input_models


def canonicalise_tace_models(model, foundation_names=()):
    """Preserve registered aliases and resolve checkpoint paths without downloading."""
    models = model if isinstance(model, list) else [model]
    if not models or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ValueError("TACE model must be a non-empty name, checkpoint path, or list of these.")
    names = set(foundation_names)
    return [m if m in names else canonicalise_input_models(m)[0] for m in models]


class TaceManager(BasePotentialManager):
    name = "tace"
    implemented_backends = ("ase",)
    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params: dict, *args, **kwargs):
        params = copy.deepcopy(calc_params)
        super().register_calculator(calc_params=params, *args, **kwargs)
        try:
            import torch
            from tace.foundations import tace_foundations
            from tace.interface.ase import TACEAseCalc
        except ImportError as exc:
            raise ImportError(
                "TACE requires a compatible TACE and PyTorch installation. "
                "Install GDPy's TACE extra with: python -m pip install '.[tace]'"
            ) from exc

        # Iteration lists names only: Mapping membership would trigger downloads.
        names = set(tace_foundations)
        models = canonicalise_tace_models(params.pop("model", None), names)
        self.calc_params.update(model=models)
        precision = params.pop("precision", "float32")
        params.setdefault("dtype", precision)
        params.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        uncertainty = params.pop("estimate_uncertainty", False)
        resolved = [str(tace_foundations[m]) if m in names else m for m in models]
        self.calc = build_a_committee_calculator(
            TACEAseCalc,
            params_list=[dict(params, model=m) for m in resolved],
            estimate_uncertainty=uncertainty,
        )
