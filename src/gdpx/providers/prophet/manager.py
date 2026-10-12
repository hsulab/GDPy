"""Optional Prophet ASE adapter for local Kairos model checkpoints."""

import copy

from ..manager_base import BasePotentialManager
from ..potential_utils import build_a_committee_calculator, canonicalise_input_models


def canonicalise_prophet_models(model):
    """Validate checkpoint inputs and resolve paths for scheduler execution."""
    models = model if isinstance(model, list) else [model]
    if not models or any(not isinstance(m, str) or not m.strip() for m in models):
        raise ValueError("Prophet model must be a non-empty checkpoint path or list of paths.")
    return canonicalise_input_models(models)


class ProphetManager(BasePotentialManager):
    name = "prophet"
    implemented_backends = ("ase",)
    valid_combinations = (("ase", "ase"),)

    def register_calculator(self, calc_params: dict, *args, **kwargs):
        params = copy.deepcopy(calc_params)
        super().register_calculator(calc_params=params, *args, **kwargs)
        models = canonicalise_prophet_models(params.pop("model", None))
        self.calc_params.update(model=models)

        try:
            import torch
            from prophet import KairosCalculator
        except ImportError as exc:
            raise ImportError(
                "Prophet requires a compatible prophet-mlip and PyTorch installation. "
                "Install GDPy's Prophet extra with: python -m pip install '.[prophet]'"
            ) from exc

        if params.get("device") is None:
            params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        # The upstream default requires CUDA kernels even on CPU.
        params.setdefault("use_kernel", False)
        uncertainty = params.pop("estimate_uncertainty", False)
        self.calc = build_a_committee_calculator(
            KairosCalculator,
            params_list=[dict(params, model_path=model) for model in models],
            estimate_uncertainty=uncertainty,
        )
